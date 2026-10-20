#include "model/kflash_budget.h"

#include "hub/hub.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

namespace ni {

namespace {
// What the cache's cost table holds, saved with it: a table measured as anything else (the verify alone, or the
// host's clock, as the server and the Nemotron loop measured before) is measured again; the calibration is kept.
constexpr const char* kCostOf = "cycle GPU ms";
}  // namespace

KfBudget::KfBudget(std::string key) : key_(std::move(key)) {
  for (int d = 0; d < kD; ++d) {
    alpha_[d] = 0.0f;
    beta_[d] = 1.0f;
  }
  if (const char* v = getenv("NI_KFLASH_FORCE"); v && *v) force_env_ = std::clamp(atoi(v), 0, kf::kMaxRows);
  load();
}

KfBudget::~KfBudget() { save(); }

void KfBudget::set_wide(bool on) {
  std::lock_guard<std::mutex> lk(mu_);
  wide_ = on;
  if (!on) return;
  if (!wide_saved_ && cost_[8] > 0 && ncost8n_ == 0) {  // a cache of 8-row verifies only: that is the base's cost
    cost8n_ = cost_[8];
    ncost8n_ = 2;
    cost_[8] = 0;
    ncost_[8] = 0;
  }
  // the log model from the cached costs (ratios to the 8-row verify's); this process's narrow cycles set the base
  if (cost8n_ > 0) {
    lb_ = std::log(cost8n_);
    for (int n = 1; n < kD; ++n) lq_[n] = cost_[n] > 0 ? std::log(cost_[n]) - lb_ : 0.0;
  }
}

void KfBudget::wide_costs() {
  cost8n_ = std::exp(lb_);
  for (int n = 1; n < kD; ++n) cost_[n] = ncost_[n] > 0 ? std::exp(lb_ + lq_[n]) : 0.0;
}

void KfBudget::set_rows(uint32_t allowed, int max_rows) {
  max_rows_ = std::clamp(max_rows, 1, kf::kMaxRows);
  allowed_ = allowed & ((max_rows_ == 32 ? 0u : 1u << max_rows_) - 1u);
  if (!allowed_) allowed_ = 1u << (max_rows_ - 1);
}

float KfBudget::logit_q(float lq) {
  const double q = std::min(std::exp(double(lq)), 0.999999);
  return float(double(lq) - std::log1p(-q));
}

// L2-regularized logistic regression of y on (1, x) toward the identity (alpha 0, beta 1): Newton steps from the
// current values. The prior weighs about as much as four records; beta stays positive (the ranking of siblings is
// the path model's).
void KfBudget::fit(const std::vector<float>& x, const std::vector<uint8_t>& y, float& alpha, float& beta, int iters) {
  constexpr double lam = 1.0;
  double a = alpha, b = beta;
  for (int it = 0; it < iters; ++it) {
    double ga = lam * a, gb = lam * (b - 1.0), haa = lam, hab = 0.0, hbb = lam;
    for (size_t i = 0; i < x.size(); ++i) {
      const double xi = x[i], z = a + b * xi, p = 1.0 / (1.0 + std::exp(-z)), w = p * (1.0 - p), r = p - y[i];
      ga += r;
      gb += r * xi;
      haa += w;
      hab += w * xi;
      hbb += w * xi * xi;
    }
    const double det = haa * hbb - hab * hab;
    if (!(det > 1e-12)) break;
    const double da = (hbb * ga - hab * gb) / det, db = (haa * gb - hab * ga) / det;
    a = std::clamp(a - da, -8.0, 8.0);
    b = std::clamp(b - db, 0.1, 4.0);
    if (std::fabs(da) + std::fabs(db) < 1e-4) break;
  }
  alpha = float(a);
  beta = float(b);
}

// Wide verifies: the expected tokens per ms of the per-cycle choice among the allowed counts at their (wide) costs
// against 8 rows at the narrow verify's cost, over the recent cycles' trees.
void KfBudget::eval_wide() {
  const size_t nc = ehist_.size() / 3;
  if (!nc || cost8n_ <= 0) {
    wide_pays_ = true;
    return;
  }
  double e8 = 0, ew = 0, tw = 0;
  for (size_t c = 0; c < nc; ++c) {
    const float* e = &ehist_[3 * c];
    e8 += e[0];
    double best = -1, be = 0, bt = 1;
    const int counts[3] = {8, 16, 32};
    for (int i = 0; i < 3; ++i) {
      const int n = counts[i];
      if (n > max_rows_ || !((allowed_ >> (n - 1)) & 1u) || cost_[n] <= 0 || e[i] <= 0) continue;
      if (e[i] / cost_[n] > best) best = e[i] / cost_[n], be = e[i], bt = cost_[n];
    }
    if (best < 0) be = e[0], bt = cost8n_;
    ew += be;
    tw += bt;
  }
  wide_pays_ = ew / tw > e8 / (double(nc) * cost8n_);
}

void KfBudget::observe(const int32_t* t, double ms) {
  std::lock_guard<std::mutex> lk(mu_);
  // kf_accept's copies (the next cycle's builder may already have rewritten the descriptor's own words); a descriptor
  // no kf_accept wrote (tests): its own words
  const bool obs = t[kf::kObs] > 0;
  const int n = std::clamp(int(obs ? t[kf::kObs] : t[kf::kN]), 1, kf::kMaxRows);
  const int cap0 = obs ? t[kf::kObs + 1] : t[kf::kCap], cap = cap0 > 0 ? cap0 : max_rows_;
  const int nodes = obs ? t[kf::kObs + 2] : t[kf::kNodes];
  const bool narrow = wide_ && max_rows_ > 8 && cap <= 8;  // a wide build verified by the 8-row verify
  ++cycles_;
  rows_sum_ += n;
  ++chosen_[n];
  if (wide_ && max_rows_ > 8) {
    // wide verifies, log cost = lb_ + lq_[n]: a narrow cycle moves the base (this process's first one sets it); a
    // wide one fits its count's ratio (its first measurement exactly) and nudges the base by a quarter of that step,
    // so the base follows the context in wide stretches too
    const double y = std::log(std::max(ms, 1e-3));
    if (narrow) {
      lb_ += (y - lb_) / double(std::min(nb_ + 1, 16));
      ++nb_;
      ++ncost8n_;
      last8n_ = cycles_;
      ++narrow_cycles_;
      if (refresh_ < 0) refresh_ = 0;
    } else {
      if (ncost_[n] == 0 || (nb_ == 0 && ncost8n_ == 0)) {
        if (ncost8n_ == 0 && nb_ == 0 && n == 8) lb_ = y, lq_[n] = 0.0;  // no 8-row verify yet: the base is wide 8 rows
        else lq_[n] = y - lb_;
      } else {
        const double e = y - (lb_ + lq_[n]), eta = 1.0 / double(std::min(ncost_[n] + 1, 16));
        lq_[n] += eta * e;
        lb_ += 0.25 * eta * e;
      }
      ++ncost_[n];
      last_[n] = cycles_;
      if (n == refresh_) refresh_ = 0;
    }
    wide_costs();
  } else {
    // the cost of n rows: a running mean over the last ~16 cycles that verified n rows
    if (ncost_[n] == 0) cost_[n] = ms;
    else cost_[n] += (ms - cost_[n]) / double(std::min(ncost_[n] + 1, 16));
    ++ncost_[n];
    last_[n] = cycles_;
    if (n == refresh_) refresh_ = 0;
  }
  if (cycles_ % kRefresh == 0 && (__builtin_popcount(allowed_) > 1 || wide_)) {  // the count measured longest ago
    int stale = 0, at = cycles_;
    for (int m = 1; m <= max_rows_; ++m)
      if (((allowed_ >> (m - 1)) & 1u) && last_[m] < at) stale = m, at = last_[m];
    if (wide_ && max_rows_ > 8 && last8n_ < at) stale = -1, at = last8n_;
    if (stale && cycles_ - at >= kRefresh) refresh_ = stale;
  }
  // wide verifies: the tree's E(8), E(16), E(32) (a build of max_rows nodes), the capacity decision every 32 cycles
  if (wide_ && max_rows_ > 8 && nodes >= std::min(max_rows_, 32)) {  // a whole build (capacity 8 builds 8 nodes mostly)
    float e[3] = {0, 0, 0};
    const int counts[3] = {8, 16, 32};
    for (int i = 0; i < 3; ++i)
      if (obs) memcpy(&e[i], &t[kf::kObs + 3 + i], 4);
      else if (counts[i] <= nodes) memcpy(&e[i], &t[kf::kE + counts[i]], 4);
    if (ehist_.size() < size_t(3 * kEWin)) {
      ehist_.insert(ehist_.end(), e, e + 3);
    } else {
      memcpy(&ehist_[3 * ehead_], e, sizeof e);
      ehead_ = (ehead_ + 1) % size_t(kEWin);
    }
    if (++since_eval_ >= 32) {
      since_eval_ = 0;
      eval_wide();
    }
  }
  const int nrec = std::clamp(int(t[kf::kNrec]), 0, kf::kMaxRows);
  for (int i = 0; i < nrec; ++i) {
    const int d = t[kf::kRecDep + i];
    if (d < 1 || d > kf::kMaxRows) continue;
    float lq;
    memcpy(&lq, &t[kf::kRecLq + i], 4);
    const float x = logit_q(lq);
    const uint8_t y = t[kf::kRecHit + i] != 0;
    if (x_[d].size() < size_t(kWin)) {
      x_[d].push_back(x);
      y_[d].push_back(y);
    } else {
      x_[d][head_[d]] = x;
      y_[d][head_[d]] = y;
      head_[d] = (head_[d] + 1) % size_t(kWin);
    }
  }
  if (++since_fit_ >= 16) {
    since_fit_ = 0;
    for (int d = 1; d < kD; ++d)
      if (x_[d].size() >= 32) fit(x_[d], y_[d], alpha_[d], beta_[d], 4);
  }
}

void KfBudget::write(KfParams& p) {
  std::lock_guard<std::mutex> lk(mu_);
  for (int d = 0; d < kD; ++d) {
    p.alpha[d] = alpha_[d];
    p.beta[d] = beta_[d];
  }
  // exploration: every allowed row count measured twice before the table decides (wide verifies: the narrow 8 rows
  // too), and the refreshes
  const bool wide = wide_ && max_rows_ > 8;
  bool narrow = wide && nb_ == 0;  // wide verifies: the base first (this process's 8-row verify cost), then the counts
  int force = 0;
  if (!narrow)
    for (int n = 1; n <= max_rows_; ++n)
      if (((allowed_ >> (n - 1)) & 1u) && ncost_[n] < 2) {
        force = n;
        break;
      }
  if (!force && wide && ncost8n_ < 2) narrow = true;
  if (!force && !narrow) {
    if (refresh_ > 0) force = refresh_;
    else if (refresh_ < 0) narrow = true;
  }
  // wide verifies: capacity 8 unless the per-cycle choice among the wider counts pays (eval_wide)
  if (wide && !force && !narrow && !wide_pays_) narrow = true;
  if (force_env_) {
    force = std::min(force_env_, max_rows_);
    narrow = false;
  }
  p.force_n = force;
  p.allowed = allowed_;
  p.max_rows = max_rows_;
  p.cap = 0;
  if (narrow) {  // the verify runs 8 rows; one cycle in kStatsEvery still builds max_rows nodes (their E(n) feed
                 // eval_wide), the others 8 (a 32-node build costs the 4B's drafting pass ~0.04 ms)
    p.cap = 8;
    p.allowed = allowed_ & 0xffu ? allowed_ & 0xffu : 1u << 7;
    p.force_n = 0;
    if (narrow_writes_++ % kStatsEvery) p.max_rows = 8;
  }
  p.cost[0] = 1.0f;
  for (int n = 1; n < kD; ++n) p.cost[n] = cost_[n] > 0 ? float(cost_[n]) : 1.0f;
}

std::string KfBudget::summary() const {
  std::lock_guard<std::mutex> lk(mu_);
  char b[1024];
  int o = snprintf(b, sizeof b, "KFlash: %d cycles, %.2f rows per cycle (cycles by rows", cycles_,
                   cycles_ ? rows_sum_ / cycles_ : 0.0);
  for (int n = 1; n <= kf::kMaxRows && o < int(sizeof b) - 16; ++n)
    if (chosen_[n]) o += snprintf(b + o, sizeof b - o, " %d:%d", n, chosen_[n]);
  if (wide_ && max_rows_ > 8) o += snprintf(b + o, sizeof b - o, "; %d at capacity 8", narrow_cycles_);
  o += snprintf(b + o, sizeof b - o, "); cost ms by rows");
  for (int n = 1; n <= max_rows_ && o < int(sizeof b) - 16; ++n)
    if ((allowed_ >> (n - 1)) & 1u) o += snprintf(b + o, sizeof b - o, " %d:%.2f", n, cost_[n]);
  if (wide_ && max_rows_ > 8) o += snprintf(b + o, sizeof b - o, " (8 at capacity 8: %.2f)", cost8n_);
  o += snprintf(b + o, sizeof b - o, "; calibration (alpha, beta) by depth");
  for (int d = 1; d <= std::min(max_rows_, 8) && o < int(sizeof b) - 24; ++d)
    o += snprintf(b + o, sizeof b - o, " (%.2f, %.2f)", alpha_[d], beta_[d]);
  return b;
}

void KfBudget::load() {
  if (key_.empty()) return;
  std::ifstream f(hub::home_dir() + "/kflash/" + key_ + ".json");
  if (!f) return;
  try {
    const nlohmann::json j = nlohmann::json::parse(f);
    const auto& c = j.at("cost");
    const auto& a = j.at("alpha");
    const auto& bb = j.at("beta");
    if (j.value("cost_of", "") == kCostOf) {
      for (int n = 1; n < kD && n < int(c.size()); ++n)
        if (c[size_t(n)].get<double>() > 0) cost_[n] = c[size_t(n)].get<double>(), ncost_[n] = 2;
      // wide verifies' extras (a cache a run without them wrote: its 8 rows are the 8-row verify's cost, set_wide)
      wide_saved_ = j.value("wide", false);
      if (wide_saved_) {
        if (j.value("cost_narrow8", 0.0) > 0) cost8n_ = j["cost_narrow8"].get<double>(), ncost8n_ = 2;
        wide_pays_ = j.value("wide_pays", true);  // the last decision, until this process's trees decide again
      }
    }
    for (int d = 1; d < kD && d < int(a.size()) && d < int(bb.size()); ++d) {
      alpha_[d] = a[size_t(d)].get<float>();
      beta_[d] = bb[size_t(d)].get<float>();
    }
  } catch (...) {
    logi("KFlash: ignoring an unreadable cache %s", key_.c_str());
  }
}

void KfBudget::save(int fresh) {
  std::lock_guard<std::mutex> lk(mu_);
  if (key_.empty() || cycles_ < 32 || cycles_ - saved_ < std::max(fresh, 1)) return;
  const std::string dir = hub::home_dir() + "/kflash";
  mkdir(dir.c_str(), 0755);
  nlohmann::json j;
  j["cost_of"] = kCostOf;
  j["cost"] = std::vector<double>(cost_, cost_ + kD);
  if (wide_ && max_rows_ > 8) {
    j["wide"] = true;
    j["cost_narrow8"] = cost8n_;
    j["wide_pays"] = wide_pays_;
  }
  j["alpha"] = std::vector<float>(alpha_, alpha_ + kD);
  j["beta"] = std::vector<float>(beta_, beta_ + kD);
  // written whole beside it, then renamed over it: a process ended mid-write (kill -9 after a stop signal), or another
  // tool saving the same key at once, leaves one whole file
  const std::string path = dir + "/" + key_ + ".json", tmp = path + "." + std::to_string(getpid()) + ".tmp";
  {
    std::ofstream f(tmp);
    f << j.dump() << "\n";
    f.close();
    if (!f) {
      unlink(tmp.c_str());
      logi("KFlash: could not write the budget's cache %s", tmp.c_str());
      return;
    }
  }
  if (rename(tmp.c_str(), path.c_str()) != 0) {
    unlink(tmp.c_str());
    logi("KFlash: could not write the budget's cache %s", path.c_str());
    return;
  }
  saved_ = cycles_;
  logi("KFlash: budget saved (%d cycles) to %s", cycles_, path.c_str());
}

}  // namespace ni

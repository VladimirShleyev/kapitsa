#include "model/kflash_budget.h"

#include "hub/hub.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sys/stat.h>

namespace ni {

KfBudget::KfBudget(std::string key) : key_(std::move(key)) {
  for (int d = 0; d < kD; ++d) {
    alpha_[d] = 0.0f;
    beta_[d] = 1.0f;
  }
  load();
}

KfBudget::~KfBudget() { save(); }

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

void KfBudget::observe(const int32_t* t, double ms) {
  const int n = std::clamp(int(t[kf::kN]), 1, kf::kMaxRows);
  ++cycles_;
  rows_sum_ += n;
  // the cost of n rows: a running mean over the last ~16 cycles that verified n rows
  if (ncost_[n] == 0) cost_[n] = ms;
  else cost_[n] += (ms - cost_[n]) / double(std::min(ncost_[n] + 1, 16));
  ++ncost_[n];
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

void KfBudget::write(KfParams& p) const {
  for (int d = 0; d < kD; ++d) {
    p.alpha[d] = alpha_[d];
    p.beta[d] = beta_[d];
  }
  // exploration: every allowed row count measured twice before the table decides
  int force = 0;
  for (int n = 1; n <= max_rows_; ++n)
    if (((allowed_ >> (n - 1)) & 1u) && ncost_[n] < 2) {
      force = n;
      break;
    }
  p.force_n = force;
  p.allowed = allowed_;
  p.max_rows = max_rows_;
  p.pad = 0;
  p.cost[0] = 1.0f;
  for (int n = 1; n < kD; ++n) p.cost[n] = cost_[n] > 0 ? float(cost_[n]) : 1.0f;
}

std::string KfBudget::summary() const {
  char b[1024];
  int o = snprintf(b, sizeof b, "KFlash: %d cycles, %.2f rows per cycle; cost ms by rows", cycles_,
                   cycles_ ? rows_sum_ / cycles_ : 0.0);
  for (int n = 1; n <= max_rows_ && o < int(sizeof b) - 16; ++n)
    if ((allowed_ >> (n - 1)) & 1u) o += snprintf(b + o, sizeof b - o, " %d:%.2f", n, cost_[n]);
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
    for (int n = 1; n < kD && n < int(c.size()); ++n)
      if (c[size_t(n)].get<double>() > 0) cost_[n] = c[size_t(n)].get<double>(), ncost_[n] = 2;
    for (int d = 1; d < kD && d < int(a.size()) && d < int(bb.size()); ++d) {
      alpha_[d] = a[size_t(d)].get<float>();
      beta_[d] = bb[size_t(d)].get<float>();
    }
  } catch (...) {
    logi("KFlash: ignoring an unreadable cache %s", key_.c_str());
  }
}

void KfBudget::save() const {
  if (key_.empty() || cycles_ < 32) return;
  const std::string dir = hub::home_dir() + "/kflash";
  mkdir(dir.c_str(), 0755);
  nlohmann::json j;
  j["cost"] = std::vector<double>(cost_, cost_ + kD);
  j["alpha"] = std::vector<float>(alpha_, alpha_ + kD);
  j["beta"] = std::vector<float>(beta_, beta_ + kD);
  std::ofstream f(dir + "/" + key_ + ".json");
  f << j.dump() << "\n";
}

}  // namespace ni

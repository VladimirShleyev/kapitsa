// S1: exactness of the on-GPU acceptance on synthetic inputs,
// no model needed. (Cases are numbered in order: the chain's, then the tree's.)
//   dflash_accept_test [trials=40000] [K=3]
// 1. dfl_dist_rows against a CPU transcription of the plain sampler's distribution (sample_stage2_rows:
//    top-k ordered (value desc, id asc), top-p on the untempered probabilities, tempered on the kept
//    prefix) on random candidate lists, including ties.
// 2. The speculative loop on prefix-conditional synthetic p(.|prefix) and q(.|prefix): chains drawn from
//    q on the host, dfl_accept on the GPU, a second cycle from the emitted token when only one token came
//    out; the joint law of the first two emitted tokens is tested against p(t1) p(t2|t1) (chi-square,
//    cells with expected count < 5 pooled), in several regimes: random p and q, q = p, q disjoint from p
//    (always rejected: residual = p), point masses, p(d) = 0 for some proposals, K = 1..7, and with the
//    dynamic verify depth (dfl_depth before dfl_accept): its exact T > 0 rules (0, and 2, the earlier one) must
//    pass; the rule that also uses the proposal's own confidence (rule 1, not used by the engine at T > 0) must fail in the depth-sensitive
//    regime (q = p at the first proposal, q = 0.9 / 0.1 vs p = 1/2 at the second: that rule emits 0.55 / 0.45).
// 3. Tree drafting (dense targets): dfl_tree builds a best-first tree from a synthetic lattice (random, or aligned with
//    the target's likely path; score temperatures vary), the target's rows of the tree's
//    paths are sampled by sample_stage2_rows_tree (top-k, top-p, temperature as the engine's sampler, each row with
//    the uniform of its token's position), dfl_accept_tree walks them; a second cycle runs when one token came out.
//    The joint law of the first two tokens is tested as in 2, and every trial's two tokens must equal plain decoding's
//    with the same position-keyed uniforms (the sampler transcribed on the CPU); T = 0: equal to the argmax decoding.
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <vector>

using namespace ni;

namespace {

// ---- chi-square p-value (regularized upper incomplete gamma) ----
double gamma_q(double a, double x) {
  if (x <= 0) return 1.0;
  const double gln = std::lgamma(a);
  if (x < a + 1) {  // series for P
    double ap = a, sum = 1.0 / a, del = sum;
    for (int n = 0; n < 1000; ++n) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (std::fabs(del) < std::fabs(sum) * 1e-14) break;
    }
    return 1.0 - sum * std::exp(-x + a * std::log(x) - gln);
  }
  double b = x + 1 - a, c = 1e300, d = 1 / b, h = d;  // continued fraction for Q
  for (int i = 1; i < 1000; ++i) {
    const double an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (std::fabs(d) < 1e-300) d = 1e-300;
    c = b + an / c;
    if (std::fabs(c) < 1e-300) c = 1e-300;
    d = 1 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1) < 1e-14) break;
  }
  return std::exp(-x + a * std::log(x) - gln) * h;
}

// chi-square of observed counts vs expected probabilities (pooling cells with expected < 5)
void chi2(const std::map<long, long>& obs, const std::map<long, double>& expp, long n, double* stat, int* dof, double* pval,
          long* outside) {
  double pool_o = 0, pool_e = 0, s = 0;
  int cells = 0;
  *outside = 0;
  for (auto& [k, c] : obs)
    if (!expp.count(k) || expp.at(k) <= 0) *outside += c;
  for (auto& [k, pr] : expp) {
    const double e = pr * n;
    const double o = obs.count(k) ? double(obs.at(k)) : 0.0;
    if (e < 5) {
      pool_o += o;
      pool_e += e;
      continue;
    }
    s += (o - e) * (o - e) / e;
    ++cells;
  }
  if (pool_e > 0) {
    if (pool_e >= 5 || cells == 0) {
      s += (pool_o - pool_e) * (pool_o - pool_e) / pool_e;
      ++cells;
    } else {
      s += (pool_o - pool_e) * (pool_o - pool_e) / std::max(pool_e, 1e-9);  // tiny tail: still counted
      ++cells;
    }
  }
  *stat = s;
  *dof = std::max(1, cells - 1);
  *pval = gamma_q(*dof / 2.0, s / 2.0);
}

uint64_t mix(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

// ---- synthetic conditional distributions ----
constexpr int kV = 24;       // token universe: ids kBase + 0 .. kBase + kV - 1
constexpr int kBase = 1000;
struct Dist {
  std::vector<int> id;
  std::vector<float> p;
};
enum Regime { kRandom, kEqual, kDisjoint, kPoint, kMixed, kBias };
const char* regime_name(int r) {
  static const char* n[] = {"random p, random q", "q = p", "q disjoint from p", "point masses", "mixed (p(d)=0, flat q)",
                            "depth-sensitive"};
  return n[r];
}

uint64_t key_of(const std::vector<int>& prefix) {
  uint64_t h = 0x9e3779b97f4a7c15ULL;
  for (int t : prefix) h = mix(h ^ uint64_t(t + 1));
  return h;
}

// p(.|prefix): support of 1..10 tokens of the universe, probabilities sorted descending (like a top-k list)
Dist p_of(const std::vector<int>& prefix, int regime) {
  std::mt19937_64 r(key_of(prefix) ^ 0x1111);
  Dist d;
  if (regime == kBias) {
    // every position: two tokens at 1/2 (which two depends on the prefix); q (below) equals p at the
    // first proposal, so d_0 is always accepted and position 1 is always decided inside the cycle
    const int a = int(r() % kV), b = (a + 1 + int(r() % (kV - 1))) % kV;
    d.id = {kBase + a, kBase + b};
    d.p = {0.5f, 0.5f};
    return d;
  }
  int ns = regime == kPoint ? 1 + int(r() % 2) : 1 + int(r() % 10);
  std::vector<int> u(kV);
  for (int i = 0; i < kV; ++i) u[i] = i;
  std::shuffle(u.begin(), u.end(), r);
  std::vector<float> w(ns);
  float tot = 0;
  for (int i = 0; i < ns; ++i) {
    w[i] = std::pow(float((r() % 1000) + 1) / 1000.0f, 2.0f);
    tot += w[i];
  }
  std::vector<int> ord(ns);
  for (int i = 0; i < ns; ++i) ord[i] = i;
  std::sort(ord.begin(), ord.end(), [&](int a, int b) { return w[a] > w[b]; });
  for (int i : ord) {
    d.id.push_back(kBase + u[i]);
    d.p.push_back(w[i] / tot);
  }
  return d;
}

// q(.|prefix): 16 distinct candidates with probabilities (the walk's softmax over its top-16)
Dist q_of(const std::vector<int>& prefix, int regime, const Dist& p) {
  std::mt19937_64 r(key_of(prefix) ^ 0x2222);
  Dist q;
  if (regime == kBias) {
    // q = p for the first proposal (prefix of length <= 1 here: the anchor's cycle starts at prefix ()),
    // else 0.9 / 0.1 on p's two tokens: a confidence rule that uses c_p verifies only the 0.9 draw
    q.id = p.id;
    if (prefix.size() % 2 == 0) q.p = {0.5f, 0.5f};
    else q.p = {0.9f, 0.1f};
    for (int i = 0; i < kV && q.id.size() < 16; ++i)
      if (std::find(q.id.begin(), q.id.end(), kBase + i) == q.id.end()) {
        q.id.push_back(kBase + i);
        q.p.push_back(0.0f);
      }
    return q;
  }
  std::vector<int> u(kV);
  for (int i = 0; i < kV; ++i) u[i] = i;
  std::shuffle(u.begin(), u.end(), r);
  if (regime == kEqual || (regime == kMixed && r() % 3 == 0)) {
    // q = p on p's support, the other candidates at 0
    for (size_t i = 0; i < p.id.size(); ++i) {
      q.id.push_back(p.id[i]);
      q.p.push_back(p.p[i]);
    }
    for (int i = 0; i < kV && q.id.size() < 16; ++i)
      if (std::find(q.id.begin(), q.id.end(), kBase + u[i]) == q.id.end()) {
        q.id.push_back(kBase + u[i]);
        q.p.push_back(0.0f);
      }
    return q;
  }
  if (regime == kDisjoint) {
    for (int i = 0; i < kV && q.id.size() < 16; ++i)
      if (std::find(p.id.begin(), p.id.end(), kBase + u[i]) == p.id.end()) q.id.push_back(kBase + u[i]);
    while (q.id.size() < 16) q.id.push_back(kBase + kV + int(q.id.size()));  // outside the universe
  } else {
    for (int i = 0; i < 16; ++i) q.id.push_back(kBase + u[i]);
  }
  std::vector<float> w(16);
  float tot = 0;
  const bool point = regime == kPoint && r() % 2 == 0;
  const bool flat = regime == kMixed && r() % 2 == 0;
  for (int i = 0; i < 16; ++i) {
    w[i] = point ? (i == 0 ? 1.0f : 0.0f) : flat ? 1.0f : std::pow(float((r() % 1000) + 1) / 1000.0f, 3.0f);
    tot += w[i];
  }
  for (int i = 0; i < 16; ++i) q.p.push_back(w[i] / tot);
  return q;
}

int draw(const Dist& q, std::mt19937_64& r, int* k_out) {
  std::uniform_real_distribution<double> U(0, 1);
  double u = U(r), c = 0;
  int k = -1;
  for (size_t i = 0; i < q.p.size(); ++i) {
    if (q.p[i] <= 0) continue;
    c += q.p[i];
    k = int(i);
    if (u < c) break;
  }
  *k_out = k;
  return q.id[k];
}

// per-trial GPU inputs (one dispatch each)
struct Slot {
  std::vector<int> tokens;  // [16]
  std::vector<int> cand;    // [7 * 16]
  std::vector<float> qp;    // [7 * 16]
  std::vector<int> wk;      // [8]
  std::vector<int> did;     // [8 * 32]
  std::vector<float> dp;    // [8 * 32]
};

// Inputs of one cycle with the anchor at position P (tokens[P] = anchor, proposals at P+1..): the
// prefix (the tokens emitted so far) conditions p and q.
void fill(Slot& s, const std::vector<int>& prefix, int P, int K, int regime, std::mt19937_64& r, std::vector<int>* chain) {
  s.tokens.assign(16, 0);
  if (P > 0) s.tokens[P] = prefix.back();
  s.cand.assign(7 * 16, -3);
  s.qp.assign(7 * 16, 0.0f);
  s.wk.assign(8, 0);
  s.did.assign(8 * 32, -1);
  s.dp.assign(8 * 32, 0.0f);
  std::vector<int> pre = prefix;
  chain->clear();
  for (int i = 0; i <= K; ++i) {
    const Dist p = p_of(pre, regime);
    for (size_t j = 0; j < p.id.size(); ++j) {
      s.did[i * 32 + j] = p.id[j];
      s.dp[i * 32 + j] = p.p[j];
    }
    if (i == K) break;
    const Dist q = q_of(pre, regime, p);
    int k;
    const int d = draw(q, r, &k);
    for (int j = 0; j < 16; ++j) {
      s.cand[i * 16 + j] = q.id[j];
      s.qp[i * 16 + j] = q.p[j];
    }
    s.wk[i] = k;
    s.tokens[P + 1 + i] = d;
    chain->push_back(d);
    pre.push_back(d);
  }
}

// ---- tree drafting (section 3) ----
// The target's rows: stage-1 candidates over the whole universe (kV ids), values log p(.|prefix) on p's support and
// lower ones elsewhere, sampled by sample_stage2_rows_tree with top-k / top-p / temperature like the engine.
constexpr int kTreeN = kV;
void target_row(const std::vector<int>& prefix, int regime, float* v, uint32_t* id) {
  const Dist p = p_of(prefix, regime == kPoint ? kPoint : kRandom);
  std::mt19937_64 r(key_of(prefix) ^ 0x3333);
  float lo = 0;
  for (size_t i = 0; i < p.id.size(); ++i) lo = std::min(lo, std::log(p.p[i]));
  int n = 0;
  for (size_t i = 0; i < p.id.size(); ++i) {
    v[n] = std::log(p.p[i]);
    id[n++] = uint32_t(p.id[i]);
  }
  for (int t = 0; t < kV && n < kTreeN; ++t) {
    if (std::find(p.id.begin(), p.id.end(), kBase + t) != p.id.end()) continue;
    v[n] = lo - 4.0f - float(r() % 8) * 0.5f;  // (ties on purpose: the sampler orders them by id)
    id[n++] = uint32_t(kBase + t);
  }
}

// The plain sampler's law of a row (sample_stage2_rows, transcribed): top-k by (value desc, id asc), top-p on the
// untempered probabilities, then the temperature on the kept prefix; T = 0: the first of that order.
struct Law {
  std::vector<uint32_t> id;
  std::vector<float> p;  // the kernel's float arithmetic (unnormalized probs; kt = their sum in order)
  float kt = 1;
};
Law row_law(const float* v, const uint32_t* id, int n, int K, float top_p, float temp) {
  std::vector<std::pair<float, uint32_t>> c(n);
  for (int i = 0; i < n; ++i) c[i] = {v[i], id[i]};
  std::sort(c.begin(), c.end(),
            [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
  Law L;
  if (temp <= 0) {
    L.id = {c[0].second};
    L.p = {1.0f};
    return L;
  }
  K = std::min(K, n);
  std::vector<float> probs(K);
  float tot = 0;
  for (int q = 0; q < K; ++q) {
    probs[q] = std::exp(c[q].first - c[0].first);
    tot += probs[q];
  }
  int keep = K;
  float cum = 0;
  for (int q = 0; q < K; ++q) {
    cum += probs[q] / tot;
    if (cum >= top_p) {
      keep = q + 1;
      break;
    }
  }
  L.kt = 0;
  for (int q = 0; q < keep; ++q) {
    L.id.push_back(c[q].second);
    L.p.push_back(std::exp((c[q].first - c[0].first) / temp));
    L.kt += L.p.back();
  }
  return L;
}
uint32_t pcg(uint32_t v) {
  const uint32_t state = v * 747796405u + 2891336453u;
  const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
// the draw sample_stage2_rows makes with RNG step `key` (the position-keyed uniform)
int law_draw(const Law& L, uint32_t seed, uint32_t key) {
  if (L.id.size() == 1) return int(L.id[0]);
  const float u = float(pcg(seed ^ pcg(key)) & 0xffffff) / 16777216.0f * L.kt;
  float acc = 0;
  for (size_t q = 0; q < L.id.size(); ++q) {
    acc += L.p[q];
    if (u < acc) return int(L.id[q]);
  }
  return int(L.id.back());
}

// A synthetic lattice (cand_i [7][16] from the universe, cand_v, edges [7][16][16]) for a cycle after `prefix`:
// random, or aligned with the target's likely path (the walk's chain is then mostly right and the siblings are p's
// next tokens), drawn from r (independent of the target's uniforms, as a drafter's lattice is).
void lattice(const std::vector<int>& prefix, int regime, std::mt19937_64& r, int* ci, float* cv, float* E) {
  std::normal_distribution<float> N(0, 1);
  std::vector<int> path = prefix;
  for (int d = 0; d < 7; ++d) {
    std::vector<int> u(kV);
    for (int i = 0; i < kV; ++i) u[i] = kBase + i;
    std::shuffle(u.begin(), u.end(), r);
    if (regime == kEqual) {  // aligned: p's support of the main path's prefix first, values log p + noise
      const Dist p = p_of(path, kRandom);
      std::vector<int> c(p.id.begin(), p.id.end());
      for (int t : u)
        if (c.size() < 16 && std::find(c.begin(), c.end(), t) == c.end()) c.push_back(t);
      for (int k = 0; k < 16; ++k) {
        ci[d * 16 + k] = c[k];
        cv[d * 16 + k] = k < int(p.id.size()) ? std::log(p.p[k]) + 0.3f * N(r) : -6.0f + N(r);
      }
      path.push_back(p.id[0]);
    } else {
      for (int k = 0; k < 16; ++k) {
        ci[d * 16 + k] = u[k];
        cv[d * 16 + k] = 1.5f * N(r);
      }
    }
    for (int j = 0; j < 16; ++j)
      for (int k = 0; k < 16; ++k) E[d * 256 + j * 16 + k] = regime == kEqual ? 0.0f : 0.7f * N(r);
  }
}

int tree_cases(long trials, uint64_t salt, int only) {
  Gpu& g = Gpu::get();
  struct TCase {
    int regime;     // lattice: kRandom or kEqual (aligned); target: kRandom, or kPoint (1-2 tokens: deep paths)
    float temp, top_p;
    int top_k;
    float tau, slope;  // > 0: a fixed score temperature; 0: a random learned state (the law must not depend on either)
    int R;             // verify rows (the tree's first R slots)
  };
  const std::vector<TCase> cases = {
      {kRandom, 0.6f, 0.95f, 20, 1.0f, 0.0f, 8}, {kEqual, 0.6f, 0.95f, 20, 0.0f, 0.0f, 8},
      {kEqual, 1.0f, 1.0f, 32, 1.0f, 0.5f, 5},   {kPoint, 0.6f, 0.95f, 20, 0.0f, 0.0f, 3},
      {kRandom, 0.0f, 1.0f, 1, 1.0f, 0.0f, 8},   {kEqual, 0.0f, 1.0f, 1, 0.0f, 0.0f, 6},
      {kEqual, 0.6f, 0.95f, 20, 0.0f, 0.0f, 1}};
  Pipeline tree_p = g.pipeline("dfl_tree"), samp_p = g.pipeline("sample_stage2_rows_tree"),
           acc_p = g.pipeline("dfl_accept_tree");
  const int B = 2048;
  Buffer ci = g.alloc(size_t(B) * 112 * 4), cv = g.alloc(size_t(B) * 112 * 4), E = g.alloc(size_t(B) * 1792 * 4);
  Buffer tok = g.alloc(size_t(B) * 16 * 4), tree = g.alloc(size_t(B) * 64 * 4), cal = g.alloc(256 * 4);
  Buffer rv = g.alloc(size_t(B) * 8 * kTreeN * 4);
  Buffer rid = g.alloc(size_t(B) * 8 * kTreeN * 4), samp = g.alloc(size_t(B) * 8 * 4), res = g.alloc(size_t(B) * 16);
  Buffer gst = g.alloc(size_t(B) * 64), path = g.alloc(size_t(B) * 64);
  int fails = 0;
  for (size_t cix = 0; cix < cases.size(); ++cix) {
    if (only >= 0 && int(cix) != only) continue;
    const TCase& cs = cases[cix];
    const int tgt = cs.regime == kPoint ? kPoint : kRandom;
    const int lat = cs.regime == kEqual ? kEqual : kRandom;
    std::map<long, long> obs, obs1;
    long done = 0, path_bad = 0, compared = 0;
    double mean_a = 0;
    std::mt19937_64 hr(4321 + cix * 7 + salt * 1000003);
    {  // the learner's state the tree reads: a random one (positions' temperatures 0.25 .. 4, coverage 0.5 .. 1)
      memset(cal.ptr(), 0, cal.size);
      std::uniform_real_distribution<float> U(0, 1);
      for (int q = 0; q < 7; ++q) {
        cal.as<float>()[144 + q] = std::exp2(4.0f * U(hr) - 2.0f);
        cal.as<float>()[152 + q] = 0.5f + 0.5f * U(hr);
      }
    }
    // one cycle for trials [0, n) after prefix pre[i] (the anchor at position P = pre[i].size()); the emitted tokens
    auto cycle = [&](const std::vector<std::vector<int>>& pre, const std::vector<uint32_t>& seed) {
      const int n = int(pre.size());
      for (int i = 0; i < n; ++i) {
        lattice(pre[i], lat, hr, ci.as<int>() + i * 112, cv.as<float>() + i * 112, E.as<float>() + i * 1792);
        int* tk = tok.as<int>() + i * 16;
        for (int j = 0; j < 16; ++j) tk[j] = 0;  // tokens[0]: the first anchor (outside the universe)
        for (size_t j = 0; j < pre[i].size(); ++j) tk[1 + j] = pre[i][j];  // the anchor at P = pre.size()
      }
      {  // the trees
        auto e = g.begin();
        for (int i = 0; i < n; ++i) {
          const uint32_t P = uint32_t(pre[i].size());
          kp::DflTree tp{P, P + 1, cs.tau, cs.slope, cs.tau > 0 ? 1u : 0u};
          e->use(tree_p);
          e->buf(0, ci, size_t(i) * 448); e->buf(1, cv, size_t(i) * 448); e->buf(2, E, size_t(i) * 1792 * 4);
          e->buf(3, tok, size_t(i) * 64); e->buf(4, tree, size_t(i) * 256);
          e->val(5, tp);
          e->buf(6, gst, size_t(i) * 64); e->buf(7, cal);
          e->dispatch(1, 1, 1, 32);
        }
        e->commit_and_wait();
      }
      for (int i = 0; i < n; ++i) {  // the target's rows: slot t's path from the tree
        const int* tk = tok.as<int>() + i * 16;
        const int* tr = tree.as<int>() + i * 64;
        const int P = int(pre[i].size());
        for (int t = 0; t < 8; ++t) {
          std::vector<int> nodes;
          for (int v = t; v > 0; v = tr[v]) nodes.push_back(tk[P + v]);
          std::vector<int> pp = pre[i];
          pp.insert(pp.end(), nodes.rbegin(), nodes.rend());
          target_row(pp, tgt, rv.as<float>() + (i * 8 + t) * kTreeN, rid.as<uint32_t>() + (i * 8 + t) * kTreeN);
        }
      }
      {
        auto e = g.begin();
        for (int i = 0; i < n; ++i) {
          const uint32_t P = uint32_t(pre[i].size());
          kp::SampleRows sr{uint32_t(kTreeN), uint32_t(cs.top_k), cs.temp, cs.top_p, seed[i], P, 0, 0, 0};
          e->use(samp_p);
          e->buf(0, rv, size_t(i) * 8 * kTreeN * 4); e->buf(1, rid, size_t(i) * 8 * kTreeN * 4);
          e->buf(2, samp, size_t(i) * 32);
          e->val(3, sr);
          e->buf(4, rid); e->buf(5, tree, size_t(i) * 256);
          e->dispatch(8, 1, 1, 1024);
          const kp::DflAcceptTree ap{P, uint32_t(cs.R)};
          e->use(acc_p);
          e->buf(0, tok, size_t(i) * 64); e->buf(1, samp, size_t(i) * 32); e->buf(2, tree, size_t(i) * 256);
          e->buf(3, res, size_t(i) * 16);
          e->val(4, ap);
          e->buf(5, gst, size_t(i) * 64); e->buf(6, path, size_t(i) * 64);
          e->dispatch(1, 1, 1, 32);
        }
        e->commit_and_wait();
      }
      std::vector<std::vector<int>> out(n);
      for (int i = 0; i < n; ++i) {
        const int a = res.as<int>()[i * 4], P = int(pre[i].size());
        for (int j = 1; j <= a + 1; ++j) out[i].push_back(tok.as<int>()[i * 16 + P + j]);
        if (out[i].back() != res.as<int>()[i * 4 + 1]) ++path_bad;  // the token after the path
        mean_a += P == 0 ? a : 0;
      }
      return out;
    };
    while (done < trials) {
      const int nb = int(std::min<long>(B, trials - done));
      std::vector<uint32_t> seeds(nb);
      for (int i = 0; i < nb; ++i)
        seeds[i] = uint32_t(mix(uint64_t(done + i) + (uint64_t(cix + 100) << 40) + (salt << 48)));
      std::vector<std::vector<int>> pre0(nb);
      auto em = cycle(pre0, seeds);
      std::vector<int> second;
      for (int i = 0; i < nb; ++i)
        if (em[i].size() < 2) second.push_back(i);
      if (!second.empty()) {
        std::vector<std::vector<int>> pre1;
        std::vector<uint32_t> s1;
        for (int i : second) {
          pre1.push_back({em[i][0]});
          s1.push_back(seeds[i]);
        }
        auto em2 = cycle(pre1, s1);
        for (size_t j = 0; j < second.size(); ++j) em[second[j]].push_back(em2[j][0]);
      }
      for (int i = 0; i < nb; ++i) {
        const int t1 = em[i][0], t2 = em[i][1];
        obs[long(t1) * 100000 + t2]++;
        obs1[t1]++;
        // pathwise: plain decoding with position-keyed uniforms writes the same two tokens
        float v[kTreeN];
        uint32_t id[kTreeN];
        target_row({}, tgt, v, id);
        const int p1 = law_draw(row_law(v, id, kTreeN, cs.top_k, cs.top_p, cs.temp), seeds[i], 0);
        target_row({p1}, tgt, v, id);
        const int p2 = law_draw(row_law(v, id, kTreeN, cs.top_k, cs.top_p, cs.temp), seeds[i], 1);
        ++compared;
        if (p1 != t1 || p2 != t2) ++path_bad;
      }
      done += nb;
    }
    // expected law of (t1, t2)
    std::map<long, double> ex, ex1;
    float v[kTreeN];
    uint32_t id[kTreeN];
    target_row({}, tgt, v, id);
    const Law l0 = row_law(v, id, kTreeN, cs.top_k, cs.top_p, cs.temp);
    for (size_t i = 0; i < l0.id.size(); ++i) {
      const double p0 = l0.p[i] / l0.kt;
      ex1[long(l0.id[i])] += p0;
      target_row({int(l0.id[i])}, tgt, v, id);
      const Law l1 = row_law(v, id, kTreeN, cs.top_k, cs.top_p, cs.temp);
      for (size_t j = 0; j < l1.id.size(); ++j) ex[long(l0.id[i]) * 100000 + long(l1.id[j])] += p0 * l1.p[j] / l1.kt;
    }
    double st, pv, st1, pv1;
    int dof, dof1;
    long out, out1;
    chi2(obs, ex, trials, &st, &dof, &pv, &out);
    chi2(obs1, ex1, trials, &st1, &dof1, &pv1, &out1);
    const bool greedy = cs.temp <= 0;
    const bool ok = (greedy || (pv > 0.01 && pv1 > 0.01)) && out == 0 && out1 == 0 && path_bad == 0;
    char rule[48];
    if (cs.tau > 0) snprintf(rule, sizeof rule, "tau %.1f+%.1f/pos", cs.tau, cs.slope);
    else snprintf(rule, sizeof rule, "learned state");
    printf("[S1 tree] lattice %-8s target %-6s T=%.1f top-p %.2f top-k %d %s, %d rows: %ld trials, mean a %.2f; "
           "joint chi2 %.1f (%d dof) p=%.3f; first token p=%.3f; outside support %ld; plain-decoding mismatches %ld of "
           "%ld: %s\n",
           lat == kEqual ? "aligned" : "random", tgt == kPoint ? "1-2" : "random", cs.temp, cs.top_p, cs.top_k, rule,
           cs.R, trials, mean_a / trials, st, dof, pv, pv1, out + out1, path_bad, compared, ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
  }
  return fails;
}

// The tree's learner (dfl_tree_learn) against a CPU transcription of it (the same float arithmetic in the same order)
// over many random cycles, tree and chain: its state (per position the grid's log-likelihoods, hits, samples, the
// temperature and coverage it picks; the running expectations) must match.
void learn_cpu(float* cal, const int* ci, const float* cv, const float* E, const int* tree, const int* path,
               const int* samp, const int* walk_k, const int* res, bool chain) {
  const uint32_t n = chain ? uint32_t(res[0]) + 1 : uint32_t(path[8]);
  const float lam = 1.0f - 1.0f / 256.0f;
  for (uint32_t j = 0; j < std::min(n, 7u); ++j) {
    const uint32_t slot = chain ? j : uint32_t(path[j]);
    const uint32_t kpar = j == 0 ? 0u : chain ? uint32_t(walk_k[j - 1]) : uint32_t(tree[48 + slot]);
    const int y = samp[slot];
    float s[16], mx = -INFINITY;
    int ks = 32;
    for (int k = 0; k < 16; ++k) {
      const float e = E[j * 256 + (j == 0 ? 0u : kpar) * 16 + k];
      s[k] = cv[j * 16 + k] + (std::isnan(e) ? 0.0f : e);
      if (ks == 32 && ci[j * 16 + k] == y) ks = k;
      mx = std::max(mx, s[k]);
    }
    float best = -INFINITY;
    int gb = 0;
    for (int g = 0; g < 17; ++g) {
      const float lg = (float(g) - 8.0f) / 4.0f, tau = std::exp2(lg);
      float z = 0.0f, zk = 0.0f;
      for (int k = 0; k < 16; ++k) {
        const float sk = (s[k] - mx) / tau;
        z += std::exp(sk);
        if (k == ks) zk = sk;
      }
      const float ll = ks < 32 ? zk - std::log(z) : 0.0f;
      const float cur = cal[j * 17 + g] * lam + ll;
      cal[j * 17 + g] = cur;
      if (cur - 0.5f * lg * lg > best) {
        best = cur - 0.5f * lg * lg;
        gb = g;
      }
    }
    const float h = cal[128 + j] * lam + (ks < 32 ? 1.0f : 0.0f), m = cal[136 + j] * lam + 1.0f;
    cal[128 + j] = h;
    cal[136 + j] = m;
    cal[144 + j] = std::exp2((float(gb) - 8.0f) / 4.0f);
    cal[152 + j] = (h + 1.0f) / (m + 1.0f);
  }
  const float cnt = std::min(cal[170] + 1.0f, 64.0f);
  cal[170] = cnt;
  auto f = [&](int w) { float x; memcpy(&x, &tree[w], 4); return x; };
  for (int r = 0; r < 8; ++r) cal[160 + r] += (f(40 + r) - cal[160 + r]) / cnt;
  cal[168] += (f(26) - cal[168]) / cnt;
  cal[169] += (float(n) - cal[169]) / cnt;
}

int learn_check(uint64_t salt) {
  Gpu& g = Gpu::get();
  const int N = 400;
  std::mt19937_64 r(99 + salt);
  std::normal_distribution<float> Nd(0, 1);
  Buffer ci = g.alloc(size_t(N) * 112 * 4), cv = g.alloc(size_t(N) * 112 * 4), E = g.alloc(size_t(N) * 1792 * 4);
  Buffer tree = g.alloc(size_t(N) * 64 * 4), path = g.alloc(size_t(N) * 64), samp = g.alloc(size_t(N) * 32);
  Buffer wk = g.alloc(size_t(N) * 32), res = g.alloc(size_t(N) * 16), cal = g.alloc(256 * 4);
  std::vector<float> ref(256, 0.0f);
  for (int q = 0; q < 7; ++q) ref[144 + q] = ref[152 + q] = 1.0f;
  memcpy(cal.ptr(), ref.data(), 256 * 4);
  std::vector<uint32_t> modes(N);
  for (int c = 0; c < N; ++c) {
    int* cic = ci.as<int>() + c * 112;
    float* cvc = cv.as<float>() + c * 112;
    float* Ec = E.as<float>() + c * 1792;
    int* tr = tree.as<int>() + c * 64;
    int* pa = path.as<int>() + c * 16;
    int* sa = samp.as<int>() + c * 8;
    int* w = wk.as<int>() + c * 8;
    int* rs = res.as<int>() + c * 4;
    for (int i = 0; i < 112; ++i) {
      cic[i] = 1000 + int(r() % 48);
      cvc[i] = 2.0f * Nd(r);
    }
    for (int i = 0; i < 1792; ++i) Ec[i] = c % 3 == 0 ? 0.0f : Nd(r);  // (some v1-like lattices: no edges)
    for (int i = 16; i < 256; ++i) Ec[i] = std::nanf("");               // position 0 has edge row 0 only
    for (int i = 0; i < 64; ++i) tr[i] = 0;
    for (int i = 0; i < 8; ++i) {
      const float er = 1.0f + 0.7f * float(i) * (0.5f + 0.5f * float(r() % 100) / 100.0f);
      memcpy(&tr[40 + i], &er, 4);
      tr[48 + i] = i == 0 ? -1 : int(r() % 16);
      w[i] = int(r() % 16);
    }
    const float ech = 1.0f + float(r() % 400) / 100.0f;
    memcpy(&tr[26], &ech, 4);
    modes[c] = uint32_t(r() % 2);
    const int a = int(r() % 8);
    rs[0] = a;
    std::vector<int> slots = {1, 2, 3, 4, 5, 6, 7};
    std::shuffle(slots.begin(), slots.end(), r);
    pa[0] = 0;
    for (int j = 1; j <= a; ++j) pa[j] = slots[size_t(j - 1)];
    pa[8] = a + 1;
    for (int j = 0; j <= std::min(a, 6); ++j) {  // the row's token: a candidate of position j (75%) or none
      const int slot = modes[c] ? j : pa[j];
      sa[slot] = r() % 4 ? cic[j * 16 + int(r() % 16)] : 7;
    }
    learn_cpu(ref.data(), cic, cvc, Ec, tr, pa, sa, w, rs, modes[c] != 0);
  }
  Pipeline lp = g.pipeline("dfl_tree_learn");
  auto e = g.begin();
  for (int c = 0; c < N; ++c) {
    e->use(lp);
    e->buf(0, ci, size_t(c) * 448); e->buf(1, cv, size_t(c) * 448); e->buf(2, E, size_t(c) * 7168);
    e->buf(3, tree, size_t(c) * 256); e->buf(4, path, size_t(c) * 64); e->buf(5, samp, size_t(c) * 32);
    e->buf(6, wk, size_t(c) * 32); e->buf(7, res, size_t(c) * 16); e->buf(8, cal);
    e->val(9, modes[c]);
    e->dispatch(1, 1, 1, 32);
  }
  e->commit_and_wait();
  double ll_err = 0, other_err = 0;
  int tau_bad = 0;
  for (int i = 0; i < 171; ++i) {
    const double d = std::fabs(double(cal.as<float>()[i]) - ref[size_t(i)]) / std::max(1.0, std::fabs(double(ref[size_t(i)])));
    if (i < 128) ll_err = std::max(ll_err, d);
    else if (i >= 144 && i < 151) tau_bad += cal.as<float>()[i] != ref[size_t(i)];
    else other_err = std::max(other_err, d);
  }
  const bool ok = ll_err < 1e-4 && other_err < 1e-5 && tau_bad == 0;
  printf("[learn] dfl_tree_learn vs CPU over %d tree / chain cycles: log-likelihood max rel. error %.2e, temperatures "
         "differing %d of 7, other state max rel. error %.2e; learned tau/cover:", N, ll_err, tau_bad, other_err);
  for (int q = 0; q < 7; ++q) printf(" %.2f/%.2f", cal.as<float>()[144 + q], cal.as<float>()[152 + q]);
  printf(": %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const long trials = argc > 1 ? atol(argv[1]) : 40000;
  const int Kmain = argc > 2 ? atoi(argv[2]) : 3;
  const uint64_t salt = argc > 3 ? uint64_t(atoll(argv[3])) : 0;  // seed stream offset (repeat with others)
  const int only = argc > 4 ? atoi(argv[4]) : -1;                   // run only case #only
  Gpu& g = Gpu::get();
  int fails = 0;

  // ---- 1. dfl_dist_rows vs the CPU transcription of sample_stage2_rows ----
  {
    const uint32_t R = 8, n = 122 * 20, Kk = 20;
    Buffer cv = g.alloc(size_t(R) * n * 4), ci = g.alloc(size_t(R) * n * 4);
    Buffer did = g.alloc(R * 32 * 4), dp = g.alloc(R * 32 * 4), dk = g.alloc(R * 4);
    std::mt19937_64 r(7);
    for (uint32_t row = 0; row < R; ++row)
      for (uint32_t i = 0; i < n; ++i) {
        // a few exact ties and a peaked head
        float v = float(int(r() % 4000)) / 400.0f;
        if (i % 97 == 0) v = 12.0f;
        cv.as<float>()[row * n + i] = v;
        ci.as<uint32_t>()[row * n + i] = uint32_t((i * 7919 + row * 13) % 248320);
      }
    const float temps[] = {0.6f, 1.0f, 0.3f};
    const float tops[] = {0.95f, 0.8f, 1.0f};
    double max_rel = 0;
    int id_bad = 0;
    for (int t = 0; t < 3; ++t) {
      kp::DflDist p{n, Kk, temps[t], tops[t]};
      auto e = g.begin();
      e->use(g.pipeline("dfl_dist_rows"));
      e->buf(0, cv); e->buf(1, ci); e->buf(2, did); e->buf(3, dp); e->buf(4, dk);
      e->val(5, p);
      e->dispatch(R, 1, 1, 1024);
      e->commit_and_wait();
      for (uint32_t row = 0; row < R; ++row) {
        std::vector<std::pair<float, uint32_t>> c(n);
        for (uint32_t i = 0; i < n; ++i) c[i] = {cv.as<float>()[row * n + i], ci.as<uint32_t>()[row * n + i]};
        std::sort(c.begin(), c.end(),
            [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        float probs[32], tot = 0;
        for (uint32_t q = 0; q < Kk; ++q) {
          probs[q] = std::exp(c[q].first - c[0].first);
          tot += probs[q];
        }
        uint32_t keep = Kk;
        float cum = 0;
        for (uint32_t q = 0; q < Kk; ++q) {
          cum += probs[q] / tot;
          if (cum >= tops[t]) {
            keep = q + 1;
            break;
          }
        }
        float kt = 0;
        for (uint32_t q = 0; q < keep; ++q) {
          probs[q] = std::exp((c[q].first - c[0].first) / temps[t]);
          kt += probs[q];
        }
        if (dk.as<uint32_t>()[row] != keep) ++id_bad;
        for (uint32_t q = 0; q < keep; ++q) {
          if (uint32_t(did.as<int>()[row * 32 + q]) != c[q].second) ++id_bad;
          const double ref = probs[q] / kt, got = dp.as<float>()[row * 32 + q];
          max_rel = std::max(max_rel, std::fabs(got - ref) / ref);
        }
      }
    }
    const bool ok = id_bad == 0 && max_rel < 1e-5;
    printf("[dist] dfl_dist_rows vs CPU sampler distribution, 3 settings x 8 rows: %d id/keep mismatches, max rel. "
           "prob error %.2e: %s\n", id_bad, max_rel, ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
  }

  // ---- 2. the speculative loop: joint law of the first two emitted tokens ----
  struct Case {
    int regime, K;
    float temp;
    float tau = 0.0f;  // > 0: dynamic verify depth (dfl_depth) with this threshold
    int rule = 0;      // dfl_depth rule (1 = own confidence at T > 0: biased, expected to fail; 2 = the earlier rule)
  };
  std::vector<Case> cases = {{kRandom, Kmain, 0.6f}, {kEqual, Kmain, 0.6f}, {kDisjoint, Kmain, 0.6f},
                             {kPoint, Kmain, 0.6f},  {kMixed, Kmain, 0.6f}, {kRandom, 1, 0.6f},
                             {kRandom, 7, 1.0f},     {kMixed, 7, 1.0f},
                             {kRandom, 3, 0.6f, 0.1f}, {kRandom, 7, 1.0f, 0.05f}, {kMixed, 7, 0.6f, 0.2f},
                             {kEqual, 7, 0.6f, 0.3f},  {kBias, 3, 0.6f, 0.3f},       {kBias, 3, 0.6f, 0.3f, 1},
                             {kBias, 3, 0.6f, 0.3f, 2}, {kMixed, 7, 0.6f, 0.2f, 2}};
  Pipeline acc = g.pipeline("dfl_accept"), dep = g.pipeline("dfl_depth");
  const int B = 4096;  // trials per command buffer
  Buffer tok = g.alloc(size_t(B) * 16 * 4), cand = g.alloc(size_t(B) * 112 * 4), qp = g.alloc(size_t(B) * 112 * 4);
  Buffer wk = g.alloc(size_t(B) * 8 * 4), did = g.alloc(size_t(B) * 256 * 4), dp = g.alloc(size_t(B) * 256 * 4);
  Buffer res = g.alloc(size_t(B) * 4 * 4), samp = g.alloc(64), scratch = g.alloc(64), kd = g.alloc(size_t(B) * 16);
  Buffer ws = g.alloc(size_t(B) * 112 * 4);  // the walk's scores: log q, so softmax(S_j) = q_j (dfl_depth)
  for (size_t ci = 0; ci < cases.size(); ++ci) {
    if (only >= 0 && int(ci) != only) continue;
    const Case& cs = cases[ci];
    std::map<long, long> obs, obs1;
    long done = 0;
    double mean_a = 0;
    std::mt19937_64 hr(1234 + cs.regime * 17 + cs.K + salt * 1000003);
    while (done < trials) {
      const int nb = int(std::min<long>(B, trials - done));
      // first cycle of each trial: anchor at position 0, prefix = (); a trial that emits a single token
      // continues with a second cycle anchored at position 1 on that token (its uniforms then come from
      // positions 2.., as in the engine)
      std::vector<std::vector<int>> chains(nb);
      auto upload = [&](int i, const Slot& s) {
        memcpy(tok.as<int>() + i * 16, s.tokens.data(), 64);
        memcpy(cand.as<int>() + i * 112, s.cand.data(), 112 * 4);
        memcpy(qp.as<float>() + i * 112, s.qp.data(), 112 * 4);
        for (int j = 0; j < 112; ++j) ws.as<float>()[i * 112 + j] = s.qp[j] > 0 ? std::log(s.qp[j]) : -1e30f;
        memcpy(wk.as<int>() + i * 8, s.wk.data(), 32);
        memcpy(did.as<int>() + i * 256, s.did.data(), 256 * 4);
        memcpy(dp.as<float>() + i * 256, s.dp.data(), 256 * 4);
      };
      auto run = [&](int count, uint32_t P, const std::vector<uint32_t>& seed) {
        auto e = g.begin();
        for (int i = 0; i < count; ++i) {
          if (cs.tau > 0) {  // the drafter's depth decision from this trial's q and walk choices
            kp::DflDepth dpp{uint32_t(cs.K), cs.tau, cs.temp, uint32_t(cs.rule)};
            e->use(dep);
            e->buf(0, ws, size_t(i) * 448); e->buf(1, wk, size_t(i) * 32); e->buf(2, qp, size_t(i) * 448);
            e->buf(3, kd, size_t(i) * 16);
            e->val(4, dpp);
            e->dispatch(1, 1, 1, 32);
          }
          kp::DflAccept ap{P, uint32_t(cs.K), cs.temp, seed[i], cs.tau > 0 ? 1u : 0u};
          e->use(acc);
          e->buf(0, tok, size_t(i) * 64);
          e->buf(1, samp); e->buf(2, did, size_t(i) * 1024); e->buf(3, dp, size_t(i) * 1024);
          e->buf(4, cand, size_t(i) * 448); e->buf(5, qp, size_t(i) * 448); e->buf(6, wk, size_t(i) * 32);
          e->buf(7, res, size_t(i) * 16);
          e->val(8, ap);
          e->buf(9, kd, size_t(i) * 16);  // dynamic depth (read only with ap.dyn)
          e->buf(10, scratch);  // drafter state for a GPU-driven next pass (unused here)
          e->dispatch(1, 1, 1, 32);
        }
        e->commit_and_wait();
      };
      std::vector<uint32_t> seeds(nb);
      for (int i = 0; i < nb; ++i) {
        Slot s;
        fill(s, {}, 0, cs.K, cs.regime, hr, &chains[i]);
        upload(i, s);
        seeds[i] = uint32_t(mix(uint64_t(done + i) + (uint64_t(ci) << 40) + (salt << 48)));
      }
      run(nb, 0, seeds);
      std::vector<int> t1(nb), t2(nb, -1), second;
      for (int i = 0; i < nb; ++i) {
        const int a = res.as<int>()[i * 4];
        const int y = res.as<int>()[i * 4 + 1];
        mean_a += a;
        std::vector<int> em(chains[i].begin(), chains[i].begin() + a);
        em.push_back(y);
        t1[i] = em[0];
        if (em.size() >= 2) t2[i] = em[1];
        else second.push_back(i);
      }
      if (!second.empty()) {
        std::vector<uint32_t> s1;
        std::vector<std::vector<int>> ch2(second.size());
        for (size_t j = 0; j < second.size(); ++j) {
          Slot s;
          fill(s, {t1[second[j]]}, 1, cs.K, cs.regime, hr, &ch2[j]);
          upload(int(j), s);
          s1.push_back(seeds[second[j]]);
        }
        run(int(second.size()), 1, s1);
        for (size_t j = 0; j < second.size(); ++j) {
          const int a = res.as<int>()[j * 4];
          const int y = res.as<int>()[j * 4 + 1];
          t2[second[j]] = a >= 1 ? ch2[j][0] : y;
        }
      }
      for (int i = 0; i < nb; ++i) {
        obs[long(t1[i]) * 100000 + t2[i]]++;
        obs1[t1[i]]++;
      }
      done += nb;
    }
    // expected law
    std::map<long, double> ex, ex1;
    const Dist p0 = p_of({}, cs.regime);
    for (size_t i = 0; i < p0.id.size(); ++i) {
      ex1[p0.id[i]] = p0.p[i];
      const Dist p1 = p_of({p0.id[i]}, cs.regime);
      for (size_t j = 0; j < p1.id.size(); ++j) ex[long(p0.id[i]) * 100000 + p1.id[j]] += double(p0.p[i]) * p1.p[j];
    }
    double st, pv, st1, pv1;
    int dof, dof1;
    long out, out1;
    chi2(obs, ex, trials, &st, &dof, &pv, &out);
    chi2(obs1, ex1, trials, &st1, &dof1, &pv1, &out1);
    const bool ok = pv > 0.01 && pv1 > 0.01 && out == 0 && out1 == 0;
    char dyn[64] = "";
    if (cs.tau > 0) snprintf(dyn, sizeof dyn, ", depth tau %.2f rule %d", cs.tau, cs.rule);
    const bool expect_fail = cs.rule == 1;
    printf("[S1] %-24s K=%d T=%.1f%s: %ld trials, mean a %.2f; joint chi2 %.1f (%d dof) p=%.3f; first token chi2 "
           "%.1f (%d dof) p=%.3f; outside support %ld: %s%s\n",
           regime_name(cs.regime), cs.K, cs.temp, dyn, trials, mean_a / trials, st, dof, pv, st1, dof1, pv1,
           out + out1, ok ? "PASS" : "FAIL", expect_fail ? (ok ? " (biased rule NOT detected)" : " (biased rule detected, as expected)") : "");
    if (ok == expect_fail) ++fails;
  }
  // ---- 3. tree drafting (dense targets): dfl_tree + sample_stage2_rows_tree + dfl_accept_tree ----
  if (only < 0 || only >= int(cases.size()))
    fails += tree_cases(trials, salt, only < 0 ? -1 : only - int(cases.size()));
  if (only < 0) fails += learn_check(salt);
  printf(fails ? "FAILED: %d\n" : "ALL PASSED\n", fails);
  return fails ? 1 : 0;
}

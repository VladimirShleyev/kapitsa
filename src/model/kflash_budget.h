// KFlash (Kapitsa's drafter generation after DFlash2: best-first trees over a drafter's candidates, verified exactly)
// on the host: the tree descriptor's layout and the builder's parameters (src/kernels/kflash.h, shared with the GPU),
// and the budget that sets them between cycles, the same for every backend and with no per-model constants:
//  - calibration: the path model's probability q of a node is mapped to the probability that the target accepts it,
//    p = sigmoid(alpha_d + beta_d logit(q)) per depth d, fitted online (logistic regression, Newton steps on the last
//    records) from the acceptance's records: every verified node whose parent was accepted, and whether it was.
//    Before any records, alpha 0 and beta 1: p = q.
//  - cost: a cycle's time with n verify rows, measured on the Mac it runs on: the first cycles try every allowed n
//    twice (unless the cache already holds a table for this model, drafter and chip), then a running mean per n.
// The builder takes the allowed n that maximizes expected tokens per ms: (1 + the summed calibrated path probabilities
// of the first n - 1 nodes) / cost[n]. Which n a backend allows is what its verify kernels run at a cost of their own:
// whole 8-row tiles on the dense Qwen models (one tile today), every row count on MoE targets, whose rows cost experts.
// The cache file (KAPITSA_HOME/kflash/<key>.json) keeps the table and the calibration.
#pragma once
#include "kernels/kflash.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ni {

class KfBudget {
 public:
  // key: the cache file's name (model, drafter, chip); "" keeps nothing
  explicit KfBudget(std::string key = "");
  ~KfBudget();
  // the row counts the backend's verify runs (bit n - 1) and its capacity; default: 1 .. 8 rows, 8
  void set_rows(uint32_t allowed, int max_rows);
  // after a cycle: the tree descriptor as its acceptance left it (rows, records) and the cycle's time (ms)
  void observe(const int32_t* tree, double ms);
  // before the next cycle's drafting pass
  void write(KfParams& p) const;
  // for logs: rows chosen on average, the calibration, the cost table
  std::string summary() const;
  int cycles() const { return cycles_; }
  double cost(int n) const { return n >= 1 && n <= kf::kMaxRows ? cost_[n] : 0.0; }
  float alpha(int d) const { return alpha_[d]; }
  float beta(int d) const { return beta_[d]; }

  // (tests) the calibration fit on given records
  static void fit(const std::vector<float>& x, const std::vector<uint8_t>& y, float& alpha, float& beta, int iters);
  static float logit_q(float lq);

 private:
  void load();
  void save() const;
  std::string key_;
  static constexpr int kD = kf::kMaxRows + 1;  // depths and row counts: index 1 .. 32
  static constexpr int kWin = 2048;            // records kept per depth
  std::vector<float> x_[kD];                   // logit(q) per depth (ring)
  std::vector<uint8_t> y_[kD];
  size_t head_[kD] = {};
  float alpha_[kD], beta_[kD];
  double cost_[kD] = {};
  int ncost_[kD] = {};
  uint32_t allowed_ = 0xffu;
  int max_rows_ = 8;
  int cycles_ = 0, since_fit_ = 0, force_ = 0;
  double rows_sum_ = 0;
};

}  // namespace ni

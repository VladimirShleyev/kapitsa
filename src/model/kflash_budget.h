// KFlash (Kapitsa's drafter generation after DFlash2: best-first trees over a drafter's candidates, verified exactly)
// on the host: the tree buffer's layout the kernels share (nh_common.h's kKf*), the parameters the GPU tree builder
// reads (nh_tree.metal's NhKfParams), and the budget that sets them between cycles, with no per-model constants:
//  - calibration: the drafter's probability q of a node is mapped to the probability that the target accepts it,
//    p = sigmoid(alpha_d + beta_d logit(q)) per depth d, fitted online (logistic regression, Newton steps on the last
//    records) from the acceptance's records: every verified node whose parent was accepted, and whether it was.
//    Before any records, alpha 0 and beta 1: p = q.
//  - cost: a cycle's wall time with n verify rows (n = 1 .. 8), measured: the first cycles try every n twice (unless
//    the cache already holds a table for this model, drafter and chip), then a running mean per n.
// The builder takes the n that maximizes expected tokens per ms: (1 + the summed path probabilities of the first
// n - 1 nodes) / cost[n]. The cache file (KAPITSA_HOME/kflash/<key>.json) keeps the table and the calibration.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ni {

namespace kf {
// the tree buffer (int32 words): see nh_common.h
constexpr int kN = 0, kPar = 1, kDep = 9, kMask = 17, kLq = 25, kA = 33, kPath = 34, kNrec = 43, kRecDep = 44,
              kRecLq = 52, kRecHit = 60, kWords = 68;
constexpr int kMaxRows = 8;
}  // namespace kf

// nh_tree.metal's NhKfParams
struct KfParams {
  float alpha[8], beta[8];  // depth d = 1 .. 7
  float cost[9];            // n = 1 .. 8 rows
  int32_t force_n;
};

class KfBudget {
 public:
  // key: the cache file's name (model, drafter, chip); "" keeps nothing
  explicit KfBudget(std::string key = "");
  ~KfBudget();
  // after a cycle: the tree buffer as its acceptance left it (rows, records) and the cycle's wall time
  void observe(const int32_t* tree, double ms);
  // before the next cycle's drafting pass
  void write(KfParams& p);
  // for logs: rows chosen on average, the calibration, the cost table
  std::string summary() const;
  int cycles() const { return cycles_; }

  // (tests) the calibration fit on given records
  static void fit(const std::vector<float>& x, const std::vector<uint8_t>& y, float& alpha, float& beta, int iters);
  static float logit_q(float lq);

 private:
  void load();
  void save() const;
  std::string key_;
  static constexpr int kWin = 2048;  // records kept per depth
  std::vector<float> x_[8];          // logit(q) per depth (ring)
  std::vector<uint8_t> y_[8];
  size_t head_[8] = {};
  float alpha_[8], beta_[8];
  double cost_[9] = {};
  int ncost_[9] = {};
  int cycles_ = 0, since_fit_ = 0, explore_ = 0, force_ = 0;
  double rows_sum_ = 0;
};

}  // namespace ni

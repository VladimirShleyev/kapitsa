// KFlash (Kapitsa's drafter generation after DFlash2: best-first trees over a drafter's candidates, verified exactly)
// on the host: the tree descriptor's layout and the builder's parameters (src/kernels/kflash.h, shared with the GPU),
// and the budget that sets them between cycles, the same for every backend and with no per-model constants:
//  - calibration: the path model's probability q of a node is mapped to the probability that the target accepts it,
//    p = sigmoid(alpha_d + beta_d logit(q)) per depth d, fitted online (logistic regression, Newton steps on the last
//    records) from the acceptance's records: every verified node whose parent was accepted, and whether it was.
//    Before any records, alpha 0 and beta 1: p = q.
//  - cost: a cycle's GPU time with n verify rows, measured on the Mac it runs on: its drafting half's command buffer
//    and its verify's, in ms, whichever tool runs the cycles (ni_run, the server), so the cache means one thing
//    whoever wrote it. The first cycles try every allowed n twice (unless the cache already holds a table for this
//    model, drafter and chip), then a running mean per n; every kRefresh cycles one cycle re-measures the allowed n
//    measured longest ago (costs drift with the context and the Mac's load, and a count the rule never picks would
//    otherwise keep its first two measurements). Wide verifies (below) model the drift: log cost(n) = log base + log
//    ratio(n), the base an 8-row verify's cost now (every cycle moves it, so a count measured at another context
//    length, or in an earlier process, is priced at this one) and the ratios each count's own, fitted online.
// The builder takes the allowed n that maximizes expected tokens per ms: (1 + the summed calibrated path probabilities
// of the first n - 1 nodes) / cost[n]. Which n a backend allows is what its verify kernels run at a cost of their own:
// whole 8-row tiles on the dense Qwen models (8 | 16 | 32 rows), every row count on MoE targets, whose rows cost experts.
// Wide verifies (set_wide: Qwen's 16 / 32 rows, one verify of capacity max_rows whose live tiles the GPU picks): the
// capacity is a choice too. A wide verify of 8 live rows costs a little more than the 8-row verify (its grids are
// GPU-driven), so each cycle verifies at capacity 8 (the rule then has 8 rows only) unless the recent cycles' trees
// (the builder still builds max_rows nodes and records E(8), E(16), E(32)) say the per-cycle choice among all counts
// gives more tokens per ms than 8 rows at the 8-row verify's cost; both costs are measured, like every count's.
// The cache file (KAPITSA_HOME/kflash/<key>.json) keeps the table and the calibration: written when the budget ends
// and on save() (the engine saves between requests, the server when it is stopped).
#pragma once
#include "kernels/kflash.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace ni {

class KfBudget {
 public:
  // key: the cache file's name (model, drafter, chip); "" keeps nothing
  explicit KfBudget(std::string key = "");
  ~KfBudget();  // save()
  // the row counts the backend's verify runs (bit n - 1) and its capacity; default: 1 .. 8 rows, 8
  void set_rows(uint32_t allowed, int max_rows);
  // the counts past 8 run as one wide verify of capacity max_rows (see above): the capacity is chosen per cycle
  void set_wide(bool on);
  // after a cycle: the tree descriptor as its acceptance left it (rows, records) and the cycle's GPU time (ms): its
  // drafting half's command buffer and its verify's, Encoder::gpu_ms() of each
  void observe(const int32_t* tree, double ms);
  // before the next cycle's drafting pass (p.cap: the capacity the verify must be encoded with)
  void write(KfParams& p);
  // the table and the calibration to the cache file, once 32 cycles were observed here and `fresh` of them since the
  // last save. From any thread (the server saves on its stop signal while a request may be in its cycles).
  void save(int fresh = 1);
  static constexpr int kSaveEvery = 256;  // the engine's save after a request: a server killed hard loses less
  // for logs: rows chosen on average, the calibration, the cost table
  std::string summary() const;
  int cycles() const { return cycles_; }
  double cost(int n) const { return n >= 1 && n <= kf::kMaxRows ? cost_[n] : 0.0; }
  double cost_narrow() const { return cost8n_; }  // an 8-row cycle at capacity 8 (wide verifies)
  float alpha(int d) const { return alpha_[d]; }
  float beta(int d) const { return beta_[d]; }

  // (tests) the calibration fit on given records
  static void fit(const std::vector<float>& x, const std::vector<uint8_t>& y, float& alpha, float& beta, int iters);
  static float logit_q(float lq);

 private:
  void load();
  std::string key_;
  mutable std::mutex mu_;  // observe, write, summary and save (the cycles' thread and a server's stop)
  static constexpr int kD = kf::kMaxRows + 1;  // depths and row counts: index 1 .. 32
  static constexpr int kWin = 2048;            // records kept per depth
  static constexpr int kRefresh = 2048;        // cycles between re-measurements of the stalest allowed row count
  std::vector<float> x_[kD];                   // logit(q) per depth (ring)
  std::vector<uint8_t> y_[kD];
  size_t head_[kD] = {};
  float alpha_[kD], beta_[kD];
  double cost_[kD] = {};
  int ncost_[kD] = {};
  uint32_t allowed_ = 0xffu;
  int max_rows_ = 8;
  int cycles_ = 0, since_fit_ = 0, force_ = 0, saved_ = 0;  // saved_: cycles_ at the last save
  double rows_sum_ = 0;
  int chosen_[kD] = {};  // cycles by verified rows (the summary's histogram)
  int last_[kD] = {};    // the cycle that last measured each row count
  int refresh_ = 0;      // a row count the next cycles verify to re-measure it (0: none; -1: the narrow 8 rows)
  int force_env_ = 0;    // NI_KFLASH_FORCE=n (tests): every cycle verifies n rows
  // wide verifies: log cost(n) = lb_ + lq_[n] (lb_: the 8-row verify's log cost now, set by this process's first
  // narrow cycle and moved by every cycle; lq_[n]: count n's log ratio to it, ncost_[n] its measurements) and the
  // narrow 8 rows' (count, last measured, measurements this process); cost_[n] / cost8n_ are their current values;
  // the capacity decision, the recent trees' E(8), E(16), E(32) (a ring), cycles run narrow
  bool wide_ = false, wide_pays_ = true, wide_saved_ = false;  // wide_saved_: a wide run wrote the cache
  double cost8n_ = 0, lb_ = 0, lq_[kD] = {};
  int ncost8n_ = 0, last8n_ = 0, nb_ = 0, narrow_cycles_ = 0, since_eval_ = 0, narrow_writes_ = 0;
  static constexpr int kStatsEvery = 8;  // capacity 8: one cycle in this many builds max_rows nodes (the E statistics)
  void wide_costs();  // cost_[n], cost8n_ from lb_, lq_
  static constexpr int kEWin = 256;
  std::vector<float> ehist_;  // 3 per cycle
  size_t ehead_ = 0;
  void eval_wide();
};

}  // namespace ni

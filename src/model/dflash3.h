// DFlash3Turbo candidate decoder (internal/DFLASH3TURBO.md §1, §4): a small autoregressive decoder that scores the
// block drafter's top-32 candidates of each position given the whole drafted path, replacing DFlash2's first-order
// selector. Loads <model>/dflash3 in the trainer's format (tools/dflash3/README.md: bf16 projections, quantized at
// load to 4-bit group 64 with mx.quantize's rounding as the DFlash2 drafter's; the embedding norm, per-row logit
// scales and the two-row acceptance head as stored), keeps the drafting tree on the GPU (D3Tree) and encodes decoder
// steps over the tree's front nodes. Kernels in src/kernels/spec_dflash3.metal.
// Opt-in: DFlash2 constructs one only with NI_DFLASH3 set (see DFlash3::find).
#pragma once
#include "model/qwen35.h"

#include <deque>
#include <map>
#include <string>
#include <vector>

namespace ni {

// The drafting tree on the GPU, shared by the decoder step and the tree builder (a chain walk or the budgeted tree).
// Node 0 = the anchor (depth 0); a node at depth j is a draft at block row j, position P + j. Expanding a node at depth j
// runs the decoder's path row j over its branch and writes its children: the backbone's 32 candidates of block row
// j + 1 with their scores (ch_*[node * 32 + k]). Node ids are allocated in order; ancestors are expanded before
// descendants. Mirrors D3Tree in spec_dflash3.metal.
struct D3Tree {
  static constexpr int kCand = 32, kMaxNodes = 64, kMaxFront = 8;
  uint32_t n_nodes;             // nodes in the tree
  uint32_t n_front;             // nodes the next decoder step expands (<= kMaxFront)
  uint32_t front[kMaxFront];    // their ids
  uint32_t P;                   // anchor position
  uint32_t walk_at;             // the speculative chain walk (d3_jac): the chain node whose row the next pass decides
  uint32_t walk_done;           // 1 once that chain is complete
  uint32_t walk_passes;         // passes that decided rows
  uint32_t spare[2];
  int32_t parent[kMaxNodes];    // -1 for the root
  int32_t token[kMaxNodes];
  int32_t depth[kMaxNodes];
  int32_t cand_k[kMaxNodes];    // index among the parent's candidates (-1 for the root)
  float logp[kMaxNodes];        // log path probability: sum of log softmax(S) (T = 1) along the root path
  float acc[kMaxNodes];         // acceptance-head probability given the parent's path
  int32_t expanded[kMaxNodes];  // 1 once a decoder step scored its children
  int32_t row[kMaxNodes];       // verify row (the builder's)
  int32_t ch_tok[kMaxNodes * kCand];
  float ch_s[kMaxNodes * kCand];  // raw scores S = U + <z, E_out[c]>
  float ch_q[kMaxNodes * kCand];  // softmax(S / T_q)
  float ch_a[kMaxNodes * kCand];  // acceptance head
};
static_assert(sizeof(D3Tree) == 4 * (16 + 8 * 64 + 4 * 64 * 32), "D3Tree layout");

struct DFlash3Config {
  int d = 1024, NL = 2, nh = 8, hd = 128, I = 2816, H = 0, vocab = 0, ncand = 32, max_pos = 16;
  float eps = 1e-6f;
  std::string backbone;  // "dflash" | "dflash2" (informational)
};

// The chain walk fused into a decoder step (DFlash3::encode_step): the front row's child goes to tokens[P + 1 + j] and
// the walk's outputs for dfl_accept32 / dfl_depth32: q, scores [7][32], k [7]
struct D3WalkArgs {
  Buffer tokens, qprob, walk_k, walk_s;
  float temp = 0.0f;  // 0: argmax; > 0: a draw from q (qt) or the argmax with a one-hot q (!qt)
  uint32_t seed = 0;
  bool qt = true;
  int maxdep = 7;     // the child at this depth is not expanded further (n_front = 0)
};

class DFlash3 {
 public:
  static constexpr int kCand = D3Tree::kCand, kMaxNodes = D3Tree::kMaxNodes, kMaxFront = D3Tree::kMaxFront;
  DFlash3(Qwen35& target, const std::string& dir);
  // The decoder directory for NI_DFLASH3 (unset or "0": none): a directory, or "1" / "auto" = <model_dir>/dflash3, then
  // dflash3/ beside the drafter directory. "" when there is none (or its config.json does not fit the target).
  static std::string find(const std::string& model_dir, const std::string& drafter_dir, const Qwen35Config& target);
  const DFlash3Config& config() const { return cfg_; }
  size_t resident_bytes() const { return resident_; }
  const std::string& dir() const { return dir_; }

  // Node 0 = the anchor tokens[P] (Pk = 0xffffffff: P from the drafter state gst, as the GPU-driven drafting pass),
  // front = {0}. spec > 0: nodes 1 .. spec below it, each its row's top-1 candidate (cand_i), all in the front.
  void encode_root(Encoder& e, const Buffer& tokens, uint32_t Pk, const Buffer& gst, const Buffer* cand_i = nullptr,
                   int spec = 0);
  // Once per drafting pass, before the steps: W_h h of the backbone's 8 block rows from its X8 table (xh / xs: the
  // half rows of its final normed hidden and their group sums, as its LM head read them).
  void encode_prep(Encoder& e, const Buffer& xh, const Buffer& xs);
  // One decoder step over the tree's front (<= 8 nodes, read on the GPU): cand_i / cand_v = the top-32 ids / draft
  // logits of block rows 1..7, [7][32]; ch_q = softmax(S / qtemp); ch_a = the acceptance head's row acc_row (0: greedy,
  // 1: sampled; 1 when the checkpoint has one row). walk: the chain builder's step on front row 0 (D3WalkArgs).
  // ind: every dispatch indirect from the step's argument block (kSlots x MTLDispatchThreadgroupsIndirectArguments,
  // written by d3_jac: zero threadgroups once the chain is complete).
  void encode_step(Encoder& e, const Buffer& cand_i, const Buffer& cand_v, float qtemp, int acc_row,
                   const D3WalkArgs* walk = nullptr, const Buffer* ind = nullptr);
  // The chain walk as speculative passes (d3_jac): the first pass scores the backbone's top-1 chain (all rows in one
  // step), every later pass the chain below the first position where the walk left the previous guess; passes after
  // the chain is complete are skipped (indirect dispatches of zero threadgroups). passes = maxdep: always the
  // sequential walk's chain and outputs, bitwise; fewer: the positions still open after the last pass draw from that
  // pass's (stale-branch) rows, which keeps the acceptance exact.
  void encode_chain(Encoder& e, const Buffer& tokens, uint32_t Pk, const Buffer& gst, const Buffer& cand_i,
                    const Buffer& cand_v, float qtemp, int acc_row, const D3WalkArgs& walk, int passes);
  static constexpr int kSlots = 9;  // distinct dispatch shapes of a step (encode_step's indirect argument block)
  const QLinear* weight(const std::string& name) const {
    auto it = named_.find(name);
    return it == named_.end() ? nullptr : it->second;
  }
  Buffer tree() const { return tree_; }
  const D3Tree& tree_host() const { return *tree_.as<D3Tree>(); }  // after the command buffer completed

  // candidate scores: one threadgroup per row (d3_score) or per row and candidate + an epilogue kernel (d3_cand +
  // d3_soft); NI_DFLASH3_SCORE=0/1
  void set_fused_score(bool on) { fused_score_ = on; }

  // ---- tests ----
  // Per-step copies of the residual stream after the input and after each layer, of y and z, for row 0..7
  // (fp32 [8][d] / [8][H]) into named buffers: x, h<l>, y, z (the last step encoded).
  void set_debug(bool on) { debug_ = on; }
  const std::map<std::string, Buffer>& debug_bufs() const { return dbg_; }
  // The decoder's weights as dequantized fp32 rows (CPU; reference checks): name -> [N][K]
  std::vector<float> dequant(const std::string& name) const;
  std::vector<std::string> weight_names() const;

 private:
  struct Layer {
    const Tensor *ln1 = nullptr, *ln2 = nullptr;
    QLinear qkv, o, gate, up, down;
  };
  int acc_rows_ = 0;  // acceptance-head rows in the checkpoint (0: none)
  void load_config(const std::string& dir);
  const Tensor* own(const Tensor& t, const char* label);
  QLinear quantize(const std::vector<std::vector<const Tensor*>>& rows, const char* label);
  void dbg(Encoder& e, const std::string& name, const Buffer& src, size_t bytes);
  void enc_res_norm(Encoder& e, const Tensor* w, const Buffer* ind = nullptr);

  Qwen35& tg_;
  std::string dir_;
  DFlash3Config cfg_;
  size_t resident_ = 0;
  std::deque<Tensor> owned_;
  std::vector<Buffer> owned_bufs_;
  std::map<std::string, const QLinear*> named_;
  QLinear w_h_, w_e_, w_out_;
  std::vector<Layer> layers_;
  const Tensor *norm_ = nullptr, *pos_ = nullptr, *emb_norm_ = nullptr;
  Buffer lscale_;          // logit_scale [16] fp32 (ones when absent)
  Buffer acc_wy_, acc_we_;  // acceptance head rows [2][d], [2][H] fp32
  float acc_wq_[2] = {0, 0}, acc_b_[2] = {0, 0};
  // step buffers (8 rows): A = W_h h of the pass's block rows, cd = candidate dot products [8][32][4]
  Buffer tree_, kv_, A_, xin_h_, xin_s_, hres_, n_, xh_, xs_, hh_, hs_, u_, qkv_, z_, cd_;
  // speculative passes: the full indirect arguments of a step (kSlots x 3 uints, filled at encode time) and one
  // argument block per pass (d3_jac writes the next pass's)
  Buffer full_args_, pass_args_;
  bool debug_ = false, fused_score_ = true;
  std::map<std::string, Buffer> dbg_;
};

} // namespace ni

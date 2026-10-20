// KFlash candidate decoder (internal/KFLASH.md §1, §4): a small autoregressive decoder that scores the
// block drafter's top-32 candidates of each position given the whole drafted path, replacing DFlash2's first-order
// selector. Loads <model>/kflash in the trainer's format (M2's README: bf16 projections, quantized at
// load to 4-bit group 64 with mx.quantize's rounding as the DFlash2 drafter's (NI_KFLASH_BITS=8: 8-bit); the embedding norm, per-row logit
// scales and the two-row acceptance head as stored), keeps the drafting tree on the GPU (KFlashTree) and encodes decoder
// steps over the tree's front nodes. Kernels in src/kernels/spec_kflash.metal. Its trees go through the shared KFlash
// builder and descriptor (src/kernels/kflash.h, encode_tree); its chain walk drafts where a chain is verified.
// Opt-in: DFlash2 constructs one only with NI_KFLASH set (see KFlash::find).
#pragma once
#include "model/kflash_budget.h"
#include "model/qwen35.h"

#include <deque>
#include <map>
#include <string>
#include <vector>

namespace ni {

namespace kp {  // the decoder kernels' parameters (src/kernels/spec_kflash.metal; the other kernels' are in kparams.h)
struct KfRoot {
  uint32_t P, spec;
};

struct KfJac {
  float temp;
  uint32_t seed, qt, maxdep, last, nslot;
};

struct KfIn {
  uint32_t H, d, emb_bits, emb_gs;
  float eps;
};

struct KfAttn {
  uint32_t d, layer, nl;
  float scale;
};

struct KfCand {
  uint32_t H, out_bits;
};

struct KfSoft {
  uint32_t H, d, acc_row, has_acc;
  float qtemp, wq, wb, temp;
  uint32_t seed, qt, maxdep, walk;
  uint32_t out_bits;
};

struct KfTreeNext {
  uint32_t P, init, maxdep, nslot;
};

struct KfDtab {
  uint32_t use_acc, pad;
};
}  // namespace kp

// The drafting tree on the GPU, shared by the decoder step and the tree builder (a chain walk or the budgeted tree).
// Node 0 = the anchor (depth 0); a node at depth j is a draft at block row j, position P + j. Expanding a node at depth j
// runs the decoder's path row j over its branch and writes its children: the backbone's 32 candidates of block row
// j + 1 with their scores (ch_*[node * 32 + k]). Node ids are allocated in order; ancestors are expanded before
// descendants. Mirrors KFlashTree in spec_kflash.metal.
struct KFlashTree {
  static constexpr int kCand = 32, kMaxNodes = 128, kMaxFront = 8;
  uint32_t n_nodes;             // nodes in the tree
  uint32_t n_front;             // nodes the next decoder step expands (<= kMaxFront)
  uint32_t front[kMaxFront];    // their ids
  uint32_t P;                   // anchor position
  uint32_t walk_at;             // the speculative chain walk (kf_jac): the chain node whose row the next pass decides
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
  float ch_s[kMaxNodes * kCand];  // scores S = logit_scale U + <z, E_out[c]>
  float ch_q[kMaxNodes * kCand];  // softmax(S / T_q)
  float ch_a[kMaxNodes * kCand];  // acceptance head
};
static_assert(sizeof(KFlashTree) == 4 * (16 + 8 * 128 + 4 * 128 * 32), "KFlashTree layout");

struct KFlashConfig {
  int d = 1024, NL = 2, nh = 8, hd = 128, I = 2816, H = 0, vocab = 0, ncand = 32, max_pos = 16;
  float eps = 1e-6f;
  std::string backbone;  // "dflash" | "dflash2" (informational)
};

// The chain walk fused into a decoder step (KFlash::encode_step): the front row's child goes to tokens[P + 1 + j] and
// the walk's outputs for dfl_accept32 / dfl_depth32: q, scores [7][32], k [7]
struct KFlashWalk {
  Buffer tokens, qprob, walk_k, walk_s;
  float temp = 0.0f;  // 0: argmax; > 0: a draw from q (qt) or the argmax with a one-hot q (!qt)
  uint32_t seed = 0;
  bool qt = true;
  int maxdep = 7;     // the child at this depth is not expanded further (n_front = 0)
};

class KFlash {
 public:
  static constexpr int kCand = KFlashTree::kCand, kMaxNodes = KFlashTree::kMaxNodes, kMaxFront = KFlashTree::kMaxFront;
  KFlash(Qwen35& target, const std::string& dir);
  // The decoder directory for NI_KFLASH (unset or "0": none): a directory, or "1" / "auto" = <model_dir>/kflash, then
  // kflash/ beside the drafter directory (and dflash3/ in both places: checkpoints exported before the rename). ""
  // when there is none (or its config.json does not fit the target).
  static std::string find(const std::string& model_dir, const std::string& drafter_dir, const Qwen35Config& target);
  const KFlashConfig& config() const { return cfg_; }
  size_t resident_bytes() const { return resident_; }
  const std::string& dir() const { return dir_; }

  // Node 0 = the anchor tokens[P] (Pk = 0xffffffff: P from the drafter state gst, as the GPU-driven drafting pass),
  // front = {0}. spec > 0: nodes 1 .. spec below it, each its row's top-1 candidate (cand_i), all in the front.
  void encode_root(Encoder& e, const Buffer& tokens, uint32_t Pk, const Buffer& gst, const Buffer* cand_i = nullptr,
                   int spec = 0);
  // ---- with a DFlash2 drafter (DFlash2 does this when NI_KFLASH finds a decoder) ----
  // The drafter's draft head (the candidates' E_out rows, by column; 4 / 8-bit for every target format), its column ->
  // token id map (null: the full vocabulary) and its row count vd; allocates the candidate and walk buffers.
  void bind_drafter(const QLinear* head, const Buffer* head_ids, uint32_t vd);
  // The end of a drafting pass: the top-32 of the draft logit rows 1..7 (logits fp32 [7][vd]) -> cand_ids / cand_cols /
  // cand_vals, W_h h from the pass's X8 table (xh / xs), and with chain the chain walk (NI_KFLASH_PASSES speculative
  // passes, default 7 = the sequential walk's result; 0: single-row steps): tokens[P + 1 ..], walk_q / walk_scores and
  // walk_k for dfl_accept32 / dfl_depth32. The walk's q at T > 0 is softmax(S / (walk_tf T)) (walk_tf 0: the argmax
  // with a one-hot q), at T = 0 the children's q is softmax(S); the acceptance head's row by T.
  void encode_draft(Encoder& e, const Buffer& logits, const Buffer& xh, const Buffer& xs, const Buffer& tokens,
                    uint32_t Pk, const Buffer& gst, const Buffer& walk_k, const SamplingParams& sp, float walk_tf,
                    bool chain);
  Buffer cand_ids() const { return c_i_; }      // [7][32] token ids of block rows 1..7
  Buffer cand_cols() const { return c_col_; }   // their draft-head columns
  Buffer cand_vals() const { return c_v_; }     // their draft logits
  Buffer walk_q() const { return w_q_; }        // [7][32] the chain walk's q
  Buffer walk_scores() const { return w_s_; }   // [7][32] its scores S

  // Once per drafting pass, before the steps: W_h h of the backbone's 8 block rows from its X8 table (xh / xs: the
  // half rows of its final normed hidden and their group sums, as its LM head read them).
  void encode_prep(Encoder& e, const Buffer& xh, const Buffer& xs);
  // One decoder step over the tree's front (<= 8 nodes, read on the GPU): cand_i / cand_col / cand_v = the top-32 token
  // ids, their rows in the out head (set_out_head) and their draft logits, block rows 1..7, [7][32]; ch_q =
  // softmax(S / qtemp); ch_a = the acceptance head's row acc_row (0: greedy, 1: sampled; 0 when the checkpoint has
  // one row). walk: the chain builder's step on front row 0 (KFlashWalk).
  // ind: every dispatch indirect from the step's argument block (kSlots x MTLDispatchThreadgroupsIndirectArguments,
  // written by kf_jac: zero threadgroups once the chain is complete).
  void encode_step(Encoder& e, const Buffer& cand_i, const Buffer& cand_col, const Buffer& cand_v, float qtemp,
                   int acc_row, const KFlashWalk* walk = nullptr, const Buffer* ind = nullptr);
  // The table the candidates' E_out rows come from, indexed by cand_col: the drafter's draft head (the target's LM head,
  // its reduced rows, or their 8-bit copy for heads in other formats; 4 / 8-bit group 64). Default: the target's LM
  // head with cand_col = token ids (4 / 8-bit heads only).
  void set_out_head(const QLinear* head) { out_head_ = head; }
  // tests: the embedding table node tokens are read from (default: the target's; any format format_supported takes)
  void set_embed_table(const QLinear* emb) { emb_ = emb; }
  // The chain walk as speculative passes (kf_jac): the first pass scores the backbone's top-1 chain (all rows in one
  // step), every later pass the chain below the first position where the walk left the previous guess; passes after
  // the chain is complete are skipped (indirect dispatches of zero threadgroups). passes = maxdep: always the
  // sequential walk's chain and outputs, bitwise; fewer: the positions still open after the last pass draw from that
  // pass's (stale-branch) rows, which keeps the acceptance exact.
  void encode_chain(Encoder& e, const Buffer& tokens, uint32_t Pk, const Buffer& gst, const Buffer& cand_i,
                    const Buffer& cand_col, const Buffer& cand_v, float qtemp, int acc_row, const KFlashWalk& walk,
                    int passes);
  static constexpr int kSlots = 9;  // distinct dispatch shapes of a step (encode_step's indirect argument block)
  // ---- trees (the shared KFlash design: descriptor, builder kf_build and budget, src/kernels/kflash.h) ----
  // The end of a tree cycle's drafting pass, after the shared builder took a first tree from the drafter's lattice
  // into desc (and its tokens): the top-32 and W_h h as encode_draft, then passes that each map the tree's rows onto
  // decoder nodes, expand the ones not expanded yet (up to 8, ancestors first) in one decoder step, and have the
  // builder take the tree again from the decoder's children tables: rows = decoder nodes, children = the 32
  // candidates of the next block row with the log of q = softmax(S / T) (T = 0: softmax(S)), calibrated per depth by
  // the budget, a child expandable once the decoder expanded it. tree_passes() passes at most (default 3); passes
  // after every node of the tree is expanded are skipped (indirect dispatches of zero threadgroups): the tree is then
  // the builder's best-first tree over the decoder's own probabilities. kp: the budget's parameters (calibration,
  // costs, row counts) as for the lattice; Pk = 0xffffffff: the anchor from the drafter state (kf_build_gpu).
  // The tree's acceptance (kf_accept) is exact whatever the tree, so the decoder only decides its shape.
  void encode_tree(Encoder& e, const Buffer& logits, const Buffer& xh, const Buffer& xs, const Buffer& tokens,
                   uint32_t Pk, const Buffer& gst, const SamplingParams& sp, const Buffer& desc, const Buffer& live,
                   const KfParams& kp);
  // encode_tree's passes alone (tests: after a pass that already took the top-32 and W_h h; desc may hold any first
  // tree, the anchor alone included)
  void encode_tree_passes(Encoder& e, const Buffer& tokens, uint32_t Pk, const Buffer& gst, const SamplingParams& sp,
                          const Buffer& desc, const Buffer& live, const KfParams& kp);
  static constexpr int kTreeSlots = kSlots + 2;  // a tree pass's dispatch shapes: the step's, kf_dtab, kf_build
  // the children tables of the last tree pass (tests): clq float, ctok / cnext int, [kMaxNodes][kCand]
  Buffer tree_clq() const { return t_clq_; }
  Buffer tree_tok() const { return t_tok_; }
  Buffer tree_next() const { return t_next_; }
  // NI_KFLASH_PASSES, or the defaults: the chain walk's speculative passes (7: the sequential walk's chain, bitwise;
  // 0: single-row steps) and the tree's decoder passes (3: with the first trained 4B decoder the trees changed little
  // past 2-3 passes and code_27b paid every extra step; the agent requests measured then share its training
  // conversation, so the cap needs a recheck on unseen agent turns)
  int chain_passes() const { return passes_ < 0 ? 7 : passes_; }
  int tree_passes() const { return passes_ > 0 ? std::min(passes_, 16) : 3; }
  const QLinear* weight(const std::string& name) const {
    auto it = named_.find(name);
    return it == named_.end() ? nullptr : it->second;
  }
  Buffer tree() const { return tree_; }
  const KFlashTree& tree_host() const { return *tree_.as<KFlashTree>(); }  // after the command buffer completed

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
  int qbits_ = 4;     // the projections' width at load (NI_KFLASH_BITS)
  const QLinear* out_head_ = nullptr;
  const QLinear* emb_ = nullptr;
  void load_config(const std::string& dir);
  const Tensor* own(const Tensor& t, const char* label);
  QLinear quantize(const std::vector<std::vector<const Tensor*>>& rows, const char* label);
  void dbg(Encoder& e, const std::string& name, const Buffer& src, size_t bytes);
  void enc_res_norm(Encoder& e, const Tensor* w, const Buffer* ind = nullptr);

  Qwen35& tg_;
  std::string dir_;
  KFlashConfig cfg_;
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
  // argument block per pass (kf_jac writes the next pass's)
  Buffer full_args_, pass_args_;
  // trees: the decoder's children tables for the shared builder and one argument block per pass (kf_tree_next)
  Buffer t_clq_, t_tok_, t_next_, tree_args_;
  // bind_drafter: the top-32 (ids, draft-head columns, logits), its staging, the chain walk's q and scores
  const Buffer* head_ids_ = nullptr;
  uint32_t vd_ = 0;
  int passes_ = -1;  // NI_KFLASH_PASSES (-1: unset)
  Buffer c_i_, c_col_, c_v_, w_q_, w_s_, t_v_[2], t_i_[2];
  void enc_top32(Encoder& e, const Buffer& logits);
  bool debug_ = false;
  std::map<std::string, Buffer> dbg_;
};

} // namespace ni

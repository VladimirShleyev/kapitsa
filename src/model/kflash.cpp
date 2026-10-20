// KFlash candidate decoder: loading (bf16 checkpoint -> 4-bit group 64 at load), the GPU drafting tree and the
// decoder step. Kernels in src/kernels/spec_kflash.metal; the GEMMs are the DFlash2 drafter's 8-row ones.
#include "model/kflash.h"
#include "model/dflash2.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unordered_map>

namespace ni {

using json = nlohmann::json;

static Pipeline kfpipe(const std::string& n) {
  static std::unordered_map<std::string, Pipeline> cache;
  auto it = cache.find(n);
  if (it == cache.end()) it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  return it->second;
}

// The decoder's config.json against the target (the kernels fix head_dim 128, 32 candidates, <= 16 path rows): the
// failed checks, empty for a supported decoder.
static std::vector<std::string> kf_config_problems(const json& c, const Qwen35Config& t) {
  std::vector<std::string> bad;
  auto check = [&](bool ok, const char* what) {
    if (!ok) bad.push_back(what);
  };
  const int d = c.value("hidden_size", 0), nh = c.value("num_attention_heads", 0), hd = c.value("head_dim", 0);
  const int I = c.value("intermediate_size", 0), H = c.value("target_hidden_size", 0), NL = c.value("num_hidden_layers", 0);
  const int mp = c.value("max_positions", 0);
  const json arch = c.value("architectures", json::array());  // DFlash3CandidateDecoder: exported before the rename
  check(arch == json::array({"KFlashCandidateDecoder"}) || arch == json::array({"DFlash3CandidateDecoder"}),
        "architecture");
  check(d > 0 && d % 64 == 0 && hd == 128 && nh * hd == d, "hidden = heads x 128, a multiple of 64");
  check(I > 0 && I % 64 == 0, "intermediate a multiple of 64");
  check(NL >= 1 && NL <= 8, "1..8 layers");
  check(H == t.hidden && H % 64 == 0, "target_hidden_size == target hidden");
  check(c.value("vocab_size", 0) == t.vocab, "vocab == target vocab");
  check(c.value("num_candidates", 0) == KFlash::kCand, "32 candidates");
  check(mp >= DFlash2::kB - 1 && mp <= 16, "max_positions 7..16");
  check(c.value("rms_norm_eps", 1e-6) > 0, "eps");
  check(c.value("format_version", 1) == 1, "format_version 1");
  return bad;
}

std::string KFlash::find(const std::string& model_dir, const std::string& drafter_dir, const Qwen35Config& t) {
  namespace fs = std::filesystem;
  const char* v = getenv("NI_KFLASH");
  if (!v || !*v || !strcmp(v, "0")) return "";
  auto problems = [&](const fs::path& d, std::vector<std::string>* bad) {
    std::ifstream f(d / "config.json");
    if (!f) return false;
    try {
      *bad = kf_config_problems(json::parse(f), t);
      return true;
    } catch (...) {
      bad->assign(1, "config.json does not parse");
      return true;
    }
  };
  std::vector<std::string> bad;
  if (strcmp(v, "1") && strcmp(v, "auto")) {  // an explicit directory must fit
    if (!problems(v, &bad)) fatal("NI_KFLASH=%s: no config.json", v);
    if (!bad.empty()) {
      std::string s;
      for (auto& b : bad) s += " [" + b + "]";
      fatal("NI_KFLASH=%s: unsupported decoder:%s", v, s.c_str());
    }
    return v;
  }
  std::vector<fs::path> cands;
  for (const char* sub : {"kflash", "dflash3"}) {
    cands.push_back(fs::path(model_dir) / sub);
    if (!drafter_dir.empty()) cands.push_back(fs::path(drafter_dir).parent_path() / sub);
  }
  for (auto& c : cands)
    if (problems(c, &bad) && bad.empty()) return c.string();
  return "";
}

void KFlash::load_config(const std::string& dir) {
  std::ifstream f(dir + "/config.json");
  if (!f) fatal("KFlash: cannot read %s/config.json", dir.c_str());
  const json c = json::parse(f);
  const auto bad = kf_config_problems(c, tg_.config());
  if (!bad.empty()) {
    std::string s;
    for (auto& b : bad) s += " [" + b + "]";
    fatal("KFlash: unsupported decoder config in %s:%s", dir.c_str(), s.c_str());
  }
  cfg_.d = c["hidden_size"];
  cfg_.NL = c["num_hidden_layers"];
  cfg_.nh = c["num_attention_heads"];
  cfg_.hd = c["head_dim"];
  cfg_.I = c["intermediate_size"];
  cfg_.H = c["target_hidden_size"];
  cfg_.vocab = c["vocab_size"];
  cfg_.ncand = c["num_candidates"];
  cfg_.max_pos = c["max_positions"];
  cfg_.eps = c.value("rms_norm_eps", 1e-6f);
  cfg_.backbone = c.value("backbone", std::string());
}

const Tensor* KFlash::own(const Tensor& t, const char* label) {
  Buffer b = Gpu::get().alloc(t.buf.size, label, /*tracked=*/false);
  memcpy(b.ptr(), t.buf.ptr(), t.buf.size);
  owned_bufs_.push_back(b);
  resident_ += b.size;
  Tensor c = t;
  c.buf = b;
  owned_.push_back(c);
  return &owned_.back();
}

// rows: blocks stacked along N, each block the parts concatenated along K (bf16 2-D, equal rows within a block, equal
// K over the blocks) -> MLX affine 4-bit group 64 with mx.quantize's rounding (dfl_quantize_cpu; NI_DFLASH_QROUND=even:
// MLX's CPU backend), as the DFlash2 drafter's projections.
QLinear KFlash::quantize(const std::vector<std::vector<const Tensor*>>& blocks, const char* label) {
  uint32_t N = 0, K = 0;
  for (size_t b = 0; b < blocks.size(); ++b) {
    uint32_t kb = 0;
    for (const Tensor* t : blocks[b]) {
      if (t->dtype != DType::BF16 || t->shape.size() != 2 || t->shape[0] != blocks[b][0]->shape[0])
        fatal("KFlash quantize %s: parts must be 2-D bf16 with equal rows", label);
      kb += uint32_t(t->shape[1]);
    }
    if (b == 0) K = kb;
    else if (kb != K) fatal("KFlash quantize %s: blocks of different widths", label);
    N += uint32_t(blocks[b][0]->shape[0]);
  }
  if (K % 64 || N % 8) fatal("KFlash quantize %s: [%u, %u] not a multiple of (8, 64)", label, N, K);
  const int bits = qbits_;
  const uint32_t wpr = K * bits / 32, gpr = K / 64;
  // horizontally concatenated blocks are copied into `cat` (sized first, so the row pointers stay valid)
  size_t cat_rows = 0;
  for (auto& blk : blocks)
    if (blk.size() > 1) cat_rows += size_t(blk[0]->shape[0]);
  std::vector<uint16_t> cat(cat_rows * K);
  std::vector<const uint16_t*> rowp;
  rowp.reserve(N);
  size_t ci = 0;
  for (auto& blk : blocks) {
    const int64_t n = blk[0]->shape[0];
    for (int64_t i = 0; i < n; ++i) {
      if (blk.size() == 1) {
        rowp.push_back(blk[0]->buf.as<uint16_t>() + size_t(i) * K);
        continue;
      }
      uint16_t* dst = cat.data() + (ci + size_t(i)) * K;
      size_t col = 0;
      for (const Tensor* t : blk) {
        const size_t w = size_t(t->shape[1]);
        memcpy(dst + col, t->buf.as<uint16_t>() + size_t(i) * w, w * 2);
        col += w;
      }
      rowp.push_back(dst);
    }
    if (blk.size() > 1) ci += size_t(n);
  }
  Gpu& g = Gpu::get();
  Buffer wq = g.alloc(size_t(N) * wpr * 4, label, false);
  Buffer sc = g.alloc(size_t(N) * gpr * 2, label, false);
  Buffer bi = g.alloc(size_t(N) * gpr * 2, label, false);
  static const bool even = getenv("NI_DFLASH_QROUND") && std::string(getenv("NI_DFLASH_QROUND")) == "even";
  dfl_quantize_cpu(rowp, K, bits, even, wq.as<uint32_t>(), sc.as<uint16_t>(), bi.as<uint16_t>());
  for (Buffer* b : {&wq, &sc, &bi}) {
    owned_bufs_.push_back(*b);
    resident_ += b->size;
  }
  QLinear L;
  owned_.push_back(Tensor{wq, DType::U32, {int64_t(N), int64_t(wpr)}});
  L.w = &owned_.back();
  owned_.push_back(Tensor{sc, DType::BF16, {int64_t(N), int64_t(gpr)}});
  L.s = &owned_.back();
  owned_.push_back(Tensor{bi, DType::BF16, {int64_t(N), int64_t(gpr)}});
  L.b = &owned_.back();
  L.N = N;
  L.K = K;
  L.bits = bits;
  L.gs = 64;
  return L;
}

KFlash::KFlash(Qwen35& target, const std::string& dir) : tg_(target), dir_(dir) {
  const double t0 = now_ms();
  // NI_KFLASH_BITS=8: the projections as 8-bit group 64 (twice the bytes a step reads); default 4
  if (const char* v = getenv("NI_KFLASH_BITS"); v && atoi(v) == 8) qbits_ = 8;
  load_config(dir);
  // the target's embedding rows (node tokens) in any format the engine runs; the candidates' LM-head rows come from a
  // 4 / 8-bit table (set_out_head: DFlash2's draft head, which is one for every target format)
  const QLinear& E = tg_.embed_table();
  const QLinear& O = tg_.lm_head();
  if (!(E.q4() || E.q8() || format_supported(E.bits, E.gs)) || E.K != uint32_t(cfg_.H))
    fatal("KFlash: the target's embedding format (%s) is not supported", format_name(E.bits, E.gs).c_str());
  if ((O.q4() || O.q8()) && O.K == uint32_t(cfg_.H)) out_head_ = &O;
  const int64_t d = cfg_.d, H = cfg_.H, I = cfg_.I;
  {
    WeightStore w;  // bf16 checkpoint, released after quantization
    w.open_dir(dir);
    auto expect = [&](const std::string& n, std::vector<int64_t> shape) -> const Tensor& {
      const Tensor& t = w.get(n);
      if (t.dtype != DType::BF16 || t.shape != shape) fatal("KFlash: %s has an unexpected dtype/shape", n.c_str());
      return t;
    };
    w_h_ = quantize({{&expect("in_proj.weight", {d, H})}}, "kf_in");
    w_e_ = quantize({{&expect("emb_proj.weight", {d, H})}}, "kf_emb");
    named_["in"] = &w_h_;
    named_["emb"] = &w_e_;
    emb_norm_ = own(expect("emb_norm.weight", {H}), "kf_norm");
    pos_ = own(expect("pos_embed", {int64_t(cfg_.max_pos), d}), "kf_pos");
    layers_.resize(cfg_.NL);
    for (int l = 0; l < cfg_.NL; ++l) {
      Layer& L = layers_[l];
      const std::string p = "layers." + std::to_string(l) + ".";
      L.ln1 = own(expect(p + "input_layernorm.weight", {d}), "kf_norm");
      L.ln2 = own(expect(p + "post_attention_layernorm.weight", {d}), "kf_norm");
      L.qkv = quantize({{&expect(p + "self_attn.q_proj.weight", {d, d})},
                        {&expect(p + "self_attn.k_proj.weight", {d, d})},
                        {&expect(p + "self_attn.v_proj.weight", {d, d})}},
                       "kf_qkv");
      L.o = quantize({{&expect(p + "self_attn.o_proj.weight", {d, d})}}, "kf_o");
      L.gate = quantize({{&expect(p + "mlp.gate_proj.weight", {I, d})}}, "kf_gate");
      L.up = quantize({{&expect(p + "mlp.up_proj.weight", {I, d})}}, "kf_up");
      L.down = quantize({{&expect(p + "mlp.down_proj.weight", {d, I})}}, "kf_down");
    }
    for (int l = 0; l < cfg_.NL; ++l) {
      const std::string p = "layers." + std::to_string(l) + ".";
      named_[p + "qkv"] = &layers_[l].qkv;
      named_[p + "o"] = &layers_[l].o;
      named_[p + "gate"] = &layers_[l].gate;
      named_[p + "up"] = &layers_[l].up;
      named_[p + "down"] = &layers_[l].down;
    }
    norm_ = own(expect("norm.weight", {d}), "kf_norm");
    w_out_ = quantize({{&expect("out_proj.weight", {H, d})}}, "kf_out");
    named_["out"] = &w_out_;
    // fp32 extras (bf16 accepted): logit_scale [max_positions], acceptance head [rows][d + H + 1] + bias [rows]
    auto fval = [&](const Tensor* t, int64_t i) {
      return t->dtype == DType::F32 ? t->buf.as<float>()[i] : bf16_to_f32(t->buf.as<uint16_t>()[i]);
    };
    auto fp = [&](const Tensor* t) { return t->dtype == DType::F32 || t->dtype == DType::BF16; };
    Gpu& g = Gpu::get();
    lscale_ = g.alloc(16 * 4, "kf_lscale", false);
    for (int j = 0; j < 16; ++j) lscale_.as<float>()[j] = 1.0f;
    if (const Tensor* ls = w.find("logit_scale")) {
      if (!fp(ls) || ls->numel() != cfg_.max_pos) fatal("KFlash: logit_scale must hold max_positions fp32/bf16 values");
      for (int j = 0; j < cfg_.max_pos; ++j) lscale_.as<float>()[j] = fval(ls, j);
    }
    acc_wy_ = g.alloc(size_t(2 * d) * 4, "kf_acc", false);
    acc_we_ = g.alloc(size_t(2 * H) * 4, "kf_acc", false);
    memset(acc_wy_.ptr(), 0, acc_wy_.size);
    memset(acc_we_.ptr(), 0, acc_we_.size);
    if (const Tensor* aw = w.find("acc_head.weight")) {
      const int64_t W = d + H + 1;
      if (!fp(aw) || aw->numel() % W || aw->numel() / W < 1 || aw->numel() / W > 2)
        fatal("KFlash: acc_head.weight must be [1 or 2][d + H + 1 = %lld] fp32/bf16", (long long)W);
      acc_rows_ = int(aw->numel() / W);
      const Tensor* ab = w.find("acc_head.bias");
      if (!ab || !fp(ab) || ab->numel() != acc_rows_) fatal("KFlash: acc_head.bias must hold one value per head row");
      for (int r = 0; r < 2; ++r) {
        const int rs = std::min(r, acc_rows_ - 1);  // one row: both modes use it
        for (int64_t i = 0; i < d; ++i) acc_wy_.as<float>()[r * d + i] = fval(aw, rs * W + i);
        for (int64_t i = 0; i < H; ++i) acc_we_.as<float>()[r * H + i] = fval(aw, rs * W + d + i);
        acc_wq_[r] = fval(aw, rs * W + d + H);
        acc_b_[r] = fval(ab, rs);
      }
    }
    for (Buffer* b : {&lscale_, &acc_wy_, &acc_we_}) {
      owned_bufs_.push_back(*b);
      resident_ += b->size;
    }
  }
  Gpu& g = Gpu::get();
  const size_t Kx = std::max<size_t>(d, I);
  tree_ = g.alloc(sizeof(KFlashTree), "kf_tree");
  memset(tree_.ptr(), 0, tree_.size);
  kv_ = g.alloc(size_t(kMaxNodes) * cfg_.NL * 2 * d * 4, "kf_kv");
  A_ = g.alloc(8 * size_t(d) * 4, "kf_A");
  cd_ = g.alloc(8 * size_t(kCand) * 16, "kf_cd");
  xin_h_ = g.alloc(16 * size_t(H) * 2, "kf_xin_h");
  xin_s_ = g.alloc(size_t(H) / 64 * 8 * 4, "kf_xin_s");
  hres_ = g.alloc(8 * size_t(d) * 4, "kf_h");
  n_ = g.alloc(8 * size_t(d) * 4, "kf_n");
  xh_ = g.alloc(16 * Kx * 2, "kf_xh");
  xs_ = g.alloc(Kx / 64 * 8 * 4, "kf_xs");
  hh_ = g.alloc(16 * size_t(I) * 2, "kf_hh");
  hs_ = g.alloc(size_t(I) / 64 * 8 * 4, "kf_hs");
  u_ = g.alloc(8 * size_t(d) * 4, "kf_u");
  qkv_ = g.alloc(8 * size_t(3 * d) * 4, "kf_qkv");
  z_ = g.alloc(8 * size_t(H) * 4, "kf_z");
  full_args_ = g.alloc(kTreeSlots * 12, "kf_args");
  pass_args_ = g.alloc(size_t(16) * kSlots * 12, "kf_args");
  t_clq_ = g.alloc(size_t(kMaxNodes) * kCand * 4, "kf_tree_clq");
  t_tok_ = g.alloc(size_t(kMaxNodes) * kCand * 4, "kf_tree_tok");
  t_next_ = g.alloc(size_t(kMaxNodes) * kCand * 4, "kf_tree_next");
  tree_args_ = g.alloc(size_t(16) * kTreeSlots * 12, "kf_tree_args");
  for (Buffer* b : {&tree_, &kv_, &A_, &cd_, &xin_h_, &xin_s_, &hres_, &n_, &xh_, &xs_, &hh_, &hs_, &u_, &qkv_, &z_,
                    &full_args_, &pass_args_, &t_clq_, &t_tok_, &t_next_, &tree_args_}) {
    memset(b->ptr(), 0, b->size);
    resident_ += b->size;
  }
  g.make_resident(owned_bufs_);
  logi("KFlash decoder %s: %d layers, d %d, %d heads x %d, MLP %d, target hidden %d, %d candidates%s, %.1f MB resident, "
       "loaded in %.0f ms", dir.c_str(), cfg_.NL, cfg_.d, cfg_.nh, cfg_.hd, cfg_.I, cfg_.H, kCand,
       acc_rows_ == 2 ? ", acceptance head (greedy, sampled)" : acc_rows_ ? ", acceptance head" : "", resident_ / 1e6,
       now_ms() - t0);
}

void KFlash::dbg(Encoder& e, const std::string& name, const Buffer& src, size_t bytes) {
  if (!debug_) return;
  auto it = dbg_.find(name);
  if (it == dbg_.end()) it = dbg_.emplace(name, Gpu::get().alloc((bytes + 15) / 16 * 16, "kf_dbg")).first;
  const uint32_t n = uint32_t(bytes / 16);
  e.use(kfpipe("copy_u32"));
  e.buf(0, src);
  e.buf(1, it->second);
  e.val(2, n);
  e.dispatch((n + 255) / 256, 1, 1, 256);
}

void KFlash::encode_root(Encoder& e, const Buffer& tokens, uint32_t Pk, const Buffer& gst, const Buffer* cand_i,
                          int spec) {
  if (spec > 0 && !cand_i) fatal("KFlash::encode_root: a speculated chain needs the candidates");
  const kp::KfRoot p{Pk, uint32_t(std::clamp(spec, 0, kMaxFront - 1))};
  e.use(kfpipe("kf_root"));
  e.buf(0, tree_);
  e.buf(1, tokens);
  e.val(2, p);
  e.buf(3, gst);
  e.buf(4, cand_i ? *cand_i : tree_);
  e.dispatch(1, 1, 1, 32);
}

// h += u; n = RMSNorm(h; w) -> fp32 n and the X8 table (dfl_res_norm over the decoder's width); ind: slot 0
void KFlash::enc_res_norm(Encoder& e, const Tensor* w, const Buffer* ind) {
  kp::DflNorm np{uint32_t(cfg_.d), cfg_.eps, 1.0f, 1u};
  e.use(kfpipe("dfl_res_norm"));
  e.buf(0, hres_); e.buf(1, u_); e.buf(2, w->buf); e.buf(3, n_); e.buf(4, xh_); e.buf(5, xs_);
  e.val(6, np);
  if (ind) e.dispatch_indirect(*ind, 0, 256);
  else e.dispatch(8, 1, 1, 256);
}

void KFlash::bind_drafter(const QLinear* head, const Buffer* head_ids, uint32_t vd) {
  out_head_ = head;
  head_ids_ = head_ids;
  vd_ = vd;
  Gpu& g = Gpu::get();
  const size_t ntg = (size_t(vd) + 2047) / 2048, ntg2 = (ntg * 32 + 2047) / 2048;
  c_i_ = g.alloc(7 * kCand * 4, "kf_cand_i");
  c_col_ = g.alloc(7 * kCand * 4, "kf_cand_col");
  c_v_ = g.alloc(7 * kCand * 4, "kf_cand_v");
  w_q_ = g.alloc(7 * kCand * 4, "kf_walk_q");
  w_s_ = g.alloc(7 * kCand * 4, "kf_walk_s");
  t_v_[0] = g.alloc(7 * ntg * kCand * 4, "kf_top_v");
  t_i_[0] = g.alloc(7 * ntg * kCand * 4, "kf_top_i");
  t_v_[1] = g.alloc(7 * ntg2 * kCand * 4, "kf_top_v");
  t_i_[1] = g.alloc(7 * ntg2 * kCand * 4, "kf_top_i");
  for (Buffer* b : {&c_i_, &c_col_, &c_v_, &w_q_, &w_s_, &t_v_[0], &t_i_[0], &t_v_[1], &t_i_[1]}) {
    memset(b->ptr(), 0, b->size);
    resident_ += b->size;
  }
  // NI_KFLASH_PASSES: speculative passes of the chain walk (default 7 = the sequential walk's result, passes after the
  // chain is complete skipped); 0 = one single-row decoder step per position
  if (const char* v = getenv("NI_KFLASH_PASSES")) passes_ = std::max(0, atoi(v));
}

// Top-32 of the draft logit rows 1..7, ordered (value desc, id asc) like dfl_top16: kf_top32 over chunks of 2048, then
// over the chunks' top-32s until one chunk remains -> c_v_ (draft logits), c_col_ (draft-head columns, the E_out rows)
// and c_i_ (token ids) [7][32].
void KFlash::enc_top32(Encoder& e, const Buffer& logits) {
  Buffer src = logits, sid = t_i_[0];
  uint32_t n = vd_, has = 0;
  int to = 0;
  for (;;) {
    const uint32_t chunks = (n + 2047) / 2048;
    const Buffer& ov = chunks == 1 ? c_v_ : t_v_[to];
    const Buffer& oi = chunks == 1 ? c_col_ : t_i_[to];
    const uint32_t map = chunks == 1 ? (head_ids_ ? 1u : 2u) : 0u;
    kp::DflTop p{n, n, chunks};
    e.use(kfpipe("kf_top32"));
    e.buf(0, src); e.buf(1, sid); e.buf(2, ov); e.buf(3, oi);
    e.val(4, p);
    e.val(5, has);
    e.buf(6, head_ids_ ? *head_ids_ : c_i_); e.buf(7, c_i_);
    e.val(8, map);
    e.dispatch(chunks, 7, 1, 256);
    if (chunks == 1) break;
    src = ov;
    sid = oi;
    has = 1;
    n = chunks * kCand;
    to ^= 1;
  }
}

void KFlash::encode_draft(Encoder& e, const Buffer& logits, const Buffer& xh, const Buffer& xs, const Buffer& tokens,
                          uint32_t Pk, const Buffer& gst, const Buffer& walk_k, const SamplingParams& sp, float walk_tf,
                          bool chain) {
  if (!vd_) fatal("KFlash::encode_draft: no drafter bound (bind_drafter)");
  enc_top32(e, logits);
  encode_prep(e, xh, xs);
  if (!chain) return;
  const bool sampled = sp.temperature > 0.0f;
  const float tq = sampled && walk_tf > 0.0f ? sp.temperature * walk_tf : 1.0f;
  KFlashWalk w;
  w.tokens = tokens;
  w.qprob = w_q_;
  w.walk_k = walk_k;
  w.walk_s = w_s_;
  w.temp = sampled ? sp.temperature : 0.0f;
  w.seed = sp.seed;
  w.qt = walk_tf > 0.0f;
  w.maxdep = 7;
  if (passes_ > 0) {
    encode_chain(e, tokens, Pk, gst, c_i_, c_col_, c_v_, tq, sampled ? 1 : 0, w, passes_);
  } else {
    encode_root(e, tokens, Pk, gst);
    for (int j = 0; j < w.maxdep; ++j) encode_step(e, c_i_, c_col_, c_v_, tq, sampled ? 1 : 0, &w);
  }
}

void KFlash::encode_prep(Encoder& e, const Buffer& xh, const Buffer& xs) {
  const uint32_t H = uint32_t(cfg_.H);
  dfl_enc_qmm8(e, {{&w_h_, A_, uint32_t(cfg_.d), 1.0f}}, xh, xs, H, H);
}

void KFlash::encode_step(Encoder& e, const Buffer& cand_i, const Buffer& cand_col, const Buffer& cand_v, float qtemp,
                         int acc_row, const KFlashWalk* walk, const Buffer* ind) {
  const uint32_t d = uint32_t(cfg_.d), H = uint32_t(cfg_.H), I = uint32_t(cfg_.I), NL = uint32_t(cfg_.NL);
  const QLinear& E = emb_ ? *emb_ : tg_.embed_table();
  if (!out_head_ || !(out_head_->q4() || out_head_->q8()) || out_head_->K != H)
    fatal("KFlash: no 4 / 8-bit table for the candidates' LM-head rows (set_out_head)");
  const QLinear& O = *out_head_;
  // slot: 0 = 8 rows, 1 = attention, 2 = candidates, 3..8 = the GEMMs (W_e, q|k|v, o, gate/up, down, W_out); the full
  // grid of each is recorded for kf_jac
  uint32_t* full = full_args_.as<uint32_t>();
  auto disp = [&](int slot, uint32_t gx, uint32_t gy, uint32_t tx) {
    full[slot * 3 + 0] = gx;
    full[slot * 3 + 1] = gy;
    full[slot * 3 + 2] = 1;
    if (ind) e.dispatch_indirect(*ind, size_t(slot) * 12, tx);
    else e.dispatch(gx, gy, 1, tx);
  };
  auto gemm = [&](int slot, const QLinear& L, const Buffer& y, uint32_t N, const Buffer& xh, const Buffer& xs, uint32_t K) {
    const uint32_t tg = dfl_enc_qmm8(e, {{&L, y, N, 1.0f}}, xh, xs, K, K, 0, 0, ind, size_t(slot) * 12);
    full[slot * 3 + 0] = tg;
    full[slot * 3 + 1] = 1;
    full[slot * 3 + 2] = 1;
  };
  {  // input rows: X8 table of RMSNorm(E[tok]) * emb_norm; residual = W_h h_{j+1} + pos[j]
    kp::KfIn p{H, d, uint32_t(E.bits), uint32_t(E.gs), cfg_.eps};
    // 4 / 8-bit group 64 here; other formats on wfmt.metal's kf_in_<format> (the formats library)
    e.use(kfpipe(E.q4() || E.q8() ? std::string("kf_in") : "kf_in_" + E.fmt()));
    e.buf(0, tree_); e.buf(1, A_); e.buf(2, E.w->buf); e.buf(3, E.s->buf); e.buf(4, E.b->buf);
    e.buf(5, emb_norm_->buf); e.buf(6, pos_->buf); e.buf(7, xin_h_); e.buf(8, xin_s_); e.buf(9, hres_);
    e.val(10, p);
    disp(0, 8, 1, 256);
  }
  gemm(3, w_e_, u_, d, xin_h_, xin_s_, H);
  enc_res_norm(e, layers_[0].ln1, ind);  // h = x; n = ln1(x)
  dbg(e, "x", hres_, size_t(8) * d * 4);
  for (uint32_t l = 0; l < NL; ++l) {
    const Layer& L = layers_[l];
    gemm(4, L.qkv, qkv_, 3 * d, xh_, xs_, d);
    {
      kp::KfAttn p{d, l, NL, 1.0f / std::sqrt(float(cfg_.hd))};
      e.use(kfpipe("kf_attn"));
      e.buf(0, tree_); e.buf(1, qkv_); e.buf(2, kv_); e.buf(3, xh_); e.buf(4, xs_);
      e.val(5, p);
      disp(1, 8, uint32_t(cfg_.nh), 64);
    }
    gemm(5, L.o, u_, d, xh_, xs_, d);
    enc_res_norm(e, L.ln2, ind);
    {
      const uint32_t tg = dfl_enc_qmm8_gu(e, L.gate, L.up, xh_, xs_, hh_, hs_, 0, ind, size_t(6) * 12);
      full[18] = tg;
      full[19] = 1;
      full[20] = 1;
    }
    gemm(7, L.down, u_, d, hh_, hs_, I);
    enc_res_norm(e, l + 1 < NL ? layers_[l + 1].ln1 : norm_, ind);
    dbg(e, "h" + std::to_string(l), hres_, size_t(8) * d * 4);
  }
  dbg(e, "y", n_, size_t(8) * d * 4);
  gemm(8, w_out_, z_, H, xh_, xs_, d);
  dbg(e, "z", z_, size_t(8) * H * 4);
  const int r = acc_rows_ == 2 ? std::clamp(acc_row, 0, 1) : 0;
  const kp::KfSoft sp{H, d, uint32_t(r), acc_rows_ ? 1u : 0u, qtemp, acc_wq_[r], acc_b_[r], walk ? walk->temp : 0.0f,
                      walk ? walk->seed : 0u, walk && walk->qt ? 1u : 0u, walk ? uint32_t(walk->maxdep) : 0u,
                      walk ? 1u : 0u, uint32_t(O.bits)};
  // without a walk its buffers are not touched: the tree is bound in their place
  const Buffer& w_tok = walk ? walk->tokens : tree_;
  const Buffer& w_q = walk ? walk->qprob : tree_;
  const Buffer& w_k = walk ? walk->walk_k : tree_;
  const Buffer& w_s = walk ? walk->walk_s : tree_;
  {
    kp::KfCand p{H, uint32_t(O.bits)};
    e.use(kfpipe("kf_cand"));
    e.buf(0, tree_); e.buf(1, z_); e.buf(2, cand_col); e.buf(3, O.w->buf); e.buf(4, O.s->buf); e.buf(5, O.b->buf);
    e.buf(6, acc_we_); e.buf(7, cd_);
    e.val(8, p);
    disp(2, 8, uint32_t(kCand), 256);
  }
  e.use(kfpipe("kf_soft"));
  e.buf(0, tree_); e.buf(1, n_); e.buf(2, cand_i); e.buf(3, cand_v); e.buf(4, cd_); e.buf(5, lscale_);
  e.buf(6, acc_wy_); e.buf(7, w_tok); e.buf(8, w_q); e.buf(9, w_k); e.buf(10, w_s);
  e.val(11, sp);
  disp(0, 8, 1, 64);
}

void KFlash::encode_chain(Encoder& e, const Buffer& tokens, uint32_t Pk, const Buffer& gst, const Buffer& cand_i,
                          const Buffer& cand_col, const Buffer& cand_v, float qtemp, int acc_row, const KFlashWalk& walk,
                          int passes) {
  const int maxdep = walk.maxdep;
  passes = std::clamp(passes, 1, std::min(maxdep, 15));
  encode_root(e, tokens, Pk, gst, &cand_i, maxdep - 1);
  const size_t blk = size_t(kSlots) * 12;
  for (int i = 0; i < passes; ++i) {
    const Buffer ind = pass_args_.view(size_t(i) * blk, blk);
    encode_step(e, cand_i, cand_col, cand_v, qtemp, acc_row, nullptr, i == 0 ? nullptr : &ind);
    const kp::KfJac p{walk.temp, walk.seed, walk.qt ? 1u : 0u, uint32_t(maxdep), i + 1 == passes ? 1u : 0u,
                      uint32_t(kSlots)};
    e.use(kfpipe("kf_jac"));
    e.buf(0, tree_); e.buf(1, cand_i); e.buf(2, walk.tokens); e.buf(3, walk.qprob); e.buf(4, walk.walk_k);
    e.buf(5, walk.walk_s); e.buf(6, full_args_); e.buf(7, pass_args_.view(size_t(i + 1) * blk, blk));
    e.val(8, p);
    e.dispatch(1, 1, 1, 32);
  }
}

void KFlash::encode_tree(Encoder& e, const Buffer& logits, const Buffer& xh, const Buffer& xs, const Buffer& tokens,
                         uint32_t Pk, const Buffer& gst, const SamplingParams& sp, const Buffer& desc, const Buffer& live,
                         const KfParams& kp) {
  if (!vd_) fatal("KFlash::encode_tree: no drafter bound (bind_drafter)");
  enc_top32(e, logits);
  encode_prep(e, xh, xs);
  encode_tree_passes(e, tokens, Pk, gst, sp, desc, live, kp);
}

void KFlash::encode_tree_passes(Encoder& e, const Buffer& tokens, uint32_t Pk, const Buffer& gst,
                                const SamplingParams& sp, const Buffer& desc, const Buffer& live, const KfParams& kp) {
  const bool sampled = sp.temperature > 0.0f;
  // the children's probability: the decoder's q = softmax(S / T) (T = 0: softmax(S)), which the budget calibrates per
  // depth. Not the acceptance head: on M2's held-out replay (4B) it ranks trees worse than q (8 rows: 4.25 vs 4.43
  // tokens per cycle) and is under-confident (0.75 predicted vs 0.87 accepted at the chain's pick).
  const bool use_acc = false;
  const float tq = sampled ? sp.temperature : 1.0f;
  const int passes = std::clamp(passes_ > 0 ? passes_ : 7, 1, 16);
  const bool gpu = Pk == 0xffffffffu;  // kDflGpu
  uint32_t* full = full_args_.as<uint32_t>();  // kf_dtab: a threadgroup per node; kf_build: one
  full[kSlots * 3 + 0] = kMaxNodes;
  full[kSlots * 3 + 1] = 1;
  full[kSlots * 3 + 2] = 1;
  full[kSlots * 3 + 3] = 1;
  full[kSlots * 3 + 4] = 1;
  full[kSlots * 3 + 5] = 1;
  const size_t blk = size_t(kTreeSlots) * 12;
  for (int i = 0; i < passes; ++i) {
    const Buffer ind = tree_args_.view(size_t(i) * blk, blk);
    const kp::KfTreeNext np{Pk, i == 0 ? 1u : 0u, 7u, uint32_t(kTreeSlots)};
    e.use(kfpipe("kf_tree_next"));
    e.buf(0, tree_); e.buf(1, desc); e.buf(2, tokens);
    e.val(3, np);
    e.buf(4, gst); e.buf(5, full_args_); e.buf(6, ind);
    e.dispatch(1, 1, 1, 32);
    encode_step(e, c_i_, c_col_, c_v_, tq, sampled ? 1 : 0, nullptr, &ind);
    const kp::KfDtab dp{use_acc ? 1u : 0u, 0u};
    e.use(kfpipe("kf_dtab"));
    e.buf(0, tree_); e.buf(1, t_clq_); e.buf(2, t_tok_); e.buf(3, t_next_);
    e.val(4, dp);
    e.dispatch_indirect(ind, size_t(kSlots) * 12, 32);
    const kp::KfBuild bp{uint32_t(kCand), Pk, 0u, 0u};
    e.use(kfpipe(gpu ? "kf_build_gpu" : "kf_build"));
    e.buf(0, t_clq_); e.buf(1, t_tok_); e.buf(2, t_next_);
    e.val(3, kp);
    e.buf(4, desc); e.buf(5, tokens); e.buf(6, live);
    e.val(7, bp);
    if (gpu) e.buf(8, gst);
    e.dispatch_indirect(ind, size_t(kSlots + 1) * 12, 32);
  }
}

std::vector<std::string> KFlash::weight_names() const {
  std::vector<std::string> v;
  for (auto& [n, _] : named_) v.push_back(n);
  return v;
}

std::vector<float> KFlash::dequant(const std::string& name) const {
  auto it = named_.find(name);
  if (it == named_.end()) fatal("KFlash::dequant: no weight %s", name.c_str());
  const QLinear& L = *it->second;
  std::vector<float> o(size_t(L.N) * L.K);
  const uint32_t gpr = L.K / 64;
  for (uint32_t n = 0; n < L.N; ++n) {
    const uint32_t* W = L.w->buf.as<uint32_t>() + size_t(n) * (L.K / 8);
    const uint16_t* S = L.s->buf.as<uint16_t>() + size_t(n) * gpr;
    const uint16_t* B = L.b->buf.as<uint16_t>() + size_t(n) * gpr;
    for (uint32_t k = 0; k < L.K; ++k)
      o[size_t(n) * L.K + k] = bf16_to_f32(S[k / 64]) * float((W[k / 8] >> (4 * (k % 8))) & 15u) + bf16_to_f32(B[k / 64]);
  }
  return o;
}

} // namespace ni

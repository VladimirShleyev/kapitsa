// KFlash candidate decoder: checks against the NumPy reference (tools/kflash_ref.py check) and timings.
//   kflash_check <model_dir> <decoder_dir> [--out dump_dir] [--prompt text | @file.txt] [--time N] [--temp T]
//                 [--seed S] [--ctx N]
// Prefills a prompt with the drafter's capture hook (DFlash2::find's drafter for the model), runs the drafting
// backbone at the next position, then:
//  1. the top-32 of every block row against a CPU sort of the draft logits (value desc, id asc);
//  2. "chain_greedy": the integrated chain walk (DFlash2::encode_draft_sel 1, T = 0): the decoder's tree is a chain
//     whose tokens are the ones written to the sequence, its choices the argmax of each node's scores;
//  3. "chain_sampled": the same at --temp (default 0.7) with --seed: the walk outputs (q, k) the acceptance reads;
//  4. "tree": host-driven fronts of 1, 3, 5, 8, 8, ... rows mixing depths (children by q), up to 64 nodes: the K/V of
//     every branch, per-row depths and parents in one step.
//  --out writes, per case, the backbone rows, the top-32, the target's embedding / LM-head rows of every token
//  involved, the tree, and the decoder's 4-bit weights + the root row's intermediates (tree case) for
//  `kflash_ref.py check <decoder_dir> <dump_dir>`.
//  --emb-bf16: the decoder reads node tokens' embedding rows from a bf16 copy of the target's table (the formats
//  library's kf_in_f16, as with a bf16 target); the dumped rows are those bf16 values.
//  --time N: GPU ms of the drafting pass with DFlash2's walk vs the decoder's chain (interleaved), of the decoder alone
//  (root + 7 steps + walks), and of one decoder step at 1 / 2 / 4 / 8 front rows (medians of N).
//  --ref-vectors <dir> --out <dump>: instead of all that, the trainer's reference vectors (<dir>/reference.safetensors:
//  h, tok, cand, u of recorded drafting passes; the trainer's ref_vectors.py): each anchor's path as one 7-row front,
//  the decoder's y, z, S and both acceptance rows -> <dump>/ref_{y,z,S,a0,a1}.npy (compare with the trainer's forward).
#include "metal/gpu.h"
#include "model/dflash2.h"
#include "model/kflash.h"
#include "model/qwen35.h"
#include "model/qwen4_exp.h"
#include "tokenizer/tokenizer.h"
#include "util/bf16.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

using namespace ni;
using json = nlohmann::json;

static void save_npy(const std::string& path, const std::string& descr, const std::vector<int64_t>& shape,
                     const void* data, size_t bytes) {
  std::string s = "(";
  for (size_t i = 0; i < shape.size(); ++i) s += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
  s += ")";
  std::string h = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + s + ", }";
  while ((10 + h.size() + 1) % 64) h += ' ';
  h += '\n';
  std::ofstream f(path, std::ios::binary);
  f.write("\x93NUMPY\x01\x00", 8);
  uint16_t hl = uint16_t(h.size());
  f.write((const char*)&hl, 2);
  f.write(h.data(), std::streamsize(h.size()));
  f.write((const char*)data, std::streamsize(bytes));
}

// One row of a 4/8-bit group-64 (or bf16) table in fp32, as the decoder's kernels dequantize it (kf_row2)
static void row_f32(const QLinear& L, uint32_t row, float* out) {
  if (L.bits == 16) {
    const uint16_t* r = L.w->buf.as<uint16_t>() + size_t(row) * L.K;
    for (uint32_t k = 0; k < L.K; ++k) out[k] = bf16_to_f32(r[k]);
    return;
  }
  const uint32_t gpr = L.K / uint32_t(L.gs);
  const uint16_t* S = L.s->buf.as<uint16_t>() + size_t(row) * gpr;
  const uint16_t* B = L.b->buf.as<uint16_t>() + size_t(row) * gpr;
  const uint32_t* W = reinterpret_cast<const uint32_t*>(L.w->buf.as<uint8_t>() + size_t(row) * L.row_bytes());
  for (uint32_t k = 0; k < L.K; ++k) {
    const uint32_t code = L.bits == 4 ? (W[k / 8] >> (4 * (k % 8))) & 15u : (W[k / 4] >> (8 * (k % 4))) & 255u;
    out[k] = bf16_to_f32(S[k / L.gs]) * float(code) + bf16_to_f32(B[k / L.gs]);
  }
}

static double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: kflash_check <model_dir> <decoder_dir> [--out dir] [--prompt text|@file] [--time N] "
                    "[--temp T] [--seed S] [--ctx N]\n");
    return 2;
  }
  const std::string mdir = argv[1], kfdir = argv[2];
  std::string out, prompt =
      "Write a C++ function that parses a comma-separated list of integers from a string and returns their sum. "
      "Handle whitespace and negative numbers, and explain the edge cases you considered.";
  int time_n = 0, ctx = 0;
  bool emb_bf16 = false;
  std::string refv;
  float temp = 0.7f;
  uint32_t seed = 1234;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
    else if (a == "--time" && i + 1 < argc) time_n = atoi(argv[++i]);
    else if (a == "--temp" && i + 1 < argc) temp = float(atof(argv[++i]));
    else if (a == "--seed" && i + 1 < argc) seed = uint32_t(strtoul(argv[++i], nullptr, 10));
    else if (a == "--ctx" && i + 1 < argc) ctx = atoi(argv[++i]);
    else if (a == "--ref-vectors" && i + 1 < argc) refv = argv[++i];
    else if (a == "--emb-bf16") emb_bf16 = true;
    else fatal("unknown option %s", a.c_str());
  }
  if (getenv("NI_DFLASH_QT")) fatal("kflash_check: unset NI_DFLASH_QT (the sampled case assumes q = softmax(S / T))");
  if (!prompt.empty() && prompt[0] == '@') {
    std::ifstream f(prompt.substr(1));
    prompt.assign(std::istreambuf_iterator<char>(f), {});
  }
  setvbuf(stdout, nullptr, _IOLBF, 0);
  setenv("NI_KFLASH", kfdir.c_str(), 1);
  Gpu& g = Gpu::get();
  std::unique_ptr<Qwen35> tp = load_qwen(mdir, 2048);
  Qwen35& tgt = *tp;
  if (!refv.empty()) {  // ---- the trainer's reference vectors ----
    if (out.empty()) fatal("--ref-vectors needs --out");
    KFlash dd(tgt, kfdir);
    const int H = dd.config().H, d = dd.config().d;
    WeightStore rv;
    rv.open_dir(refv, nullptr, [](const std::string& f) { return f == "reference.safetensors"; });
    const Tensor& th = rv.get("h");
    const Tensor& tt = rv.get("tok");
    const Tensor& tc = rv.get("cand");
    const Tensor& tu = rv.get("u");
    const int N = int(th.shape[0]);
    if (th.dtype != DType::F16 || th.shape[1] != 7 || th.shape[2] != H) fatal("reference h: f16 [N, 7, %d]", H);
    Buffer xh = g.alloc(size_t(16) * H * 2, "rv_xh"), xs = g.alloc(size_t(H) / 64 * 8 * 4, "rv_xs");
    Buffer ci = g.alloc(7 * 32 * 4, "rv_ci"), cv = g.alloc(7 * 32 * 4, "rv_cv");
    std::vector<float> oy(size_t(N) * 7 * d), oz(size_t(N) * 7 * H), oS(size_t(N) * 7 * 32), oa[2];
    oa[0].resize(size_t(N) * 7 * 32);
    oa[1].resize(size_t(N) * 7 * 32);
    dd.set_debug(true);
    for (int n = 0; n < N; ++n) {
      // X8 table of the block rows 1..7 (h as the drafter's LM head read it: half rows + per-64 sums), row 0 zero
      memset(xh.ptr(), 0, xh.size);
      memset(xs.ptr(), 0, xs.size);
      const _Float16* hs = th.buf.as<_Float16>() + size_t(n) * 7 * H;
      for (int r = 1; r < 8; ++r)
        for (int k = 0; k < H; ++k) {
          const _Float16 v = hs[size_t(r - 1) * H + k];
          xh.as<_Float16>()[size_t(r) * H + k] = v;
          xs.as<float>()[(k / 64) * 8 + r] += float(v);
        }
      memcpy(ci.ptr(), tc.buf.as<int32_t>() + size_t(n) * 224, 224 * 4);
      memcpy(cv.ptr(), tu.buf.as<float>() + size_t(n) * 224, 224 * 4);
      for (int row = 0; row < 2; ++row) {
        KFlashTree& T = *dd.tree().as<KFlashTree>();
        T.n_nodes = 7;
        T.n_front = 7;
        for (int j = 0; j < 7; ++j) {
          T.front[j] = uint32_t(j);
          T.parent[j] = j - 1;
          T.token[j] = tt.buf.as<int32_t>()[size_t(n) * 8 + j];
          T.depth[j] = j;
          T.expanded[j] = 0;
        }
        auto e = g.begin();
        dd.encode_prep(*e, xh, xs);
        dd.encode_step(*e, ci, ci, cv, 1.0f, row);  // the LM head itself is the out table: columns = token ids
        e->commit_and_wait();
        for (int j = 0; j < 7; ++j)
          for (int k = 0; k < 32; ++k) {
            oa[row][(size_t(n) * 7 + j) * 32 + k] = T.ch_a[j * 32 + k];
            if (row == 0) oS[(size_t(n) * 7 + j) * 32 + k] = T.ch_s[j * 32 + k];
          }
        if (row == 0) {
          memcpy(oy.data() + size_t(n) * 7 * d, dd.debug_bufs().at("y").ptr(), size_t(7) * d * 4);
          memcpy(oz.data() + size_t(n) * 7 * H, dd.debug_bufs().at("z").ptr(), size_t(7) * H * 4);
        }
      }
    }
    mkdir(out.c_str(), 0755);
    save_npy(out + "/ref_y.npy", "<f4", {N, 7, d}, oy.data(), oy.size() * 4);
    save_npy(out + "/ref_z.npy", "<f4", {N, 7, H}, oz.data(), oz.size() * 4);
    save_npy(out + "/ref_S.npy", "<f4", {N, 7, 32}, oS.data(), oS.size() * 4);
    save_npy(out + "/ref_a0.npy", "<f4", {N, 7, 32}, oa[0].data(), oa[0].size() * 4);
    save_npy(out + "/ref_a1.npy", "<f4", {N, 7, 32}, oa[1].data(), oa[1].size() * 4);
    printf("[ref-vectors] %d anchors x 7 rows (each path one 7-row front) -> %s\n", N, out.c_str());
    return 0;
  }
  const std::string ddir = DFlash2::find(mdir, tgt.config());
  if (ddir.empty()) fatal("no DFlash2 / DFlash drafter for %s", mdir.c_str());
  DFlash2 dr(tgt, ddir);
  KFlash* kf = dr.kflash();
  if (!kf) fatal("the decoder did not load");
  const int H = kf->config().H, d = kf->config().d;
  printf("== kflash_check %s (drafter %s, decoder %s, H %d, d %d)\n", mdir.c_str(), ddir.c_str(), kfdir.c_str(), H, d);

  Tokenizer tok;
  tok.load(mdir + "/tokenizer.json");
  std::vector<int> ids = tok.encode("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n");
  if (ctx > int(ids.size())) {  // a longer context: repeat the prompt
    std::vector<int> r;
    while (int(r.size()) < ctx) r.insert(r.end(), ids.begin(), ids.end());
    r.resize(size_t(ctx));
    ids = r;
  }
  const int n = int(ids.size());
  auto seq = tgt.new_seq(std::max(8192, n + 64));
  Seq& s = *seq;
  for (int i = 0; i < n; ++i) s.tokens()[i] = ids[i];
  auto dseq = dr.new_seq();
  DFlashSeq& ds = *dseq;
  SamplingParams greedy;
  greedy.temperature = 0.0f;
  dr.begin_prefill(n);
  for (int p = 0; p < n; p += tgt.prefill_chunk()) {
    const int T = std::min(tgt.prefill_chunk(), n - p);
    auto e = g.begin();
    tgt.encode_prefill(*e, s, p, T, greedy, 0, p + T == n);
    e->commit_and_wait();
    if (!tgt.ane_finish_chunk()) fatal("the NI_ANE prefill split failed; rerun with NI_ANE=0");
  }
  dr.end_prefill();
  {
    auto e = g.begin();
    dr.encode_prefill_commit(*e, ds);
    e->commit_and_wait();
  }
  const int P = n;
  printf("prompt %d tokens, anchor %d at P = %d\n", n, s.tokens()[P], P);
  int fails = 0;
  if (!out.empty()) mkdir(out.c_str(), 0755);
  json meta;
  meta["model"] = mdir;
  meta["drafter"] = ddir;
  meta["decoder"] = kfdir;
  meta["P"] = P;
  meta["cases"] = json::array();
  const QLinear& E0 = tgt.embed_table();
  const QLinear& O = tgt.lm_head();
  // --emb-bf16: a bf16 copy of the embedding (each row dequantized in fp32 as the kernels do, rounded to bf16)
  QLinear Eb;
  Tensor Ebt;
  if (emb_bf16) {
    const uint32_t V = E0.N;
    Buffer b = g.alloc(size_t(V) * H * 2, "emb_bf16", false);
    std::vector<float> row(H);
    for (uint32_t t = 0; t < V; ++t) {
      row_f32(E0, t, row.data());
      for (int k = 0; k < H; ++k) b.as<uint16_t>()[size_t(t) * H + k] = f32_to_bf16(row[k]);
    }
    Ebt = Tensor{b, DType::BF16, {int64_t(V), int64_t(H)}};
    Eb.w = &Ebt;
    Eb.s = &Ebt;  // unread for bf16
    Eb.b = &Ebt;
    Eb.N = V;
    Eb.K = uint32_t(H);
    Eb.bits = 16;
    Eb.gs = 64;
    kf->set_embed_table(&Eb);
    printf("decoder embedding: a bf16 copy of the target's %s table (kf_in_%s)\n", format_name(E0.bits, E0.gs).c_str(),
           Eb.fmt().c_str());
  }
  const QLinear& E = emb_bf16 ? Eb : E0;

  // dumps one case: backbone rows, top-32, the tree, the target rows of every token involved
  auto dump_case = [&](const std::string& name, const std::string& kind, float t, float qtemp, uint32_t sd, int acc_row) {
    const KFlashTree& T = kf->tree_host();
    json c;
    c["name"] = name;
    c["kind"] = kind;
    c["temp"] = t;
    c["qtemp"] = qtemp;
    c["seed"] = sd;
    c["acc_row"] = acc_row;
    c["P"] = P;
    c["n_nodes"] = T.n_nodes;
    meta["cases"].push_back(c);
    if (out.empty()) return;
    const std::string cd = out + "/" + name;
    mkdir(cd.c_str(), 0755);
    save_npy(cd + "/h.npy", "<f4", {8, H}, dr.final_hidden_buf().ptr(), size_t(8) * H * 4);
    save_npy(cd + "/cand_i.npy", "<i4", {7, 32}, dr.cand32_ids_buf().ptr(), 7 * 32 * 4);
    save_npy(cd + "/cand_v.npy", "<f4", {7, 32}, dr.cand32_vals_buf().ptr(), 7 * 32 * 4);
    std::set<int> toks;
    for (int i = 0; i < 7 * 32; ++i) toks.insert(dr.cand32_ids_buf().as<int>()[i]);
    for (uint32_t i = 0; i < T.n_nodes; ++i) toks.insert(T.token[i]);
    std::vector<int> rid(toks.begin(), toks.end());
    std::vector<float> er(rid.size() * size_t(H)), orow(rid.size() * size_t(H));
    for (size_t i = 0; i < rid.size(); ++i) {
      row_f32(E, uint32_t(rid[i]), er.data() + i * H);
      row_f32(O, uint32_t(rid[i]), orow.data() + i * H);
    }
    const int64_t M = int64_t(rid.size());
    save_npy(cd + "/row_ids.npy", "<i4", {M}, rid.data(), rid.size() * 4);
    save_npy(cd + "/erows.npy", "<f4", {M, H}, er.data(), er.size() * 4);
    save_npy(cd + "/orows.npy", "<f4", {M, H}, orow.data(), orow.size() * 4);
    const int64_t NN = KFlashTree::kMaxNodes, C = KFlashTree::kCand;
    save_npy(cd + "/tree_parent.npy", "<i4", {NN}, T.parent, NN * 4);
    save_npy(cd + "/tree_token.npy", "<i4", {NN}, T.token, NN * 4);
    save_npy(cd + "/tree_depth.npy", "<i4", {NN}, T.depth, NN * 4);
    save_npy(cd + "/tree_cand_k.npy", "<i4", {NN}, T.cand_k, NN * 4);
    save_npy(cd + "/tree_logp.npy", "<f4", {NN}, T.logp, NN * 4);
    save_npy(cd + "/tree_acc.npy", "<f4", {NN}, T.acc, NN * 4);
    save_npy(cd + "/tree_expanded.npy", "<i4", {NN}, T.expanded, NN * 4);
    save_npy(cd + "/tree_ch_tok.npy", "<i4", {NN, C}, T.ch_tok, NN * C * 4);
    save_npy(cd + "/tree_ch_s.npy", "<f4", {NN, C}, T.ch_s, NN * C * 4);
    save_npy(cd + "/tree_ch_q.npy", "<f4", {NN, C}, T.ch_q, NN * C * 4);
    save_npy(cd + "/tree_ch_a.npy", "<f4", {NN, C}, T.ch_a, NN * C * 4);
    save_npy(cd + "/chain.npy", "<i4", {7}, s.tokens() + P + 1, 7 * 4);  // what a chain case drafted
  };

  // ---- 1. top-32 against a CPU sort of the draft logits ----
  {
    auto e = g.begin();
    dr.encode_draft_sel(*e, s, ds, P, greedy, 2);
    e->commit_and_wait();
    const uint32_t Vd = dr.head_rows();
    const uint32_t* hid = dr.head_ids();
    int bad = 0;
    for (int r = 1; r < 8; ++r) {
      std::vector<std::pair<float, uint32_t>> v(Vd);
      for (uint32_t k = 0; k < Vd; ++k) v[k] = {dr.logits()[size_t(r) * Vd + k], hid ? hid[k] : k};
      std::partial_sort(v.begin(), v.begin() + 32, v.end(), [](auto& a, auto& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
      });
      for (int k = 0; k < 32; ++k) {
        const int col = dr.cand32_cols_buf().as<int>()[(r - 1) * 32 + k];
        if (dr.cand32_ids_buf().as<int>()[(r - 1) * 32 + k] != int(v[k].second) ||
            dr.cand32_vals_buf().as<float>()[(r - 1) * 32 + k] != v[k].first || col < 0 || uint32_t(col) >= Vd ||
            (hid ? hid[col] : uint32_t(col)) != v[k].second)
          ++bad;
      }
    }
    printf("[top32] 7 rows x 32 of %u draft logits vs CPU sort: %d mismatches: %s\n", Vd, bad, bad ? "FAIL" : "PASS");
    fails += bad != 0;
  }

  // ---- 2. greedy chain (the integrated walk) ----
  // the chain's nodes in the tree: from the root, the child whose token is the next drafted one (with speculative passes
  // the tree also holds the guessed nodes that were not taken)
  auto chain_nodes = [&](const KFlashTree& T) {
    std::vector<int> ch = {0};
    for (int j = 0; j < 7; ++j) {
      int nx = -1;
      for (uint32_t i = 1; i < T.n_nodes; ++i)
        if (T.parent[i] == ch.back() && T.token[i] == s.tokens()[P + 1 + j]) nx = int(i);
      if (nx < 0) break;
      ch.push_back(nx);
    }
    return ch;
  };
  auto chain_check = [&](const char* name, const SamplingParams& sp) {
    auto e = g.begin();
    dr.encode_draft_sel(*e, s, ds, P, sp, 1);
    e->commit_and_wait();
    const KFlashTree& T = kf->tree_host();
    const std::vector<int> ch = chain_nodes(T);
    int bad = ch.size() == 8 ? 0 : 1;
    for (int j = 0; j < 7 && j + 1 < int(ch.size()); ++j) {
      const int x = ch[j], c = ch[j + 1];
      // the row's scores come from the chain node itself when it was expanded (always, with 7 passes)
      if (!T.expanded[x] || T.depth[c] != j + 1 || T.cand_k[c] != dr.walk_k()[j] || T.ch_tok[x * 32 + T.cand_k[c]] != T.token[c] ||
          dr.cand32_ids_buf().as<int>()[j * 32 + T.cand_k[c]] != T.token[c])
        ++bad;
      for (int k = 0; k < 32; ++k) {
        if (dr.walk_scores()[j * 32 + k] != T.ch_s[x * 32 + k]) ++bad;
        const float qw = dr.qprob()[j * 32 + k];
        if (sp.temperature > 0 ? qw != T.ch_q[x * 32 + k] : qw != 0.0f) ++bad;
      }
    }
    printf("[%s] chain:", name);
    for (int j = 0; j < 7; ++j) printf(" %d", s.tokens()[P + 1 + j]);
    printf(" (k:");
    for (int j = 0; j < 7; ++j) printf(" %d", dr.walk_k()[j]);
    printf("; %u speculative passes); tree / sequence / walk outputs consistent: %s\n", T.walk_passes, bad ? "FAIL" : "PASS");
    fails += bad != 0;
  };
  // the speculative passes (KFlash::encode_chain) against the sequential walk: with 7 passes the same chain and walk
  // outputs, bitwise; the passes it took
  auto spec_check = [&](const char* name, const SamplingParams& sp) {
    const float tq = sp.temperature > 0 ? sp.temperature : 1.0f;
    Buffer q1 = g.alloc(7 * 32 * 4, "t"), k1 = g.alloc(64, "t"), s1 = g.alloc(7 * 32 * 4, "t");
    Buffer q2 = g.alloc(7 * 32 * 4, "t"), k2 = g.alloc(64, "t"), s2 = g.alloc(7 * 32 * 4, "t");
    std::vector<int> chain_seq(7), chain_spec(7);
    int bad = 0, passes_used = 0;
    for (int mode = 0; mode < 2; ++mode) {
      auto e = g.begin();
      dr.encode_draft_sel(*e, s, ds, P, sp, 2);
      KFlashWalk w;
      w.tokens = s.tok;
      w.qprob = mode ? q2 : q1;
      w.walk_k = mode ? k2 : k1;
      w.walk_s = mode ? s2 : s1;
      w.temp = sp.temperature;
      w.seed = sp.seed;
      w.qt = true;
      w.maxdep = 7;
      if (mode == 0) {
        kf->encode_root(*e, s.tok, uint32_t(P), kf->tree());
        for (int j = 0; j < 7; ++j) kf->encode_step(*e, dr.cand32_ids_buf(), dr.cand32_cols_buf(), dr.cand32_vals_buf(), tq, sp.temperature > 0, &w);
      } else {
        kf->encode_chain(*e, s.tok, uint32_t(P), kf->tree(), dr.cand32_ids_buf(), dr.cand32_cols_buf(), dr.cand32_vals_buf(), tq,
                         sp.temperature > 0, w, 7);
      }
      e->commit_and_wait();
      for (int j = 0; j < 7; ++j) (mode ? chain_spec : chain_seq)[j] = s.tokens()[P + 1 + j];
      if (mode) passes_used = int(kf->tree_host().walk_passes);
    }
    if (chain_seq != chain_spec) ++bad;
    if (memcmp(q1.ptr(), q2.ptr(), q1.size) || memcmp(k1.ptr(), k2.ptr(), 28) || memcmp(s1.ptr(), s2.ptr(), s1.size)) ++bad;
    printf("[%s] speculative passes vs the sequential walk: chain", name);
    for (int j = 0; j < 7; ++j) printf(" %d", chain_spec[j]);
    printf(", %d pass%s; chain and walk outputs bitwise equal: %s\n", passes_used, passes_used == 1 ? "" : "es",
           bad ? "FAIL" : "PASS");
    fails += bad != 0;
  };
  spec_check("spec_greedy", greedy);
  chain_check("chain_greedy", greedy);
  dump_case("chain_greedy", "chain", 0.0f, 1.0f, 0, 0);
  SamplingParams samp;
  samp.temperature = temp;
  samp.seed = seed;
  spec_check("spec_sampled", samp);
  chain_check("chain_sampled", samp);
  dump_case("chain_sampled", "chain", temp, temp, seed, 1);

  // ---- 3. host-driven tree: fronts mixing depths ----
  {
    auto e = g.begin();
    dr.encode_draft_sel(*e, s, ds, P, greedy, 2);
    kf->encode_root(*e, s.tok, uint32_t(P), kf->tree());  // explicit P: the drafter-state buffer is not read
    kf->set_debug(true);
    kf->encode_step(*e, dr.cand32_ids_buf(), dr.cand32_cols_buf(), dr.cand32_vals_buf(), 1.0f, 0);
    e->commit_and_wait();
    kf->set_debug(false);
    KFlashTree& T = *kf->tree().as<KFlashTree>();
    // children of `node` ranked by q
    auto ranked = [&](int node) {
      std::vector<int> k(32);
      for (int i = 0; i < 32; ++i) k[i] = i;
      std::stable_sort(k.begin(), k.end(), [&](int a, int b) { return T.ch_q[node * 32 + a] > T.ch_q[node * 32 + b]; });
      return k;
    };
    auto add = [&](int parent, int k) {
      const int id = int(T.n_nodes++);
      T.parent[id] = parent;
      T.token[id] = T.ch_tok[parent * 32 + k];
      T.depth[id] = T.depth[parent] + 1;
      T.cand_k[id] = k;
      T.logp[id] = T.logp[parent] + std::log(std::max(T.ch_q[parent * 32 + k], 1e-30f));
      T.acc[id] = T.ch_a[parent * 32 + k];
      T.expanded[id] = 0;
      T.row[id] = id;
      return id;
    };
    std::vector<int> front;
    {
      const auto r0 = ranked(0);
      front = {add(0, r0[0]), add(0, r0[1]), add(0, r0[2])};  // 3 rows, depth 1
    }
    std::vector<int> root_rank = ranked(0);
    int steps = 1, rows = 1;
    for (int it = 0; !front.empty(); ++it) {
      T.n_front = uint32_t(front.size());
      for (size_t i = 0; i < front.size(); ++i) T.front[i] = uint32_t(front[i]);
      auto e2 = g.begin();
      kf->encode_step(*e2, dr.cand32_ids_buf(), dr.cand32_cols_buf(), dr.cand32_vals_buf(), 1.0f, 0);
      e2->commit_and_wait();
      ++steps;
      rows += int(front.size());
      std::vector<int> next;
      for (size_t i = 0; i < front.size(); ++i) {
        const int f = front[i];
        if (T.depth[f] >= 6) continue;  // its children (depth 7) cannot be expanded
        const auto r = ranked(f);
        const int want = it == 1 && i < 3 ? 2 : 1;  // the third step widens to 8 rows
        for (int w = 0; w < want && int(T.n_nodes) < KFlashTree::kMaxNodes && next.size() < 8; ++w) next.push_back(add(f, r[w]));
      }
      if (it == 0) {  // root's 4th and 5th children join the depth-2 nodes: a front of depths 1 and 2
        next.push_back(add(0, root_rank[3]));
        next.push_back(add(0, root_rank[4]));
      }
      front = next;
    }
    {  // a speculated subtree in one step: a new child a of the root and, before any of them is scored, a's children
      // b, b2 (the backbone's top-2 of row 2) and b's child c (top-1 of row 3): tree-causal attention inside the step
      const int a = add(0, root_rank[5]);
      auto spec = [&](int parent, int k) {  // a child picked from the backbone's candidates of its row
        const int id = int(T.n_nodes++);
        T.parent[id] = parent;
        T.depth[id] = T.depth[parent] + 1;
        T.token[id] = dr.cand32_ids_buf().as<int>()[(T.depth[id] - 1) * 32 + k];
        T.cand_k[id] = k;
        T.logp[id] = 0.0f;
        T.acc[id] = 0.0f;
        T.expanded[id] = 0;
        T.row[id] = id;
        return id;
      };
      const int b = spec(a, 0), c = spec(b, 0), b2 = spec(a, 1);
      T.n_front = 4;
      T.front[0] = uint32_t(c);  // descendants before their ancestors in the front: order must not matter
      T.front[1] = uint32_t(a);
      T.front[2] = uint32_t(b2);
      T.front[3] = uint32_t(b);
      auto e2 = g.begin();
      kf->encode_step(*e2, dr.cand32_ids_buf(), dr.cand32_cols_buf(), dr.cand32_vals_buf(), 1.0f, 0);
      e2->commit_and_wait();
      ++steps;
      rows += 4;
    }
    printf("[tree] %u nodes, %d decoder steps over %d front rows (depths up to %d; the last step a speculated "
           "subtree of 4)\n", T.n_nodes, steps, rows, *std::max_element(T.depth, T.depth + T.n_nodes));
    dump_case("tree", "tree", 0.0f, 1.0f, 0, 0);
    if (!out.empty()) {
      for (const char* k : {"x", "h0", "h1", "y", "z"}) {
        auto it = kf->debug_bufs().find(k);
        if (it == kf->debug_bufs().end()) continue;
        const int w = std::string(k) == "z" ? H : d;
        save_npy(out + "/tree/dbg_" + k + ".npy", "<f4", {8, w}, it->second.ptr(), size_t(8) * w * 4);
      }
    }
  }
  // ---- 4. the backbone's top-1 chain as one 7-row step (a speculative chain walk's first pass) ----
  {
    auto e = g.begin();
    dr.encode_draft_sel(*e, s, ds, P, greedy, 2);
    kf->encode_root(*e, s.tok, uint32_t(P), kf->tree());
    e->commit_and_wait();
    KFlashTree& T = *kf->tree().as<KFlashTree>();
    for (int j = 1; j < 7; ++j) {
      T.parent[j] = j - 1;
      T.depth[j] = j;
      T.token[j] = dr.cand32_ids_buf().as<int>()[(j - 1) * 32];
      T.cand_k[j] = 0;
      T.expanded[j] = 0;
    }
    T.n_nodes = 7;
    T.n_front = 7;
    for (int j = 0; j < 7; ++j) T.front[j] = uint32_t(6 - j);  // reversed: order must not matter
    auto e2 = g.begin();
    kf->encode_step(*e2, dr.cand32_ids_buf(), dr.cand32_cols_buf(), dr.cand32_vals_buf(), 1.0f, 0);
    e2->commit_and_wait();
    int n_exp = 0;
    for (int j = 0; j < 7; ++j) n_exp += T.expanded[j];
    printf("[spec_chain] the backbone's top-1 chain, 7 rows in one step: %d nodes expanded\n", n_exp);
    if (n_exp != 7) ++fails;
    dump_case("spec_chain", "tree", 0.0f, 1.0f, 0, 0);
  }

  if (!out.empty()) {  // the decoder's 4-bit weights as the engine quantized them
    const std::string wd = out + "/weights";
    mkdir(wd.c_str(), 0755);
    for (const std::string& nm : kf->weight_names()) {
      const QLinear* L = kf->weight(nm);
      save_npy(wd + "/" + nm + ".w.npy", "<u4", {int64_t(L->N), int64_t(L->K / 8)}, L->w->buf.ptr(), size_t(L->N) * L->K / 2);
      save_npy(wd + "/" + nm + ".s.npy", "<u2", {int64_t(L->N), int64_t(L->K / 64)}, L->s->buf.ptr(), size_t(L->N) * L->K / 32);
      save_npy(wd + "/" + nm + ".b.npy", "<u2", {int64_t(L->N), int64_t(L->K / 64)}, L->b->buf.ptr(), size_t(L->N) * L->K / 32);
    }
    std::ofstream(out + "/meta.json") << meta.dump(2);
    printf("dumps -> %s (kflash_ref.py check %s %s)\n", out.c_str(), kfdir.c_str(), out.c_str());
  }

  // ---- timings ----
  if (time_n > 0) {
    Buffer tq = g.alloc(7 * 32 * 4, "t_q"), tk = g.alloc(64, "t_k"), tws = g.alloc(7 * 32 * 4, "t_ws");
    std::vector<double> t_walk2, t_chain, t_dec;
    // (a) the drafting pass with DFlash2's walk (sel 0) and with the decoder's chain (sel 1), interleaved
    for (int i = 0; i <= time_n; ++i)
      for (int v = 0; v < 2; ++v) {
        auto e = g.begin();
        dr.encode_draft_sel(*e, s, ds, P, greedy, v);
        e->commit_and_wait();
        if (i) (v ? t_chain : t_walk2).push_back(e->gpu_ms());
      }
    // (b) the decoder alone: root + 7 x (step + walk), one command buffer after the backbone's
    {
      auto e = g.begin();
      dr.encode_draft_sel(*e, s, ds, P, greedy, 2);
      e->commit_and_wait();
    }
    KFlashWalk w;
    w.tokens = s.tok;
    w.qprob = tq;
    w.walk_k = tk;
    w.walk_s = tws;
    auto chain = [&](Encoder& e) {
      kf->encode_root(e, s.tok, uint32_t(P), kf->tree());
      for (int j = 0; j < 7; ++j) kf->encode_step(e, dr.cand32_ids_buf(), dr.cand32_cols_buf(), dr.cand32_vals_buf(), 1.0f, 0, &w);
    };
    for (int i = 0; i <= time_n; ++i) {
      auto e = g.begin();
      chain(*e);
      e->commit_and_wait();
      if (i) t_dec.push_back(e->gpu_ms());
    }
    printf("[time] drafting pass GPU ms (median of %d, interleaved): DFlash2 walk %.3f, KFlash chain %.3f (%+.3f); "
           "decoder alone (root + 7 steps + walks) %.3f\n", time_n, median(t_walk2), median(t_chain),
           median(t_chain) - median(t_walk2), median(t_dec));
    {  // the chain as speculative passes: at most 1, 2, 3 and 7 (= the sequential walk's chain) passes
      printf("[time] decoder chain as speculative passes, GPU ms (median of %d; passes run):", time_n);
      for (int cap : {7, 3, 2, 1}) {
        std::vector<double> t;
        uint32_t used = 0;
        for (int i = 0; i <= time_n; ++i) {
          auto e = g.begin();
          kf->encode_chain(*e, s.tok, uint32_t(P), kf->tree(), dr.cand32_ids_buf(), dr.cand32_cols_buf(), dr.cand32_vals_buf(), 1.0f, 0, w, cap);
          e->commit_and_wait();
          if (i) t.push_back(e->gpu_ms());
          used = kf->tree_host().walk_passes;
        }
        printf("  cap %d: %.3f (%u)", cap, median(t), used);
      }
      printf("\n");
    }
    {  // the fixed cost of a dependent dispatch: 40 one-threadgroup kf_root dispatches in a serial encoder
      std::vector<double> t1, t40;
      for (int i = 0; i <= time_n; ++i)
        for (int n_disp : {1, 40}) {
          auto e = g.begin();
          for (int k = 0; k < n_disp; ++k) kf->encode_root(*e, s.tok, uint32_t(P), kf->tree());
          e->commit_and_wait();
          if (i) (n_disp == 1 ? t1 : t40).push_back(e->gpu_ms());
        }
      printf("[time] a dependent trivial dispatch costs %.2f us (command buffer of 1: %.3f ms, of 40: %.3f ms)\n",
             (median(t40) - median(t1)) / 39 * 1000, median(t1), median(t40));
    }
    // (c) one decoder step at F front rows (F of the root's children, depth 1): command buffers of 1 and 5 steps
    {
      auto e = g.begin();
      kf->encode_root(*e, s.tok, uint32_t(P), kf->tree());
      kf->encode_step(*e, dr.cand32_ids_buf(), dr.cand32_cols_buf(), dr.cand32_vals_buf(), 1.0f, 0);
      e->commit_and_wait();
      KFlashTree& T = *kf->tree().as<KFlashTree>();
      for (int i = 1; i <= 8; ++i) {
        T.parent[i] = 0;
        T.token[i] = T.ch_tok[i - 1];
        T.depth[i] = 1;
        T.cand_k[i] = i - 1;
        T.expanded[i] = 0;
      }
      T.n_nodes = 9;
      if (Encoder::profiling()) {  // NI_PROFILE: each dispatch of 10 one-row steps in its own command buffer
        Encoder::profile_report(1);  // drop what came before
        T.n_front = 1;
        T.front[0] = 1;
        for (int i = 0; i < 10; ++i) {
          auto e2 = g.begin();
          kf->encode_step(*e2, dr.cand32_ids_buf(), dr.cand32_cols_buf(), dr.cand32_vals_buf(), 1.0f, 0);
          e2->commit_and_wait();
        }
        Encoder::profile_report(10);
        return 0;
      }
      printf("[time] one decoder step, GPU ms (median of %d):", time_n);
      for (int F : {1, 2, 4, 8}) {
        T.n_front = uint32_t(F);
        for (int i = 0; i < F; ++i) T.front[i] = uint32_t(1 + i);
        std::vector<double> t1, t5;
        for (int i = 0; i <= time_n; ++i)
          for (int reps : {1, 5}) {
            auto e2 = g.begin();
            for (int r = 0; r < reps; ++r)
              kf->encode_step(*e2, dr.cand32_ids_buf(), dr.cand32_cols_buf(), dr.cand32_vals_buf(), 1.0f, 0);
            e2->commit_and_wait();
            if (i) (reps == 1 ? t1 : t5).push_back(e2->gpu_ms());
          }
        printf("  %d row%s %.3f (in a 5-step buffer %.3f each)", F, F > 1 ? "s" : "", median(t1), median(t5) / 5);
      }
      printf("\n");
    }
  }
  printf(fails ? "FAILED: %d\n" : "ALL PASSED\n", fails);
  return fails ? 1 : 0;
}

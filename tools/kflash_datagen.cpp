// KFlash training data (internal/KFLASH.md §3, milestone M2): the target's own answers, then per
// answer the target's distributions and the block drafter's per-position outputs, streamed to the candidate
// decoder's trainer (tools/kflash/). The caller holds ~/.cache/nace-gpu.lock.
//
//   kflash_datagen gen <model_dir> --requests R.jsonl [--batch 8] [--max-pos 65536] [--time-limit S] > answers.jsonl
//     One request per line: {"id", "messages" [, "tools", "template_kwargs"] | "prompt_ids", "temperature",
//     "seed", "max_tokens", "meta"}; messages are rendered by the model's chat template as the engine renders them.
//     Up to --batch requests decode at once, one row each in a batched 8-row verify pass (Qwen35::encode_verify_multi,
//     the engine's batched plain decode: Kapitsa's verify numerics; T > 0 samples with top-p 0.95, top-k 20 and the
//     request's seed). Writes {"id", "prompt", "answer", "finish", "temperature", "seed", "meta"} per request as it
//     finishes (answer: the generated ids, the end-of-turn token included). --time-limit: no new request after S s;
//     requests still in flight --grace s later (default 240) are dropped unwritten.
//
//   kflash_datagen data <model_dir> --seqs answers.jsonl [--dflash auto|<dir>] [--anchors N|all] [--seed S]
//                   [--shard i/n] [--max-answer N] > stream.bin
//     Per sequence (prompt + answer): the prompt prefill (drafter context as the engine commits it; one prefill per
//     distinct prompt, its state restored for each answer), the answer rows' prefill with the drafter captures and
//     every row's logits (the target's top-64 at T = 1), then drafting passes at N anchors drawn without replacement
//     from the answer positions (all of them with "all"), each committed exactly as the decode loop would have:
//     the ring holds positions [P - 2047, P), the block is [tokens[P], mask x 7].
//
// Stream (little-endian): "KFSTRM01", u32 n, n bytes of JSON (the constants below and the run's settings), then
// records of u32 tag, u32 payload bytes, payload:
//   "SEQ1": u32 seq (line index in --seqs), u32 flags (bit 0: sampled answer), i32 n (tokens), i32 n_prompt,
//           u32 id_len, id bytes (padded to 4), i32 tokens[n], then rows r = n_prompt .. n - 2 (nrows = n - 1 - n_prompt;
//           row r is the target's distribution of token r + 1): i32 top_ids[nrows][64], f32 top_p[nrows][64]
//           (softmax over the whole vocabulary, descending)
//   "ANC1": u32 seq, i32 P (anchor position), i32 npos (positions with a token: min(7, n - 1 - P)), then for block
//           positions p = 1..7 (token P + p; target row P + p - 1): f16 h[7][H] (the drafter's final normed hidden =
//           the draft head's input), i32 cand[7][32] (top-32 ids of the draft head, descending), f32 u[7][32] (their raw
//           draft logits), f32 u_tgt[7][64] (draft logits of the target's top-64 ids of row P + p - 1), f32 u_next[7]
//           (draft logit of tokens[P + p]); NaN where a token is outside the draft head or p > npos
//   "END1": u32 seq, i32 anchors
// The draft logits are the drafter's as the decode loop computes them (its reduced head by default: NI_DFLASH_HEAD).
#include "engine/engine.h"
#include "metal/gpu.h"
#include "model/dflash2.h"
#include "model/qwen35.h"
#include "model/qwen4_exp.h"
#include "tokenizer/tokenizer.h"
#include "util/log.h"

#include <minja/chat-template.hpp>
#include <nlohmann/json.hpp>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ni;

namespace {

constexpr int kTopN = 64;  // target distribution entries kept per row
constexpr int kCand = 32;  // draft candidates per block position
constexpr int kPos = 7;    // block positions with a proposal (block of 8)

FILE* g_out = nullptr;  // the data stream / JSONL (the process's stdout; stray library prints go to stderr)

void out_bytes(const void* p, size_t n) {
  if (fwrite(p, 1, n, g_out) != n) fatal("kflash_datagen: write failed (reader gone?)");
}
template <class T> void out_val(const T& v) { out_bytes(&v, sizeof v); }
void out_tag(const char* tag, uint32_t bytes) {
  out_bytes(tag, 4);
  out_val(bytes);
}

std::string arg_after(int argc, char** argv, const char* key, const std::string& def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], key)) return argv[i + 1];
  return def;
}

// The prompt's system turn: the tokens before its first "<|im_start|>user" (0 if none). Prompts that open with the
// same long system turn (the agent prompts' system prompt and tools, ~5.5k tokens) share its prefill.
struct SystemTurn {
  int im_start = -1, user = -1;
  explicit SystemTurn(const Tokenizer& tok) {
    im_start = tok.find("<|im_start|>");
    const std::vector<int> u = tok.encode("user");
    if (u.size() == 1) user = u[0];
  }
  int length(const std::vector<int>& p) const {
    for (size_t i = 0; i + 1 < p.size(); ++i)
      if (p[i] == im_start && p[i + 1] == user) return int(i);
    return 0;
  }
};
constexpr int kMinShared = 1024;  // shorter system turns are prefilled with the prompt

// GPU copies of attention-cache positions [0, L) of `src` into `dst` (layout [kv heads][cap][head dim], each its own cap)
void copy_prefix_kv(Qwen35& m, Encoder& e, Seq& src, Seq& dst, int L) {
  const Qwen35Config& c = m.config();
  const size_t row = size_t(c.head_dim) * 2;
  for (int i = 0; i < c.layers; ++i) {
    if (c.is_linear[i]) continue;
    for (int kv = 0; kv < 2; ++kv) {
      const Buffer& a = kv ? src.vc[i] : src.kc[i];
      const Buffer& b = kv ? dst.vc[i] : dst.kc[i];
      for (int h = 0; h < c.hkv; ++h)
        m.encode_copy(e, a.view(size_t(h) * src.cap * row, size_t(L) * row), b.view(size_t(h) * dst.cap * row, size_t(L) * row));
    }
  }
}

// ---- gen: the target's answers, batched plain decoding on the 8-row verify ----

struct Request {
  std::string id;
  std::vector<int> prompt;
  SamplingParams sp;
  int max_tokens = 1024;
  ojson meta;
};

struct Live {
  Request r;
  Seq* s = nullptr;  // a pool sequence (the engine never frees a Seq: --batch of them, KV dropped per request)
  int P = 0;  // the row the next pass feeds (tokens[P] known, tokens[P + 1] sampled by it)
  std::vector<int> answer;
};

int run_gen(int argc, char** argv) {
  const std::string dir = argv[2];
  const std::string req_path = arg_after(argc, argv, "--requests", "");
  if (req_path.empty()) fatal("gen: --requests <file.jsonl>");
  const int batch = std::clamp(std::stoi(arg_after(argc, argv, "--batch", "8")), 1, 8);
  const long max_pos = std::stol(arg_after(argc, argv, "--max-pos", "65536"));  // KV positions in flight
  const double limit_ms = 1000.0 * std::stod(arg_after(argc, argv, "--time-limit", "1e9"));
  // past this, requests in flight are dropped (not written; a later run answers them): a slowed machine must not hold
  // the GPU lock long past --time-limit
  const double hard_ms = limit_ms + 1000.0 * std::stod(arg_after(argc, argv, "--grace", "240"));

  Tokenizer tok;
  tok.load(dir + "/tokenizer.json");
  const std::string src = chat_template_source(dir);
  if (src.empty()) fatal("no chat template in %s", dir.c_str());
  minja::chat_template tmpl(src, "", "<|im_end|>");

  std::vector<Request> reqs;
  {
    std::ifstream f(req_path);
    if (!f) fatal("cannot read %s", req_path.c_str());
    std::string line;
    while (std::getline(f, line)) {
      if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
      const ojson j = ojson::parse(line);
      Request r;
      r.id = j.value("id", std::to_string(reqs.size()));
      if (j.contains("prompt_ids")) {
        for (auto& v : j["prompt_ids"]) r.prompt.push_back(v.get<int>());
      } else {
        ChatRequest c;
        c.messages = j.at("messages");
        if (j.contains("tools")) c.tools = j["tools"];
        if (j.contains("template_kwargs")) c.template_kwargs = j["template_kwargs"];
        r.prompt = tok.encode(render_chat(tmpl, c, true));
      }
      r.sp.temperature = j.value("temperature", 0.0f);
      r.sp.top_p = j.value("top_p", 0.95f);
      r.sp.top_k = j.value("top_k", 20);
      r.sp.seed = j.value("seed", 0u);
      r.max_tokens = j.value("max_tokens", 1024);
      if (j.contains("meta")) r.meta = j["meta"];
      if (r.prompt.size() < 2) fatal("request %s: empty prompt", r.id.c_str());
      reqs.push_back(std::move(r));
    }
  }
  Gpu& g = Gpu::get();
  std::unique_ptr<Qwen35> mp = load_qwen(dir, 2048);
  Qwen35& m = *mp;
  std::vector<int> stops = m.config().eos;
  if (int im_end = tok.find("<|im_end|>"); im_end >= 0) stops.push_back(im_end);
  auto is_stop = [&](int t) { return std::find(stops.begin(), stops.end(), t) != stops.end(); };

  int max_limit = 0;
  for (const Request& r : reqs) max_limit = std::max(max_limit, int(r.prompt.size()) + r.max_tokens + 16);
  std::vector<std::unique_ptr<Seq>> pool;
  std::vector<Seq*> free_seqs;
  for (int i = 0; i < batch; ++i) {
    pool.push_back(m.new_seq(max_limit, 0));
    free_seqs.push_back(pool.back().get());
  }
  // shared system turns: prefilled once into a sequence of their own, its KV and recurrent state copied per request
  const SystemTurn sys(tok);
  struct Shared {
    std::vector<int> toks;
    std::unique_ptr<Seq> s;
    Checkpoint ck;
  };
  std::deque<Shared> shared;
  auto shared_for = [&](const std::vector<int>& prompt, int L) -> Shared* {
    for (Shared& t : shared)
      if (int(t.toks.size()) == L && std::equal(t.toks.begin(), t.toks.end(), prompt.begin())) return &t;
    if (shared.size() >= 4) return nullptr;
    Shared t;
    t.toks.assign(prompt.begin(), prompt.begin() + L);
    t.s = m.new_seq(L + 16);
    memcpy(t.s->tokens(), t.toks.data(), size_t(L) * 4);
    SamplingParams none;
    for (int p = 0; p < L; p += m.prefill_chunk()) {
      auto e = g.begin();
      m.encode_prefill(*e, *t.s, p, std::min(m.prefill_chunk(), L - p), none, 0, false);
      e->commit_and_wait();
      if (!m.ane_finish_chunk()) fatal("the NI_ANE prefill split failed; rerun with NI_ANE=0");
    }
    t.ck = m.new_checkpoint();
    auto e = g.begin();
    m.save(*e, *t.s, t.ck, L);
    e->commit_and_wait();
    shared.push_back(std::move(t));
    return &shared.back();
  };
  const double t0 = now_ms();
  std::deque<Live> live;
  std::deque<size_t> pending;  // request indices not started, in file order
  for (size_t i = 0; i < reqs.size(); ++i) pending.push_back(i);
  size_t started = 0;
  long in_flight = 0, tokens_out = 0, passes = 0;
  double pass_ms = 0;
  auto emit = [&](Live& l, const char* finish) {
    ojson o;
    o["id"] = l.r.id;
    o["prompt"] = l.r.prompt;
    o["answer"] = l.answer;
    o["finish"] = finish;
    o["temperature"] = l.r.sp.temperature;
    o["seed"] = l.r.sp.seed;
    if (!l.r.meta.is_null()) o["meta"] = l.r.meta;
    const std::string s = o.dump() + "\n";
    out_bytes(s.data(), s.size());
    fflush(g_out);
  };
  while (true) {
    // admit requests: a free batch row, the KV positions in flight within --max-pos (one at a time beyond it); the
    // first of the next 32 that fits
    for (size_t k = 0; k < pending.size() && k < 32 && int(live.size()) < batch && now_ms() - t0 < limit_ms;) {
      const Request& r = reqs[pending[k]];
      const long need = long(r.prompt.size()) + r.max_tokens + 16;
      if (!live.empty() && in_flight + need > max_pos) {
        ++k;
        continue;
      }
      pending.erase(pending.begin() + long(k));
      ++started;
      Live l;
      l.r = r;
      l.s = free_seqs.back();
      free_seqs.pop_back();
      m.reset(*l.s);
      m.drop_kv(*l.s);
      if (!m.reserve(*l.s, int(need))) fatal("gen: no memory for a KV cache of %ld positions", need);
      const int n = int(r.prompt.size());
      memcpy(l.s->tokens(), r.prompt.data(), size_t(n) * 4);
      SamplingParams none;
      int start = 0;
      const int L = sys.length(r.prompt);
      if (L >= kMinShared && L < n - 1) {
        if (Shared* t = shared_for(r.prompt, L)) {
          auto e = g.begin();
          copy_prefix_kv(m, *e, *t->s, *l.s, L);
          m.restore(*e, *l.s, t->ck);
          e->commit_and_wait();
          start = L;
        }
      }
      for (int p = start; p < n - 1; p += m.prefill_chunk()) {  // rows [start, n - 1); the last prompt row is the first pass's
        const int T = std::min(m.prefill_chunk(), n - 1 - p);
        auto e = g.begin();
        m.encode_prefill(*e, *l.s, p, T, none, 0, false);
        e->commit_and_wait();
        if (!m.ane_finish_chunk()) fatal("the NI_ANE prefill split failed; rerun with NI_ANE=0");
      }
      l.P = n - 1;
      in_flight += need;
      live.push_back(std::move(l));
    }
    if (live.empty()) break;
    if (now_ms() - t0 > hard_ms) {
      logi("gen: hard time limit: %zu requests in flight dropped", live.size());
      break;
    }
    {
      const double tp = now_ms();
      auto e = g.begin_grouped();
      std::vector<Qwen35::VerifyGroup> gr;
      for (Live& l : live) {
        Qwen35::VerifyGroup v{l.s, l.P, 1, l.r.sp, nullptr, 0};
        v.tok_out = true;
        gr.push_back(v);
      }
      m.encode_verify_multi(*e, gr);
      for (Live& l : live) m.commit_rows(*l.s, 1);
      e->commit_and_wait();
      pass_ms += now_ms() - tp;
      ++passes;
    }
    for (auto it = live.begin(); it != live.end();) {
      Live& l = *it;
      const int t = l.s->tokens()[l.P + 1];
      l.answer.push_back(t);
      ++l.P;
      ++tokens_out;
      const bool stop = is_stop(t), full = int(l.answer.size()) >= l.r.max_tokens;
      if (stop || full) {
        emit(l, stop ? "stop" : "length");
        in_flight -= long(l.r.prompt.size()) + l.r.max_tokens + 16;
        free_seqs.push_back(l.s);
        it = live.erase(it);
      } else {
        ++it;
      }
    }
    if (passes % 500 == 0)
      logi("gen: %zu/%zu requests started, %ld tokens, %.1f tok/s, %.2f ms per pass", started, reqs.size(), tokens_out,
           tokens_out / ((now_ms() - t0) / 1000.0), pass_ms / passes);
  }
  logi("gen: %zu requests, %ld tokens in %.0f s (%.1f tok/s), %.2f ms per batched pass", started, tokens_out,
       (now_ms() - t0) / 1000.0, tokens_out / ((now_ms() - t0) / 1000.0), passes ? pass_ms / passes : 0.0);
  return 0;
}

// ---- data: the training stream ----

struct SeqRec {
  uint32_t index = 0;
  std::string id;
  std::vector<int> prompt, answer;
  bool sampled = false;
};

// top-k of x[0, n) (descending; ties by lower index): the k largest values and their indices
void topk(const float* x, int n, int k, int* idx, float* val) {
  // a sorted list of k, most elements rejected by one comparison against its smallest
  int cnt = 0;
  for (int i = 0; i < n; ++i) {
    const float v = x[i];
    if (cnt == k && !(v > val[k - 1])) continue;
    int j = cnt < k ? cnt++ : k - 1;
    while (j > 0 && v > val[j - 1]) {
      val[j] = val[j - 1];
      idx[j] = idx[j - 1];
      --j;
    }
    val[j] = v;
    idx[j] = i;
  }
  for (int j = cnt; j < k; ++j) idx[j] = -1, val[j] = -std::numeric_limits<float>::infinity();
}

int run_data(int argc, char** argv) {
  const std::string dir = argv[2];
  const std::string seqs_path = arg_after(argc, argv, "--seqs", "");
  if (seqs_path.empty()) fatal("data: --seqs <answers.jsonl>");
  std::string ddir = arg_after(argc, argv, "--dflash", "auto");
  const std::string anchors_arg = arg_after(argc, argv, "--anchors", "96");
  const bool all_anchors = anchors_arg == "all";
  const int n_anchors = all_anchors ? 0 : std::stoi(anchors_arg);
  const uint64_t seed = std::stoull(arg_after(argc, argv, "--seed", "1"));
  const int max_answer = std::stoi(arg_after(argc, argv, "--max-answer", "100000"));
  int shard_i = 0, shard_n = 1;
  {
    const std::string sh = arg_after(argc, argv, "--shard", "0/1");
    if (sscanf(sh.c_str(), "%d/%d", &shard_i, &shard_n) != 2 || shard_n < 1 || shard_i < 0 || shard_i >= shard_n)
      fatal("--shard i/n");
  }
  std::vector<SeqRec> seqs;
  {
    std::ifstream f(seqs_path);
    if (!f) fatal("cannot read %s", seqs_path.c_str());
    std::string line;
    uint32_t idx = 0;
    while (std::getline(f, line)) {
      if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
      const uint32_t i = idx++;
      if (int(i % uint32_t(shard_n)) != shard_i) continue;
      const nlohmann::json j = nlohmann::json::parse(line);
      SeqRec r;
      r.index = i;
      r.id = j.value("id", std::to_string(i));
      r.prompt = j.at("prompt").get<std::vector<int>>();
      r.answer = j.at("answer").get<std::vector<int>>();
      if (int(r.answer.size()) > max_answer) r.answer.resize(size_t(max_answer));
      r.sampled = j.value("temperature", 0.0) > 0;
      if (r.answer.size() < 2 || r.prompt.empty()) continue;
      seqs.push_back(std::move(r));
    }
  }
  // sequences sharing a prompt are processed together: one prompt prefill, its state restored for each answer
  std::map<std::vector<int>, std::vector<size_t>> groups;
  for (size_t i = 0; i < seqs.size(); ++i) groups[seqs[i].prompt].push_back(i);

  Gpu& g = Gpu::get();
  std::unique_ptr<Qwen35> mp = load_qwen(dir, 2048);
  Qwen35& m = *mp;
  if (ddir == "auto") ddir = DFlash2::find(dir, m.config());
  if (ddir.empty()) fatal("no DFlash drafter for %s", dir.c_str());
  DFlash2 dfl(m, ddir);
  const DFlash2Config& C = dfl.config();
  const int H = C.H, V = m.config().vocab;
  const uint32_t Vd = dfl.head_rows();
  const size_t nH = C.target_layers.size() * size_t(H);
  std::vector<int> col_of(size_t(V), -1);  // token id -> draft logit column
  for (uint32_t c = 0; c < Vd; ++c) col_of[dfl.head_ids() ? dfl.head_ids()[c] : c] = int(c);
  auto d = dfl.new_seq(), dsnap = dfl.new_seq();
  Checkpoint ck = m.new_checkpoint();

  {  // stream header
    nlohmann::json hj;
    hj["format"] = "kflash-stream";
    hj["version"] = 1;
    hj["H"] = H;
    hj["B"] = kPos + 1;
    hj["K"] = kCand;
    hj["topn"] = kTopN;
    hj["vocab"] = V;
    hj["draft_head_rows"] = Vd;
    hj["model"] = dir;
    hj["drafter"] = ddir;
    hj["drafter_v1"] = C.v1;
    hj["drafter_bits"] = dfl.bits();
    hj["mask_id"] = C.mask_id;
    hj["anchors"] = anchors_arg;
    hj["seed"] = seed;
    hj["shard"] = std::to_string(shard_i) + "/" + std::to_string(shard_n);
    hj["sequences"] = seqs.size();
    const std::string s = hj.dump();
    out_bytes("KFSTRM01", 8);
    out_val(uint32_t(s.size()));
    out_bytes(s.data(), s.size());
  }

  constexpr int LC = 256;  // answer rows per prefill chunk (their logits: LC x vocab fp32)
  Buffer logits = g.alloc(size_t(LC) * V * 4, "kf_logits");
  // one target sequence and one capture buffer for every sequence (the engine never frees a Seq)
  int max_n = 0, max_answer_rows = 0;
  for (const SeqRec& r : seqs) {
    max_n = std::max(max_n, int(r.prompt.size() + r.answer.size()));
    max_answer_rows = std::max(max_answer_rows, int(r.answer.size()));
  }
  auto seq = m.new_seq(max_n + 16, 0);
  Seq& s = *seq;
  Buffer capt = g.alloc(size_t(max_answer_rows + 8) * nH * 2, "kf_capt");
  const unsigned nth = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
  SamplingParams greedy;
  const double t0 = now_ms();
  long n_seq = 0, n_anc = 0;
  double gpu_draft = 0;

  // a long shared system turn is prefilled once, its state (and the drafter ring) restored for every prompt that
  // starts with it; the groups are in token order, so prompts sharing it are adjacent
  Tokenizer tok;
  tok.load(dir + "/tokenizer.json");
  const SystemTurn system_turn(tok);
  std::vector<int> sys_cached;
  Checkpoint ck_sys = m.new_checkpoint();
  auto dsys = dfl.new_seq();
  auto prefill = [&](int from, int to) {
    for (int p = from; p < to; p += m.prefill_chunk()) {
      const int T = std::min(m.prefill_chunk(), to - p);
      auto e = g.begin();
      m.encode_prefill(*e, s, p, T, greedy, 0, false);
      e->commit_and_wait();
      if (!m.ane_finish_chunk()) fatal("the NI_ANE prefill split failed; rerun with NI_ANE=0");
    }
  };

  for (auto& [prompt, members] : groups) {
    const int np = int(prompt.size());
    int group_n = 0;
    for (size_t i : members) group_n = std::max(group_n, np + int(seqs[i].answer.size()));
    if (!m.reserve(s, group_n + 16)) fatal("data: no memory for a KV cache of %d positions", group_n + 16);
    memcpy(s.tokens(), prompt.data(), size_t(np) * 4);
    const int L = system_turn.length(prompt);
    int start = 0;
    if (L >= kMinShared && L < np) {
      const std::vector<int> sys(prompt.begin(), prompt.begin() + L);
      if (sys != sys_cached) {  // the system turn's prefill, its state and ring kept aside
        m.reset(s);
        DFlash2::reset(*d);
        dfl.begin_prefill(L, 0);
        prefill(0, L);
        dfl.end_prefill();
        auto e = g.begin();
        dfl.encode_prefill_commit(*e, *d);
        m.save(*e, s, ck_sys, L);
        dfl.encode_ring_copy(*e, *d, *dsys);
        e->commit_and_wait();
        sys_cached = sys;
      } else {
        auto e = g.begin();
        m.restore(*e, s, ck_sys);
        dfl.encode_ring_copy(*e, *dsys, *d);
        e->commit_and_wait();
      }
      start = L;
    } else {
      sys_cached.clear();  // positions [0, np) are overwritten below
      m.reset(s);
      DFlash2::reset(*d);
    }
    // prompt prefill with the drafter's prompt capture (the engine's path), then its state kept aside
    dfl.begin_prefill(np, start);
    prefill(start, np);
    dfl.end_prefill();
    {
      auto e = g.begin();
      dfl.encode_prefill_commit(*e, *d);
      if (members.size() > 1) {
        m.save(*e, s, ck, np);
        dfl.encode_ring_copy(*e, *d, *dsnap);
      }
      e->commit_and_wait();
    }
    for (size_t mi = 0; mi < members.size(); ++mi) {
      const SeqRec& r = seqs[members[mi]];
      const int n = np + int(r.answer.size());
      if (mi > 0) {  // back to the state after the prompt
        auto e = g.begin();
        m.restore(*e, s, ck);
        dfl.encode_ring_copy(*e, *dsnap, *d);
        e->commit_and_wait();
      }
      std::vector<int> toks(prompt);
      toks.insert(toks.end(), r.answer.begin(), r.answer.end());
      memcpy(s.tokens(), toks.data(), size_t(n) * 4);
      // answer rows [np, n): drafter captures of every row, the target's distribution of rows np .. n - 2
      const int nrows = n - 1 - np;
      CaptureHook hk;
      hk.slot.assign(m.config().layers, -1);
      for (size_t i = 0; i < C.target_layers.size(); ++i) hk.slot[C.target_layers[i]] = int(i);
      hk.ld = uint32_t(nH);
      hk.pos_base = np;
      hk.rows = n - np;
      hk.dst = capt;
      std::vector<int> top_ids(size_t(nrows) * kTopN);
      std::vector<float> top_p(size_t(nrows) * kTopN);
      m.set_capture(&hk);
      for (int p0 = np; p0 < n; p0 += LC) {
        const int T = std::min(LC, n - p0);
        auto e = g.begin();
        m.encode_prefill(*e, s, p0, T, greedy, 0, false);
        m.encode_logits_rows(*e, T, logits);
        e->commit_and_wait();
        if (!m.ane_finish_chunk()) fatal("the NI_ANE prefill split failed; rerun with NI_ANE=0");
        std::vector<std::thread> th;
        for (unsigned t = 0; t < nth; ++t)
          th.emplace_back([&, t] {
            std::vector<float> val(kTopN);
            for (int rr = int(t); rr < T; rr += int(nth)) {
              const int row = p0 + rr - np;
              if (row >= nrows) continue;
              const float* x = logits.as<float>() + size_t(rr) * V;
              int* ti = top_ids.data() + size_t(row) * kTopN;
              float* tp = top_p.data() + size_t(row) * kTopN;
              topk(x, V, kTopN, ti, val.data());
              const float mx = val[0];
              double z = 0;
              for (int i = 0; i < V; ++i) z += std::exp(double(x[i] - mx));
              for (int k = 0; k < kTopN; ++k) tp[k] = float(std::exp(double(val[k] - mx)) / z);
            }
          });
        for (auto& t : th) t.join();
      }
      m.set_capture(nullptr);
      // SEQ1
      {
        const uint32_t idl = uint32_t(r.id.size()), idpad = (idl + 3) / 4 * 4;
        const uint32_t bytes = 4 * 5 + idpad + 4 * uint32_t(n) + uint32_t(nrows) * kTopN * 8;
        out_tag("SEQ1", bytes);
        out_val(r.index);
        out_val(uint32_t(r.sampled ? 1 : 0));
        out_val(int32_t(n));
        out_val(int32_t(np));
        out_val(idl);
        std::string idb = r.id;
        idb.resize(idpad, '\0');
        out_bytes(idb.data(), idpad);
        out_bytes(toks.data(), size_t(n) * 4);
        out_bytes(top_ids.data(), top_ids.size() * 4);
        out_bytes(top_p.data(), top_p.size() * 4);
      }
      // anchors: answer positions P with a next token (P <= n - 2), ascending
      std::vector<int> anchors;
      for (int P = np; P <= n - 2; ++P) anchors.push_back(P);
      if (!all_anchors && int(anchors.size()) > n_anchors) {
        std::mt19937_64 rng(seed * 0x9E3779B97F4A7C15ULL + r.index * 0xBF58476D1CE4E5B9ULL + 1);
        std::shuffle(anchors.begin(), anchors.end(), rng);
        anchors.resize(size_t(n_anchors));
        std::sort(anchors.begin(), anchors.end());
      }
      const float nan = std::numeric_limits<float>::quiet_NaN();
      std::vector<_Float16> h16(size_t(kPos) * H);
      std::vector<int> cand(size_t(kPos) * kCand);
      std::vector<float> cu(size_t(kPos) * kCand), ut(size_t(kPos) * kTopN), un(kPos);
      for (int P : anchors) {
        auto e = g.begin();
        while (d->ctx < P) {  // ring rows [ctx, P) from the answer captures, 8 at a time
          const int k = std::min(8, P - d->ctx);
          d->stage = capt.view(size_t(d->ctx - np) * nH * 2, size_t(8) * nH * 2);
          d->staged = k;
          dfl.encode_commit_staged(*e, *d);  // d->ctx += k
        }
        dfl.encode_draft(*e, s, *d, P, greedy);
        e->commit_and_wait();
        gpu_draft += e->gpu_ms();
        memcpy(s.tokens() + P + 1, toks.data() + P + 1, size_t(std::min(kPos, n - 1 - P)) * 4);  // the walk wrote there
        const int npos = std::min(kPos, n - 1 - P);
        const float* hn = dfl.final_hidden();
        const float* lg = dfl.logits();
        for (int p = 1; p <= kPos; ++p) {
          const int q = p - 1;
          for (int i = 0; i < H; ++i) h16[size_t(q) * H + i] = _Float16(hn[size_t(p) * H + i]);
          const float* row = lg + size_t(p) * Vd;
          int* ci = cand.data() + size_t(q) * kCand;
          float* cv = cu.data() + size_t(q) * kCand;
          topk(row, int(Vd), kCand, ci, cv);
          for (int k = 0; k < kCand; ++k) ci[k] = ci[k] < 0 ? -1 : int(dfl.head_ids() ? dfl.head_ids()[ci[k]] : uint32_t(ci[k]));
          const int trow = P + p - 1 - np;  // target row index (row P + p - 1)
          for (int k = 0; k < kTopN; ++k) {
            float v = nan;
            if (p <= npos) {
              const int id = top_ids[size_t(trow) * kTopN + k];
              if (id >= 0 && col_of[size_t(id)] >= 0) v = row[col_of[size_t(id)]];
            }
            ut[size_t(q) * kTopN + k] = v;
          }
          un[q] = p <= npos && col_of[size_t(toks[size_t(P + p)])] >= 0 ? row[col_of[size_t(toks[size_t(P + p)])]] : nan;
        }
        const uint32_t bytes = 12 + uint32_t(h16.size()) * 2 + uint32_t(cand.size()) * 8 + uint32_t(ut.size()) * 4 + kPos * 4;
        out_tag("ANC1", bytes);
        out_val(r.index);
        out_val(int32_t(P));
        out_val(int32_t(npos));
        out_bytes(h16.data(), h16.size() * 2);
        out_bytes(cand.data(), cand.size() * 4);
        out_bytes(cu.data(), cu.size() * 4);
        out_bytes(ut.data(), ut.size() * 4);
        out_bytes(un.data(), un.size() * 4);
        ++n_anc;
      }
      out_tag("END1", 8);
      out_val(r.index);
      out_val(int32_t(anchors.size()));
      fflush(g_out);
      ++n_seq;
      if (n_seq % 20 == 0)
        logi("data: %ld/%zu sequences, %ld anchors, %.1f s, drafting pass %.2f ms GPU", n_seq, seqs.size(), n_anc,
             (now_ms() - t0) / 1000.0, n_anc ? gpu_draft / n_anc : 0.0);
    }
  }
  logi("data: %ld sequences, %ld anchors in %.1f s (drafting pass %.2f ms GPU)", n_seq, n_anc, (now_ms() - t0) / 1000.0,
       n_anc ? gpu_draft / n_anc : 0.0);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || (strcmp(argv[1], "gen") && strcmp(argv[1], "data"))) {
    fprintf(stderr, "usage: kflash_datagen gen <model_dir> --requests R.jsonl [--batch 8] [--max-pos N] [--time-limit S]\n"
                    "       kflash_datagen data <model_dir> --seqs answers.jsonl [--dflash auto|dir] [--anchors N|all] "
                    "[--seed S] [--shard i/n] [--max-answer N]\n");
    return 1;
  }
  // stdout carries the output alone: anything the library prints goes to stderr
  const int fd = dup(1);
  dup2(2, 1);
  g_out = fdopen(fd, "wb");
  setvbuf(g_out, nullptr, _IOFBF, 1 << 20);
  const int rc = !strcmp(argv[1], "gen") ? run_gen(argc, argv) : run_data(argc, argv);
  fflush(g_out);
  return rc;
}

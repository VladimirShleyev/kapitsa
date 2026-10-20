#include "model/nh_drafter.h"
#include "model/dspark.h"
#include "model/lilicorr.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>

namespace ni {

namespace {

// the language model's hidden size: llm_config (Nemotron Omni releases), text_config, or the whole config.json
int target_hidden(const std::string& model_dir) {
  std::ifstream f(model_dir + "/config.json");
  if (!f) return 0;
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(f);
  } catch (...) {
    return 0;
  }
  const nlohmann::json* t = &j;
  for (const char* k : {"llm_config", "text_config"})
    if (j.is_object() && j.contains(k) && j[k].is_object()) {
      t = &j[k];
      break;
    }
  return t->is_object() ? t->value("hidden_size", 0) : 0;
}

}  // namespace

std::string NhDrafter::find(const std::string& model_dir) {
  const std::string l = LiliCorr::find(model_dir);
  if (!l.empty()) return l;
  const int hidden = target_hidden(model_dir);
  for (const char* sub : {"/dflash", "/dspark"})
    if (DSpark::is_dspark(model_dir + sub, hidden)) return model_dir + sub;
  return "";
}

std::string NhDrafter::kind(const std::string& dir) {
  if (DSpark::is_dspark(dir, target_hidden(dir))) return "dspark";
  return dir.empty() ? "" : "lilicorr";
}

size_t NhDrafter::plan_bytes(const std::string& dir, size_t* per_slot) {
  return kind(dir) == "dspark" ? DSpark::plan_bytes(dir, per_slot) : LiliCorr::plan_bytes(dir, per_slot);
}

std::unique_ptr<NhDrafter> NhDrafter::open(NemotronH& target, const std::string& dir) {
  if (kind(dir) == "dspark") return std::make_unique<DSpark>(target, dir);
  return std::make_unique<LiliCorr>(target, dir);
}

void NhDrafter::reset(LiliSeq& d) { d.ctx = d.hi = 0; }

void NhDrafter::rewind(LiliSeq& d, int pos) { d.ctx = pos; }

void NhDrafter::stage_init(LiliSeq& d) {
  if (d.stage) return;
  const NhCapture& v = verify_hook();
  d.stage = Gpu::get().alloc(size_t(kB) * v.ld * 4, "nh_draft_stage");
  d.staged = 0;
  d.stage_hook.feat = d.stage;
  d.stage_hook.ld = v.ld;
  d.stage_hook.slot = v.slot;
  d.stage_hook.pos_base = 0;
  d.stage_hook.rows = kB;
}

void NhDrafter::encode_commit_path(Encoder& e, LiliSeq& d, int m, const Buffer& path) {
  if (m <= 0) return;
  const NhCapture& v = verify_hook();
  if (!kfeat_) kfeat_ = Gpu::get().alloc(size_t(kB) * v.ld * 4, "nh_kf_feat");
  const uint32_t ld = v.ld;
  e.use(NemotronH::pipeline("nh_kf_gather"));
  e.buf(0, v.feat);
  e.buf(1, kfeat_);
  e.buf(2, path);
  e.val(3, ld);
  e.dispatch(uint32_t(m), (ld + 255) / 256, 1, 256);
  enc_commit_rows(e, d, kfeat_, m, d.ctx);
  d.ctx += m;
  d.hi = std::max(d.hi, d.ctx);
}

void NhDrafter::enc_kf_build(Encoder& e, NemotronH& tg, Seq& s, int P, const Buffer& cand, const Buffer& tlq, int K,
                             const Buffer& kfp) {
  const struct {
    uint32_t S, K, P;
  } bp{uint32_t(kB - 1), uint32_t(K), uint32_t(P)};
  e.use(NemotronH::pipeline("nh_kf_build"));
  e.buf(0, tlq);
  e.buf(1, cand);
  e.buf(2, kfp);
  e.buf(3, tg.tree_buf());
  e.buf(4, s.tok);
  e.buf(5, tg.live_rows());
  e.bytes(6, &bp, sizeof bp);
  e.dispatch(1, 1, 1, 32);
}

void NhDrafter::encode_commit_staged(Encoder& e, LiliSeq& d) {
  if (d.staged <= 0) return;
  enc_commit_rows(e, d, d.stage, d.staged, d.ctx);
  d.ctx += d.staged;
  d.hi = std::max(d.hi, d.ctx);
  d.staged = 0;
}

}  // namespace ni

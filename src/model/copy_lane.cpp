#include "model/copy_lane.h"

#include <algorithm>
#include <cstdlib>

namespace ni {

bool copy_lookup(const int* tok, int P, CopyChain& out, int min_match) {
  out.match = 0;
  if (P < 2) return false;
  const int a = tok[P - 1], b = tok[P];
  int best = -1, best_ml = 0, seen = 0;
  for (int i = P - 1; i >= 1; --i) {  // an occurrence of (a, b) ending at i < P
    if (tok[i] != b || tok[i - 1] != a) continue;
    int ml = 2;
    while (ml < CopyChain::kMaxMatch && i - ml >= 0 && tok[i - ml] == tok[P - ml]) ++ml;
    if (ml > best_ml) {
      best = i;
      best_ml = ml;
    }
    if (++seen >= CopyChain::kMaxOcc || best_ml >= CopyChain::kMaxMatch) break;
  }
  if (best < 0 || best_ml < min_match) return false;
  for (int j = 0; j < CopyChain::kLen; ++j) {
    const int p = best + 1 + j;  // what followed the occurrence; past the anchor, the chain's own earlier tokens
    out.tok[j] = p <= P ? tok[p] : out.tok[p - P - 1];
  }
  out.match = best_ml;
  out.expect = copy_expect(best_ml);
  return true;
}

float copy_expect(int match) {
  // Measured on the harness's recorded coding-agent session (bench/harness/agent-traces/opencode-ledger.json:
  // opencode fixing a Python project on Qwen3.6-35B-A3B, 19 requests: the prompts as agent_replay.py sends them and
  // the 3,558 answer tokens as the model writes them), teacher-forced at every answer token with copy_lookup itself:
  // the mean number of the 7 copied tokens right in a row, by match length (tools/copy_lane_check on
  // bench/harness/tools/copy_lane_seqs.py's sequences; positions 430 / 254 / 329 / 235 / 346 / 257 / 492 / 455, 760
  // without a copy chain).
  if (match < 2) return 0.0f;
  if (match == 2) return 1.09f;
  if (match == 3) return 2.38f;
  if (match <= 5) return 3.16f;
  if (match <= 7) return 3.74f;
  if (match <= 11) return 4.23f;
  if (match <= 15) return 4.41f;
  if (match <= 31) return 4.92f;
  return 5.96f;
}

bool copy_lane_enabled(bool by_default) {
  static const int forced = [] {  // NI_COPY: 0 = off, 1 = on, unset = the target's default
    const char* v = getenv("NI_COPY");
    return v && *v ? (atoi(v) != 0 ? 1 : 0) : -1;
  }();
  return forced < 0 ? by_default : forced == 1;
}

float copy_lane_margin(bool moe_target) {
  // Drafters rate themselves differently. Greedy runs on the agent session's requests 6, 7, 12, 14 and 18 (ni_run,
  // NI_COPY_LOG; the copy's outcome replayed from the known continuation), where the drafted chain was verified:
  // - Qwen3.6-35B-A3B (incoai's DFlash2 drafter): it accepted 4.35 drafts against an E_d of 5.45. A margin of -1.5
  //   took the copy in 88 of 226 cycles and accepted 4.66 drafts a cycle (4.35 at 0, 4.72 at -2), losing on none
  //   of the five requests (-2 lost on the plain-text one).
  // - Qwen3.8-27B (z-lab's DFlash2 drafter): 5.12 against 5.92; over 165 cycles -0.5 gained the most (+21 drafts;
  //   -1.5 none, and it cost decode speed: an edit answer of 102 tokens -10%).
  // - Qwen3.5-4B (z-lab's DFlash v1 drafter): 3.08 against 3.07; over 363 cycles 0 and -0.5 gained the most (+572
  //   drafts), -1.0 and lower lost on fresh code.
  // So -1.5 on the MoE target, -0.5 on the dense ones.
  static const char* v = getenv("NI_COPY_MARGIN");
  if (v && *v) return float(atof(v));
  return moe_target ? -1.5f : -0.5f;
}

int copy_run_match() {
  static const int n = [] {
    const char* v = getenv("NI_COPY_RUN");
    return v && *v ? std::max(0, atoi(v)) : kCopyRunMatch;
  }();
  return n;
}

}  // namespace ni

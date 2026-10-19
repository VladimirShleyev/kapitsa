// Copy lane: the conversation's own continuation as a block of drafts. Coding agents write much of what they have
// read: an edit's old text from a file read earlier, diff context, paths, identifiers, the same JSON keys. At an anchor
// the copy lane finds the earlier occurrence of the text before it that matches furthest back and proposes the tokens
// that followed it. The block drafters attend to a window of the context (DFlash2: the last 2047 positions); the
// lookup reads the whole conversation, on the CPU while the GPU drafts. Which chain a cycle verifies, the drafter's or
// this one, is decided on the GPU (DFlash2::set_copy, dfl_copy_pick); the target verifies either exactly.
#pragma once

namespace ni {

struct CopyChain {
  static constexpr int kLen = 7;        // proposals (an 8-row verify: the anchor + 7)
  static constexpr int kMaxMatch = 64;  // backward match length counted at most
  static constexpr int kMaxOcc = 256;   // earlier occurrences of the last two tokens examined (the most recent)
  // shortest match proposed: in fresh code (bench/prompts/code_35b.txt on the 35B) chains after a 2-token match were
  // almost never right (0 copied tokens in 14 of 19 cycles), while an agent's 2-token matches continue 1.09 on average
  static constexpr int kMinMatch = 3;
  int tok[kLen] = {};
  int match = 0;         // tokens matched backwards from the anchor, its own included (kMinMatch .. kMaxMatch)
  float expect = 0.0f;   // expected proposals right in a row (of kLen), from the match length (copy_expect)
};

// The copy chain at anchor position P: tok[0 .. P] are the conversation's tokens, tok[P] the anchor. Among the
// earlier occurrences of (tok[P-1], tok[P]), the one whose match with the text before the anchor reaches furthest back
// (ties: the most recent); the chain is what followed it, continued periodically past P when the occurrence is less
// than kLen tokens back. false: no occurrence matching at least min_match tokens.
bool copy_lookup(const int* tok, int P, CopyChain& out, int min_match = CopyChain::kMinMatch);

// Expected proposals right in a row (of 7) of a copy chain whose match is `match` tokens long.
float copy_expect(int match);

// NI_COPY: 0 = off, 1 = on, unset: by_default (DFlash2::copy_lane: the target's measured default). NI_COPY_MARGIN: a
// cycle verifies the copy chain when its expectation exceeds the drafter's own by more than this (default -1.5 on MoE
// targets, whose drafter rates itself high, -0.5 on dense ones; a large negative value takes the copy whenever one
// exists).
bool copy_lane_enabled(bool by_default);
float copy_lane_margin(bool moe_target);

}  // namespace ni

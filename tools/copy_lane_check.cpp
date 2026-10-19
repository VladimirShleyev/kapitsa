// The copy lane's lookup (src/model/copy_lane.h), on the CPU.
//   copy_lane_check                checks copy_lookup on small hand-made sequences
//   copy_lane_check <seqs file>    the calibration behind copy_expect: one sequence per line, "<history ids>|<output
//                                  ids>" (comma-separated; bench/harness/tools/copy_lane_seqs.py writes them from an
//                                  agent trace); at every output position, teacher-forced, the copy chain of the text
//                                  so far and how many of its 7 tokens the output continues with, by match length
#include "model/copy_lane.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

static int fails = 0;

static void expect_chain(const char* name, const std::vector<int>& seq, bool found, int match, const std::vector<int>& chain,
                         int min_match = 2) {
  CopyChain c;
  const bool f = copy_lookup(seq.data(), int(seq.size()) - 1, c, min_match);
  bool ok = f == found && (!found || c.match == match);
  for (size_t j = 0; ok && found && j < chain.size(); ++j) ok = c.tok[j] == chain[j];
  printf("%-44s %s", name, ok ? "ok" : "FAIL");
  if (!ok) {
    printf(" (found %d, match %d, chain", int(f), c.match);
    for (int j = 0; j < CopyChain::kLen; ++j) printf(" %d", c.tok[j]);
    printf(")");
    ++fails;
  }
  printf("\n");
}

static std::vector<int> parse_ids(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string x;
  while (std::getline(ss, x, ','))
    if (!x.empty()) v.push_back(atoi(x.c_str()));
  return v;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    // the anchor is the last token; the chain continues past it periodically
    expect_chain("periodic continuation", {5, 6, 7, 8, 9, 5, 6}, true, 2, {7, 8, 9, 5, 6, 7, 8});
    expect_chain("a run of one token", {3, 3, 3}, true, 2, {3, 3, 3, 3, 3, 3, 3});
    expect_chain("no earlier occurrence", {1, 2, 3, 4}, false, 0, {});
    expect_chain("too short", {1, 1}, false, 0, {});
    // (3, 4) occurs twice before the anchor: after 1 2 (matches 4 back) and after 9 (2 back); the longer match wins
    expect_chain("longest match over most recent",
                 {1, 2, 3, 4, 50, 51, 52, 53, 54, 55, 56, 9, 3, 4, 60, 61, 62, 63, 64, 65, 66, 1, 2, 3, 4}, true, 4,
                 {50, 51, 52, 53, 54, 55, 56});
    // equal matches: the most recent
    expect_chain("ties to the most recent", {7, 3, 4, 10, 11, 12, 13, 14, 15, 16, 8, 3, 4, 20, 21, 22, 23, 24, 25, 26, 9, 3, 4},
                 true, 2, {20, 21, 22, 23, 24, 25, 26});
    // the engine's minimum (kMinMatch = 3): a 2-token match is no chain, a 3-token one is
    expect_chain("a 2-token match below kMinMatch", {5, 6, 7, 8, 9, 5, 6}, false, 0, {}, CopyChain::kMinMatch);
    expect_chain("a 3-token match at kMinMatch", {4, 5, 6, 7, 8, 4, 5, 6}, true, 3, {7, 8, 4, 5, 6, 7, 8},
                 CopyChain::kMinMatch);
    CopyChain c;
    std::vector<int> big(70, 1);
    copy_lookup(big.data(), 69, c, 2);
    printf("%-44s %s\n", "match counted up to kMaxMatch", c.match == CopyChain::kMaxMatch ? "ok" : "FAIL");
    if (c.match != CopyChain::kMaxMatch) ++fails;
    printf("%s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
  }
  std::ifstream f(argv[1]);
  if (!f) {
    fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  const int nb = 8;
  const int lo[nb] = {2, 3, 4, 6, 8, 12, 16, 32}, hi[nb] = {2, 3, 5, 7, 11, 15, 31, 64};
  long pos[nb] = {}, sum[nb] = {}, full[nb] = {}, total = 0, none = 0;
  std::string line;
  int nseq = 0;
  while (std::getline(f, line)) {
    const size_t bar = line.find('|');
    if (bar == std::string::npos) continue;
    std::vector<int> seq = parse_ids(line.substr(0, bar));
    const std::vector<int> out = parse_ids(line.substr(bar + 1));
    ++nseq;
    for (size_t t = 0; t < out.size(); ++t) {
      ++total;
      CopyChain c;
      if (!copy_lookup(seq.data(), int(seq.size()) - 1, c, 2)) {  // every match length, the 2-token ones too
        ++none;
      } else {
        int a = 0;
        while (a < CopyChain::kLen && t + a < out.size() && c.tok[a] == out[t + a]) ++a;
        for (int b = 0; b < nb; ++b)
          if (c.match >= lo[b] && c.match <= hi[b]) {
            ++pos[b];
            sum[b] += a;
            full[b] += a == CopyChain::kLen;
          }
      }
      seq.push_back(out[t]);
    }
  }
  printf("%d sequences, %ld output positions, no copy chain at %ld (the engine also skips matches under %d)\n", nseq,
         total, none, CopyChain::kMinMatch);
  printf("match length | positions | mean tokens right in a row (of 7) | all 7 | copy_expect\n");
  for (int b = 0; b < nb; ++b)
    if (pos[b])
      printf("%5d-%-5d  | %9ld | %5.2f | %5.1f%% | %.2f\n", lo[b], hi[b], pos[b], double(sum[b]) / pos[b],
             100.0 * full[b] / pos[b], copy_expect(lo[b]));
  return 0;
}

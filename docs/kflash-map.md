# KFlash: where the work is

A map of the KFlash work as of 2026-10-20: which branch holds what, what was measured, what is still open, and how
to resume. KFlash is Kapitsa's tree-verify generation of speculative decoding. The target checks a best-first tree of
the drafter's candidates instead of one chain, exactly, sized by a budget that each Mac measures for itself.

## Branches on origin

| Branch | What it holds | State |
| --- | --- | --- |
| `main` | the released engine | KFlash not merged; `main` has moved on since KFlash forked at `701d431` |
| `kflash` | everything: trees (Qwen + Nemotron), the shared budget, the candidate decoder (trainer, Metal, results), budget persistence, 16/32-row verifies | integration branch, no PRs, tip `43947bf` |
| `kflash-trees` | the release candidate: trees only, no decoder | this branch; forked from `701d431` |
| PR #109 (`claude/dflash-3-kapitsa-f294ab`) | the copy lane (prompt lookup inside the DFlash loop) | open; an experiment, not KFlash |
| `claude/copy-lane-ext` | the copy lane's follow-up work in progress: batched decode, Nemotron, skipping unused drafts | `4c58527`, marked WIP; it builds; `dflash_accept_test` passes on one seed stream but fails one new case on the default stream (unchecked); never run on a model; not KFlash |

### `kflash-trees` commits (no decoder code; authored by the user)

| Commit | What |
| --- | --- |
| `2c597ef` | The trees, squashed from `kflash` `e8c84b0..44e0e72`, before any decoder code: Qwen dense trees (7 nodes in one 8-row tile), Nemotron-H trees (Mamba-2 tree scan), the shared tree descriptor / builder / acceptance, and KfBudget (per-depth calibration, cost per row count, cached per model / drafter / chip). |
| `4afa619` | `kapitsa serve` saves the KfBudget cache on shutdown and every 256 cycles; every tool measures a cycle's whole GPU time (same patch as `kflash` `bc46716`). |
| `2acc61b` | One switch and one rule: trees on dense targets, the chain on MoE targets. `NI_KFLASH=1` forces trees, `NI_KFLASH=0` chains everywhere. |
| `fa3edc2` | `tools/tensor_probe.mm` builds with SDKs before macOS 26 (found on the M1 Pro fleet Mac). |
| (this file) | the map |

**In progress, not pushed yet: dynamic trees on MoE targets.**
- The user's rule: trees on every target; on MoE targets the tree's size is dynamic, never a fixed 8 rows. The
  budget picks 1..8 live rows per cycle, and the verify's expert work covers only those rows.
- Nemotron-H's budget is already dynamic (rows 1..8). The Qwen3.6-35B needs it built.
- Owner: the session "KFlash: 16- and 32-row tree verifies on dense Qwen" (local id
  `local_4e5ea98d-7f93-4f73-97ca-9355be24eb94`) on the coordinator Mac, working toward `kflash-trees`.
- If that session is gone, restart from this brief. Allow `set_rows(0xFF, 8)` for MoE in `src/model/dflash2.cpp`;
  have the 35B's row-scaled verify work (`enc_moe_rows`, `moe_unique_rows`, attention, GDN, sampling) honour the
  tree's live row count, GPU-driven like the chain's dynamic depth; then turn trees on by default everywhere. Check
  exactness, then A/B against the 35B's chain.

### `kflash` commit map

| Commits | Piece | Status |
| --- | --- | --- |
| `e8c84b0` | Nemotron-H trees: DSpark / LiLiCorr lattices, Mamba-2 tree scan, exact acceptance | in `kflash-trees` |
| `a7aaea4`, `7224485`, `de5c4d0`, `18c6576` | Qwen dense trees; online calibration and measured cost; `tree_sim.py`; one shared descriptor / builder / acceptance / budget | in `kflash-trees` |
| `44e0e72` | docs for `NI_DFLASH_TREE` / `NI_KFLASH` | in `kflash-trees` |
| `75260f0`, `8d220ef`, `d3f55b4`, `d9139c4`, `fc2a4a4`, `8f4351f`, `7a9af69`, `e5aad53` | the candidate decoder in Metal, rescoring the tree, plus its pass cap, 8-bit option and budget-cache key | `kflash` only (opt-in) |
| `169c4a0` and the RESULTS commits (`bb0dd71`, `bcadc5c`, `fcb8be9`) | the decoder's data generator, MLX trainer, evaluation, `tools/kflash/RESULTS.md` | `kflash` only |
| `bc46716` | budget persistence | in `kflash-trees` as `4afa619` |
| `43947bf` | 16- and 32-row tree verifies on dense Qwen; the size rule picks 8 / 16 / 32 per cycle | `kflash` only, not yet decided for `kflash-trees` |

**The switches differ by branch.** On `kflash`, `NI_KFLASH=1` on a Qwen model loads the candidate decoder; on
Nemotron it turns trees on. On `kflash-trees`, `NI_KFLASH` is only the tree switch.

## Measured (end to end, output identical to plain decoding, unless noted)

Trees of 7 nodes vs `main`, interleaved, M4 Pro 48 GB:

| | code | prose | agent requests |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B | +9.9% | +18.6% | +5.4% |
| Qwen3.5-4B | +9.8% | +10.8% | +3.7% |

- **M1 Pro 16 GB (Qwen3.5-4B):** +2% to +16% on code across greedy and sampled runs, +6% to +13% on the agent replay.
  The budget picks 8 rows, and on that GPU 8 rows cost 6% more than 1 row.
- **Qwen3.6-35B MoE with a forced 7-node tree (always 8 rows):** tokens per cycle rose 15-24%, but cycle time rose
  16-20%. Speed: code 0%, prose +1%, agent requests -3% / +2%, sampled code -7%. The chain stays until the dynamic
  MoE tree lands.
- **16/32-row verifies on the M4 Pro:**
  - At 4-bit, 16 rows cost 1.6-1.75× an 8-row verify for 1.09-1.14× the tokens, so the rule keeps 8.
  - At 8-bit, it keeps 8.
  - Qwen3.5-4B in bf16: 16 rows cost 1.05× and give +11-16% tok/s (sizes forced; the interleaved bf16 A/B was
    deferred).
- **Replay, bigger trees:** tokens per cycle over the chain on the 27B are +4% at 7 nodes, +16% at 15 and +21% at 31
  on the agent session, and +9% / +23% / +28% on code. That is before the cost of the extra rows.
- **Candidate decoder** (`kflash` only, opt-in):
  - Replay on mixed held-out text, accepted tokens over calibrated trees: +2.7% / +5.0% / +6.5% at 8 / 16 / 32 rows.
  - Engine, 48 unseen agent turns: +12.9% decode speed for a session's mix (+28% on tool-call turns).
  - Code and prose: -2% to -4%.
  - It learned the harness's tool-call syntax, not general text. Retraining is parked.

## Not in git (by design)

- **Private docs:** the KFlash spec, parked ideas (including the online-tuned selector, for Alisher), the fleet Mac
  setup script, and the sessions' notes and result tables. They live outside this public repo, in a private handoff
  bundle.
- **The 4B decoder's weights and its training data:** in the coordinator Mac's model cache
  (`<model>/kflash/`, `~/.cache/kapitsa/kflash-dev/`). Never committed.
- **The fleet Mac's local branches:** `benchmarks/m1pro-*` predate this work and look superseded by #135;
  `m1pro/kflash-budget-save` landed as `bc46716`.

## Open items

1. Dynamic MoE trees (in progress, above), then the 35B A/B.
2. Nemotron speed: tree vs chain on Lightning. This needs the 23 GB model; the user said not to download it for now.
3. 16/32-row verifies: the interleaved A/B is deferred; decide whether they join `kflash-trees`.
4. Merging to `main` waits for the user. `main` has moved on, so bring the branch up to date first, then rerun the
   checks and a short A/B.
5. Parked:
   - the decoder's measured on/off switch;
   - the online-tuned selector;
   - the Qwen3.6-35B on the shared budget (subsumed by item 1);
   - the 4-bit verify on the M5's matrix units, which needs an M5 in the fleet.

## How to resume

```bash
git fetch origin && git switch kflash-trees && git pull --ff-only
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 12
# exactness (GPU; hold the shared lock, at most 20 minutes per hold)
lockf -k -t 3000 ~/.cache/nace-gpu.lock build/dflash_accept_test 20000
python3 tools/nh_synth.py /tmp/nh_small --dspark --small && build/nh_tree_check /tmp/nh_small
build/ni_run ~/.cache/kapitsa/models/qwen3.5-4b-4bit @bench/prompts/code_27b.txt 256 0 --dflash none > /tmp/plain.txt
build/ni_run ~/.cache/kapitsa/models/qwen3.5-4b-4bit @bench/prompts/code_27b.txt 256 0 --dflash auto --ref /tmp/plain.txt
# A/B: interleave chain and tree runs (C B B C) on code, prose and agent requests; read tok/s and tok/cycle
NI_KFLASH=0 build/ni_run <model> @bench/prompts/code_27b.txt 512 0 --dflash auto --ignore-eos
build/ni_run <model> @bench/prompts/code_27b.txt 512 0 --dflash auto --ignore-eos
```

The KfBudget cache is at `~/.cache/kapitsa/kflash/<model>-<drafter>-b<bits>-<chip>.json`. Delete it to make the
budget re-measure.

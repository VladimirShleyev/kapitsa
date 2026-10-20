# KFlash M2: the first candidate decoder, Qwen3.5-4B (DFlash v1 backbone)

2026-10-20, M4 Pro 48 GB. The decoder of `internal/KFLASH.md` §1 (2 causal layers, d 1024, 8 x 128 heads, SwiGLU 2816,
33.6M parameters; format in README.md) trained on the 4-bit Qwen3.5-4B's own answers with its DFlash v1 drafter frozen,
evaluated offline on held-out sequences by replaying greedy decoding exactly.

## Data

1640 requests from local sources (`prompts.py`), answered by the 4-bit target itself through `kflash_datagen gen`
(batched 8-row verify decoding); 1532 answers were kept (the last 108 were dropped when the machine ran out of swap and
disk). 831k answer tokens over 4.55M prompt tokens:

| domain | train seqs | held-out seqs | prompt tokens (mean) | answer tokens (train) | sampled (T = 0.6) |
| --- | --- | --- | --- | --- | --- |
| agent (opencode session, 19 requests) | 57 | 2 | 17.2k | 9.2k | 44 |
| agent_syn (agent turns over local files) | 398 | 39 | 7.5k | 83.4k | 146 |
| code (write / read) | 532 | 37 | 336 | 371.8k | 205 |
| chat (25% thinking on) | 246 | 18 | 105 | 129.5k | 92 |
| math (70% thinking on) | 178 | 23 | 42 | 169.5k | 82 |
| bench (speed prompts) | - | 2 | 2.4k | - | - |

The stream (`kflash_datagen data`) was checked against the engine's own DFlash decodes: replaying them through the
stream gives the engine's cycles (code_27b 512 tokens: 173 cycles, 2.960 vs 2.954 tokens per cycle; an agent turn,
768 tokens: 150 cycles, 5.120 vs 5.113; one cycle differs at a near-tie in each).

## Training

`train.py` on the stream, MLX on the M4 Pro: batch 256 anchors (up to 1792 block positions), AdamW 3e-4 with cosine
decay to 3e-5, gradient clip 1, the agent sequences drafted at every answer position once (weight 2), the others with
128 fresh anchors in each of 2 epochs, every anchor used 3 times on average: 330k drafting passes generated,
4873 steps = 1.25M anchor samples (up to 8.7M block positions). About 1.5 h of GPU in 18-minute lock chunks
(3 h of wall time beside the other sessions' benchmarks; the run was resumed once from its step-400 checkpoint after a session restart).

Held-out validation during training (121 held-out sequences x 32 anchors; mean accepted length per anchor of the
greedy chain, teacher-forced = exact):

| step | 400 | 800 | 1200 | 2000 | 2800 | 3600 | **4000** | 4400 | 4873 (end) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decoder chain | 3.281 | 3.367 | 3.429 | 3.479 | 3.519 | 3.560 | **3.595** | 3.576 | 3.570 |
| DFlash v1 chain | 3.218 | | | | | | 3.218 | | |

The exported checkpoint is step 4000 (+11.7% over the v1 chain on this metric). The training loss on sequences already
seen fell far below the held-out loss (LK 0.1-0.2 vs 0.47): 1411 training answers are few for 33.6M parameters, and
more distinct answers, not more passes, are what the next run needs (answer text can come from any server: the
target's distributions and the drafter's features along it are computed locally).

## Held-out results

`eval.py` on the 121 held-out sequences (the target's own greedy answers to prompts whose sources were never trained
on): every answer position drafted (67,137 passes), greedy decodes replayed exactly cycle by cycle (16,883 cycles).
Every method sees the same 32 candidates per block row; V verify rows = the anchor + V - 1 drafted nodes (the chain is
8). The v1 trees are best-first over DFlash v1's marginals, raw or with M1's two online calibrations (per-depth
temperature and coverage, `dfl_tree_learn`; per-depth logistic on logit q, KfBudget), learned from past cycles and
started warm. Tokens per cycle (accepted + 1):

| method | agent | agent_syn | code | chat | math | bench | all |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DFlash v1 chain (today) | 3.550 | 3.357 | 3.726 | 2.942 | 5.584 | 5.273 | 3.981 |
| decoder chain | 4.464 | 4.279 | 3.627 | 2.802 | 5.794 | 5.034 | 4.064 |
| decoder chain, 4-bit | 4.502 | 4.284 | 3.643 | 2.778 | 5.794 | 5.010 | 4.065 |
| v1 tree 8, raw marginals | 3.891 | 3.744 | 4.091 | 3.309 | 5.794 | 5.476 | 4.337 |
| v1 tree 8, M1 temperature calib | 3.891 | 3.743 | 4.093 | 3.297 | 5.823 | 5.389 | 4.339 |
| v1 tree 8, M1 logistic calib | 3.835 | 3.741 | 4.094 | 3.326 | 5.813 | 5.361 | 4.343 |
| decoder tree 8, q | 4.721 | 4.502 | 4.020 | 3.187 | 6.014 | 5.241 | 4.425 |
| decoder tree 8, q + logistic calib | 4.763 | 4.512 | 4.022 | 3.207 | 6.014 | 5.268 | 4.433 |
| decoder tree 8, acceptance head | 4.619 | 4.377 | 3.870 | 3.114 | 5.658 | 5.020 | 4.254 |
| decoder tree 8, q, 4-bit | 4.785 | 4.504 | 4.015 | 3.195 | 6.013 | 5.268 | 4.426 |
| v1 tree 16, raw marginals | 4.132 | 4.040 | 4.539 | 3.655 | 6.284 | 6.053 | 4.758 |
| v1 tree 16, M1 temperature calib | 4.164 | 4.046 | 4.541 | 3.654 | 6.309 | 5.982 | 4.764 |
| v1 tree 16, M1 logistic calib | 4.148 | 4.041 | 4.531 | 3.656 | 6.311 | 5.908 | 4.760 |
| decoder tree 16, q | 5.009 | 4.799 | 4.564 | 3.597 | 6.658 | 5.948 | 4.945 |
| decoder tree 16, q + logistic calib | 5.033 | 4.802 | 4.567 | 3.622 | 6.659 | 5.948 | 4.954 |
| decoder tree 16, acceptance head | 4.940 | 4.758 | 4.427 | 3.534 | 6.442 | 5.715 | 4.819 |
| decoder tree 16, q, 4-bit | 5.081 | 4.814 | 4.570 | 3.607 | 6.646 | 6.018 | 4.952 |
| v1 tree 32, raw marginals | 4.442 | 4.239 | 4.834 | 3.921 | 6.532 | 6.276 | 5.032 |
| v1 tree 32, M1 temperature calib | 4.405 | 4.236 | 4.823 | 3.923 | 6.546 | 6.315 | 5.030 |
| v1 tree 32, M1 logistic calib | 4.391 | 4.236 | 4.828 | 3.920 | 6.549 | 6.276 | 5.031 |
| decoder tree 32, q | 5.230 | 4.974 | 4.927 | 3.943 | 6.986 | 6.238 | 5.286 |
| decoder tree 32, q + logistic calib | 5.230 | 4.994 | 4.938 | 3.959 | 6.973 | 6.238 | 5.295 |
| decoder tree 32, acceptance head | 5.155 | 4.937 | 4.824 | 3.844 | 6.846 | 6.200 | 5.182 |
| decoder tree 32, q, 4-bit | 5.230 | 4.975 | 4.929 | 3.948 | 6.989 | 6.238 | 5.289 |

| verify rows | best v1 tree (calibrated) | best decoder tree | gain, tokens/cycle | gain, accepted/cycle |
| --- | --- | --- | --- | --- |
| 8 | 4.343 (v1-kf-tree-8) | 4.433 (dec-kf-tree-8) | +2.1% | +2.7% |
| 16 | 4.764 (v1-temp-tree-16) | 4.954 (dec-kf-tree-16) | +4.0% | +5.0% |
| 32 | 5.032 (v1-tree-32) | 5.295 (dec-kf-tree-32) | +5.2% | +6.5% |
| chain (8) | 3.981 (v1-chain) | 4.064 (dec-chain) | +2.1% | +2.8% |

Per domain, the best decoder tree against the best calibrated v1 tree, in accepted tokens per cycle:

| rows | agent (opencode) | agent_syn | code | chat | math | bench | all |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | +30.2% | +28.0% | -2.3% | -5.1% | +4.0% | -4.6% | +2.7% |
| 16 | +27.5% | +24.8% | +0.7% | -1.3% | +6.6% | -2.1% | +5.0% |
| 32 | +22.9% | +23.3% | +2.7% | +1.2% | +7.9% | -1.4% | +6.5% |

- The gain follows how much the held-out answers share with the training answers. Share of each held-out answer's
  8-grams that occur in some training answer: agent (requests 6 and 18 of the opencode session whose other 16 requests
  were trained on) 58%, agent_syn (held-out files, but the training's task templates and tool flow) 51%, math (templated
  problems) 42%, code 7%, chat 3.5%. The gain orders the same way: +30%, +28%, +4%, -2%, -5% at 8 rows.
- M1's calibrations move the v1 trees by under 0.2%; the backbone's raw marginals already rank well for best-first.
- The decoder's own q ranks better than its acceptance head; the head is under-confident (predicted 0.75 vs actual 0.87
  at the chain's pick, ECE 0.118: it was trained with sampled answers, evaluated on greedy ones), which M1's online
  logistic calibration corrects at run time.
- Quantized to 4-bit group 64 as the engine runs it, the decoder is within 0.1% of bf16.

### Unseen agent turns

To separate learned templates from generalization: 48 agent turns that share nothing with training but the harness
(`prompts.py --unseen`): held-out files only, task phrasings the training never used, and a tool flow it never saw (a
grep, then a read), answered greedily by the target. Their answers still share 41% of their 8-grams with training
answers (the harness's tool-call syntax), against 51% for the template-matched set. 2650 cycles:

| verify rows | best v1 tree (calibrated) | best decoder tree | gain, tokens/cycle | gain, accepted/cycle |
| --- | --- | --- | --- | --- |
| 8 | 3.398 | 3.980 | +17.1% | +24.3% |
| 16 | 3.727 | 4.343 | +16.5% | +22.6% |
| 32 | 3.959 | 4.578 | +15.6% | +20.9% |
| chain (8) | 3.052 | 3.682 | +20.6% | +30.7% |

So the agent gain is not overlap with training requests: on coding-agent turns the decoder clears the decision bar
(about +10% accepted tokens per cycle over trees on the published drafter alone) twice over, at every row count. DFlash
v1, trained on generic text, drafts this harness's tool-call text poorly (its chain: 3.05 tokens per cycle here against
3.7 on code), and the decoder learns it from the target's own answers. On new code and chat the decoder adds nothing
(-5% to +3%), and over the whole held-out mix it stays under the bar (+2.7 / +5.0 / +6.5%).

## In the engine

M3's Metal decoder (branch claude/zen-nightingale-c62aa7 at fc2a4a4: projections at 4-bit group 64, trees ranked by
the decoder's q) with these weights, greedy, interleaved A B B A on the M4 Pro, every output identical to plain decoding.
A = DFlash v1 (its chain with NI_DFLASH_TREE=0, else the lattice's best-first tree, M1's default), B = the decoder
(NI_KFLASH=1); the budget's calibration caches were warm in both arms. Four unseen agent turns (the longest answers of the
set above) and the speed benchmark's code and prose prompts:

| prompt (answer tokens) | chain tok/cycle | chain tok/s | tree tok/cycle | tree tok/s |
| --- | --- | --- | --- | --- |
| agent_unseen-2 (768) | 5.218 -> 5.364 | 270.4 -> 268.4 (-0.7%) | 5.383 -> 5.620 | 275.1 -> 279.4 (+1.5%) |
| agent_unseen-43 (743) | 4.240 -> 4.443 | 217.6 -> 219.4 (+0.8%) | 4.652 -> 4.963 | 234.2 -> 242.6 (+3.6%) |
| agent_unseen-11 (423) | 3.546 -> 3.802 | 184.7 -> 188.2 (+1.9%) | 3.854 -> 4.199 | 196.8 -> 207.4 (+5.4%) |
| agent_unseen-37 (394) | 4.010 -> 3.930 | 211.9 -> 198.5 (-6.3%) | 4.296 -> 4.319 | 222.1 -> 216.4 (-2.5%) |
| code_27b (512) | 2.954 -> 2.971 | 178.4 -> 170.3 (-4.5%) | 3.308 -> 3.276 | 192.8 -> 183.9 (-4.6%) |
| p700 prose (512) | 4.867 -> 4.867 | 283.3 -> 273.8 (-3.4%) | 4.986 -> 5.110 | 284.6 -> 281.9 (-0.9%) |

The engine reproduces the offline replay sequence by sequence (decoder chain at 4-bit, offline vs engine: 3.811 / 3.802,
4.449 / 4.443, 3.940 / 3.930, 5.333 / 5.364). These long agent turns (edits and explanations) gain little, and the
offline replay says why. Over the 48 unseen agent turns, the decoder chain's gain by answer length:

| answer | turns | DFlash v1 chain | decoder chain (4-bit) |
| --- | --- | --- | --- |
| under 150 tokens (32 of 33 are pure tool calls) | 33 | 2.272 | 3.270 (+43.9%) |
| 150-300 tokens | 7 | 3.022 | 3.402 (+12.6%) |
| over 300 tokens | 8 | 4.079 | 4.210 (+3.2%) |

Over all 48 unseen agent turns, each decoded to its exact answer length (M3's run on origin/kflash, trees ranked by q,
A = the lattice's best-first tree, B = the decoder's tree at pass caps 1 / 3 (3 is the default), arm order rotated per
turn, per-arm budget caches warming across turns), session totals over 8069 tokens:

| | A: lattice tree | B, cap 1 | B, cap 3 |
| --- | --- | --- | --- |
| cycles | 2415 | 2127 | 2063 |
| tokens per cycle | 3.341 | 3.794 (+13.6%) | 3.911 (+17.1%) |
| decode time | 47.05 s | 42.20 s (-10.3%) | 41.46 s (-11.9%) |
| decode tok/s | 171.5 | 191.2 (+11.5%) | 194.6 (+13.5%) |

By answer length (cap 3 vs A, tok/s): under 150 tokens +28.4%, 150-300 +2.9%, over 300 +4.3%; weighted like the
opencode session (10 of 15 turns under 150 tokens) +12.9%. The +17.1% tokens per cycle is exactly the offline replay's
number for this set. Every output equals plain decoding except two near-ties (logit gaps 0.0021-0.0022): one where the
recorded answer itself came from a slightly different numeric path, one where the tree verify takes the other token.

So what the decoder learned is the harness's tool-call syntax (tags, parameter names, paths and values copied from the
context), which DFlash v1 drafts badly; prose and code inside long turns gain about what new code does, nothing. A
previous A/B on agent requests 12 and 18 compared trees, not chains, on requests that share the training conversation;
it is withdrawn.

## Next

- The decision: on coding-agent turns the decoder clears the bar offline (+21-24% accepted tokens per cycle over
  calibrated v1 trees on unseen turns) and gives +12.9% decode speed for a session's mix of turns in the engine, but the
  gain is the tool-call syntax (+28% speed on tool-call turns, +3-4% on long ones); on everything else it adds nothing
  (code -2 to -4% speed for its 0.8 ms of drafting per cycle), and over the mixed held-out set it stays under the bar
  (+3-7%). As trained here it is a tool-call specialist, not a better drafter.
- Data, not passes: the training loss on seen sequences fell to a third of the held-out loss. More distinct code, chat
  and reasoning answers (their text can come from any server; the target's distributions and the drafter's features
  along them are computed here), agent weight 1 instead of 2, and a larger greedy term for greedy decoding.
- The copy candidates of the spec (context continuations added to C_p) would give the decoder the material for long
  agent turns, which copy from the files they read.
- The same pipeline for the 35B and 27B (M4), where answer generation dominates the cost here.

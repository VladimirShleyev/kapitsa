# KFlash candidate decoder: data, training, evaluation, checkpoint (M2)

The candidate decoder of `internal/KFLASH.md` §1, trained as in §3: a small autoregressive model that re-scores
the block drafter's top-32 candidates at each block position given every token drafted before it, so drafting can
verify trees instead of DFlash's chain of independent marginals. The backbone (DFlash / DFlash2) and the 4-bit target
stay frozen; everything they contribute comes from the engine through the stream below.

| step | tool | GPU |
| --- | --- | --- |
| 1. request set (local sources only) | `prompts.py --out requests.jsonl` | no |
| 2. the target's own answers | `gen_answers.py` -> `kflash_datagen gen` (batched 8-row verify: Kapitsa's numerics) | yes, 18-min lock chunks |
| 3. training stream | `kflash_datagen data` (spawned by the trainer, one process per shard and epoch) | yes |
| 4. training | `train.py --answers answers.jsonl --out DIR` (MLX) | yes, 18-min lock chunks |
| 5. offline evaluation | `eval.py --seqs DIR/heldout_seqs.jsonl --ckpt DIR/best` | yes |
| 6. export | `export.py --ckpt DIR/best --model <model dir>` -> `<model>/kflash/` | no |

The first decoder (Qwen3.5-4B) and its held-out and engine numbers: RESULTS.md.

Venv: any Python with `mlx` and `numpy` (the MLX trainer); `prompts.py` and `gen_answers.py` are stdlib only.
`kflash_datagen` builds with the other tools (`cmake --build build --target kflash_datagen`). Every GPU step
holds `~/.cache/nace-gpu.lock`.

## Data

`prompts.py` builds requests from local material only: the recorded opencode session (`bench/harness/agent-traces`,
its 19 requests as the chat template renders them with tools, thinking off), coding-agent turns over local source files
(the session's system prompt and tools, a task, the agent's `read` of the file and the file as the tool returns it),
code to write (tasks x languages) and to read (explain / review / test / port a local snippet), chat (explanations,
writing, advice, README summaries, translations; a quarter with thinking on) and math word problems (thinking on).
Held-out by source (file, topic, task), never by answer: agent requests 6, 12 and 18, ~10% of the rest, and the speed
benchmark prompts (`bench/prompts` code_27b, p700, p4000). Train prompts get a greedy answer or one sampled at T = 0.6
(agent requests: greedy and three seeds), held-out prompts a greedy answer (the evaluation replays greedy decoding).

`kflash_datagen gen` renders requests with the model's chat template exactly as the engine does (agent request 6
reproduces the harness's token ids), shares the prefill of a long common system turn (the agent prompts' 5.5k-token
system prompt with tools: its KV and recurrent state are copied into each sequence; answers identical to a full
prefill), and decodes up to 8 requests at once, one row each in the engine's batched 8-row verify.

### Stream format (`kflash_datagen data`, little-endian)

`"KFSTRM01"`, u32 n, n bytes of JSON (H, B, K, topn, vocab, draft_head_rows, model, drafter, anchors, seed, shard),
then records `u32 tag, u32 payload bytes, payload`:

| tag | payload |
| --- | --- |
| `SEQ1` | u32 seq (line in --seqs), u32 flags (bit 0: sampled answer), i32 n, i32 n_prompt, u32 id_len, id (padded to 4), i32 tokens[n], then for rows r = n_prompt .. n - 2 (row r = the target's distribution of token r + 1): i32 top_ids[nrows][64], f32 top_p[nrows][64] (softmax over the vocabulary, descending) |
| `ANC1` | u32 seq, i32 P (anchor), i32 npos (= min(7, n - 1 - P)), then for block rows p = 1..7 (token P + p, target row P + p - 1): f16 h[7][H] (the drafter's final normed hidden = its draft head's input), i32 cand[7][32] (top-32 ids of the draft head, descending), f32 u[7][32] (raw draft logits), f32 u_tgt[7][64] (draft logits of the target's top-64 ids), f32 u_next[7] (draft logit of tokens[P + p]); NaN outside the draft head or past npos |
| `END1` | u32 seq, i32 anchors |

A drafting pass at anchor P sees the ring the decode loop would have: positions [P - 2047, P) committed from the
target's captures, block [tokens[P], mask x 7]. Checked against the engine: replaying the engine's own DFlash decode
of `bench/prompts/code_27b.txt` (512 tokens, greedy) through the stream gives 173 cycles and 2.960 tokens per cycle
with the same acceptance by position; the engine measured 173 cycles and 2.954 (one cycle differs at a near-tie).
`stream.py` reads it; `stream.example()` joins an anchor with its sequence's tokens and target rows.

## Decoder

Per drafting pass the backbone gives, for block rows r = 1..7, h_r, C_r (top-32) and U_r. A tree node at depth j (the
root = the anchor, depth 0) is expanded by decoder row j over its branch (root .. node), which scores its children, the
candidates C_{j+1}:

    x_j = W_h h_{j+1} + W_e (RMSNorm(E[tok_j]) * emb_norm) + pos[j]             tok_0 = anchor, tok_j = the node's token
    2 causal pre-norm layers (MHA 8 x 128, no RoPE, no q/k norm, scale 1/sqrt(128); SwiGLU 2816), y_j = RMSNorm(x, norm)
    z_j = W_o y_j                                                                (d -> H)
    S_j[k] = logit_scale[j] * U_{j+1}[k] + <z_j, E_out[C_{j+1}[k]]>                 q_j = softmax(S_j / T_q)
    A_j[k, r] = <w_y[r], y_j> + <w_e[r], E_out[c] * rsqrt(mean(E_out[c]^2) + 1e-6)> + w_q[r] * log_softmax(S_j)[k] + b[r]
                r = 0: P(c = target argmax | path) (greedy); r = 1: min(1, p_target(c) / q(c)) at T = 1 (sampled)

E is the target's token embedding table and E_out its LM head (tied on the 4B: one table), rows dequantized from the
target's own 4-bit tables. U_r[k] = <h_r, E_out[c]> (the draft head is the target's LM head), so S = <logit_scale h + z,
E_out[c]>: the decoder corrects the backbone's hidden. W_o = 0 at initialization: the decoder starts as the backbone.
RMSNorm(x) = x * rsqrt(mean(x^2) + 1e-6) * w. `model.py` is the reference forward (MLX); `ref_vectors.py` writes its
outputs on recorded drafting passes for a GPU implementation to compare against.

### Checkpoint: `<model>/kflash/{config.json, model.safetensors}`

| tensor | shape | dtype |
| --- | --- | --- |
| `in_proj.weight` (W_h) | [1024, H] | bf16 |
| `emb_norm.weight` | [H] | bf16 |
| `emb_proj.weight` (W_e) | [1024, H] | bf16 |
| `pos_embed` | [16, 1024] (row j) | bf16 |
| `layers.{0,1}.input_layernorm.weight`, `layers.{0,1}.post_attention_layernorm.weight` | [1024] | bf16 |
| `layers.{0,1}.self_attn.{q,k,v,o}_proj.weight` | [1024, 1024] | bf16 |
| `layers.{0,1}.mlp.{gate,up}_proj.weight` / `down_proj.weight` | [2816, 1024] / [1024, 2816] | bf16 |
| `norm.weight` | [1024] | bf16 |
| `out_proj.weight` (W_o) | [H, 1024] | bf16 |
| `logit_scale` | [16] (row j) | f32 |
| `acc_head.weight` = per row [w_y \| w_e \| w_q] | [2, 1024 + H + 1] | f32 |
| `acc_head.bias` | [2] | f32 |

The engine quantizes the 2-D `*_proj` weights at load to MLX affine 4-bit group 64 (mx.quantize's rounding, as the
DFlash2 loader does); everything else is read as stored. config.json: hidden_size, num_hidden_layers,
num_attention_heads, head_dim, intermediate_size, target_hidden_size, vocab_size, num_candidates (32), max_positions
(16), rms_norm_eps, block_size (8), architectures ["KFlashCandidateDecoder"], format_version 1, emb_norm, logit_scale,
acc_rows, backbone ("dflash" / "dflash2"), backbone_dir, and the training run's metadata.

## Training

Teacher-forced along the target's path, per block row: the LK loss (lambda KL + (1 - lambda) TV, lambda =
exp(-2 stopgrad(1 - TV))) between the target's distribution and q over the 32 candidates plus the path token when it is
outside them (its draft logit from the stream), + 0.1 x NLL of the target's argmax, + 0.5 x BCE of the two acceptance
rows; rows after the first one whose path token is not among the 32 candidates are masked (no tree reaches them). AdamW,
warmup then cosine decay, gradient clip 1. Generator runs stream into a replay buffer, each anchor used `--reuse` times
on average: the agent sequences once at every answer position (their 5-20k-token prompts cost more to prefill than
drafting all of their ~200 answer positions), weighted x2; the others with 128 fresh anchors in each of 3 epochs.

## Evaluation

`eval.py` drafts at every answer position of the held-out sequences and replays greedy decoding exactly (the sequences
are the target's own greedy answers, so its argmax along the accepted path is known): a cycle at anchor P accepts the
longest drafted path whose tokens equal the target's argmax, the next starts at P + accepted + 1. Methods: the v1 chain
(each row's top-1: what Kapitsa runs today on DFlash v1), best-first trees over the backbone's marginals, the decoder's
greedy chain, and best-first trees over the decoder's path probability or its acceptance head, with V verify rows (the
anchor + V - 1 drafted nodes; the chain is V = 8).

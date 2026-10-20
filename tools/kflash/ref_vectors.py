#!/usr/bin/env python3
"""Reference vectors for the Metal decoder (M3): a decoder checkpoint's outputs on recorded drafting passes, so the GPU
implementation can be checked row by row against this MLX forward (model.py).

Writes <out>/reference.safetensors with, for N anchors (block rows 1..7, path = the target's tokens):
  h [N, 7, H] f16          the backbone's final normed hidden of rows 1..7
  tok [N, 8] i32           the path: tok[:, 0] = the anchor, tok[:, j] = the token of block row j
  cand [N, 7, 32] i32      the backbone's top-32 per row;  u [N, 7, 32] f32 their draft logits
  y [N, 7, d] f32          the decoder's final normed hidden of path rows 0..6 (row j expands the node at depth j)
  z [N, 7, H] f32          W_o y
  S [N, 7, 32] f32         scores of the children (row j: candidates of block row j + 1)
  A [N, 7, 32, 2] f32      acceptance logits (greedy, sampled)
Row j of the outputs depends only on path rows 0..j (causal), so a tree node's expansion is row dep(n) of its branch.

usage: ref_vectors.py --ckpt DIR --stream FILE --out DIR [--n 16] [--cpu]
"""
import argparse, os, sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as dm  # noqa: E402
import stream as ds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.expanduser("~/.cache/kapitsa/models/qwen3.5-4b-4bit"))
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.cpu:
        mx.set_default_device(mx.cpu)
    m, cfg = dm.load_checkpoint(a.ckpt)
    te = dm.TargetEmbeddings(a.model)
    _, ex = ds.examples(open(a.stream, "rb"))
    exs = []
    for e in ex:
        if e["valid"].all():
            exs.append(e)
        if len(exs) >= a.n:
            break
    h = np.stack([e["h"] for e in exs])
    tok = np.stack([np.concatenate([e["tok_in"], e["tok_next"][-1:]]) for e in exs]).astype(np.int32)
    cand = np.stack([e["cand"] for e in exs])
    u = np.stack([e["u"] for e in exs]).astype(np.float32)
    hh = mx.array(h.astype(np.float32))
    y, z = m.trunk(hh, te.emb(mx.array(tok[:, :7])))
    ce = te.out(mx.array(cand))
    S = m.scores(z, mx.array(u), ce)
    logq = S - mx.logsumexp(S, axis=-1, keepdims=True)
    A = m.accept_logits(y, ce, logq)
    os.makedirs(a.out, exist_ok=True)
    mx.save_safetensors(os.path.join(a.out, "reference.safetensors"),
                        {"h": mx.array(h), "tok": mx.array(tok), "cand": mx.array(cand), "u": mx.array(u),
                         "y": y, "z": z, "S": S, "A": A},
                        metadata={"ckpt": os.path.abspath(a.ckpt), "model": a.model})
    print(f"{len(exs)} anchors -> {a.out}/reference.safetensors")


if __name__ == "__main__":
    main()

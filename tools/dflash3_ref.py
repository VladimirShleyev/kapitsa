#!/usr/bin/env python3
"""DFlash3Turbo candidate decoder: random-init checkpoints and the NumPy reference the Metal decoder is checked against.

  dflash3_ref.py init <out_dir> --hidden H [--vocab 248320] [--seed 0] [--zero-out] [--backbone dflash|dflash2]
      A random checkpoint in the trainer's format (tools/dflash3/README.md: d 1024, 2 layers, 8 heads x 128, SwiGLU
      2816, W_h / W_e: H -> d, emb_norm, W_o: d -> H, logit_scale, the two-row acceptance head), safetensors +
      config.json. Every part is random (W_o, the embedding norm, the logit scales and the acceptance head too) so a
      check exercises all of it; --zero-out sets W_o = 0 and logit_scale = 1 (S = U: the backbone's own scores).

  dflash3_ref.py check <ckpt_dir> <dump_dir> [--no-mimic]
      Recomputes every decoder row of the engine's dump (tools/dflash3_check --out <dump_dir>) from the checkpoint:
      the 4-bit weights quantized here with mx.quantize's rounding (compared code for code with the engine's), each
      expanded node's whole branch run from scratch (no K/V cache), then S, softmax(S / T_q), the acceptance head and
      the walk's choices. --no-mimic skips the fp16 rounding of the GEMM inputs the engine's 8-row GEMMs apply.

Formulas (the trainer's, tools/dflash3/model.py): x_j = W_h h_{j+1} + W_e (RMSNorm(E[t_j]) * emb_norm) + pos[j],
pre-norm layers (RMSNorm x * rsqrt(mean x^2 + eps) * w; multi-head attention over the branch rows 0..j, no RoPE, scale
1/sqrt(128); SwiGLU silu(gate) * up), y = RMSNorm(x; norm), z = out_proj y,
S[k] = logit_scale[j] U_{j+1}[k] + <z, E_out[c_k]>, acceptance row r: sigmoid(<w_y[r], y> + <w_e[r], E_out[c_k]> /
rms(E_out[c_k]) + w_q[r] log softmax(S)[k] + b[r]) (r = 0 greedy, 1 sampled).
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

# ---- bf16 / safetensors ----


def f32_to_bf16(x):
    """float32 -> bf16 bits (round to nearest even), uint16."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    u = u + (((u >> 16) & 1) + 0x7FFF)
    return (u >> 16).astype(np.uint16)


def bf16_to_f32(b):
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def save_safetensors(path, tensors, meta=None):
    """tensors: name -> (dtype string 'BF16' / 'F32', ndarray of uint16 / float32)."""
    header, off, blobs = {}, 0, []
    for name, (dt, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    if meta:
        header["__metadata__"] = meta
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def load_safetensors(path):
    """name -> float32 ndarray (bf16 / f16 / f32 converted)."""
    raw = open(path, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    out, base = {}, 8 + n
    for name, v in header.items():
        if name == "__metadata__":
            continue
        a, b = v["data_offsets"]
        buf = raw[base + a:base + b]
        if v["dtype"] == "BF16":
            arr = bf16_to_f32(np.frombuffer(buf, np.uint16))
        elif v["dtype"] == "F16":
            arr = np.frombuffer(buf, np.float16).astype(np.float32)
        elif v["dtype"] == "F32":
            arr = np.frombuffer(buf, np.float32).copy()
        else:
            sys.exit(f"{path}: {name}: dtype {v['dtype']}")
        out[name] = arr.reshape(v["shape"])
    return out


# ---- mx.quantize (affine, 4-bit, group 64; MLX Metal rounding: IEEE fp32, round half away from zero) ----


def _round_away(x):
    x = np.asarray(x, dtype=np.float32).astype(np.float64)
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(np.float32)


def quantize4(w):
    """w: float32 [N, K] holding bf16 values -> codes uint8 [N, K], scales / biases as bf16 bits [N, K / 64]
    (dfl_quantize_cpu, even = false)."""
    N, K = w.shape
    g = w.reshape(N, K // 64, 64).astype(np.float32)
    lo = np.minimum(g.min(-1), np.float32(3.38953139e38))
    hi = np.maximum(g.max(-1), np.float32(0.0))
    n_bins = np.float32(15.0)
    scale = np.maximum((hi - lo) / n_bins, np.float32(1e-7)).astype(np.float32)
    side = np.abs(lo) > np.abs(hi)
    scale = np.where(side, scale, -scale).astype(np.float32)
    edge = np.where(side, lo, hi).astype(np.float32)
    q0 = _round_away(edge / scale)
    at_zero = q0 == 0
    scale = np.where(at_zero, scale, edge / np.where(at_zero, np.float32(1), q0)).astype(np.float32)
    bias = np.where(at_zero, np.float32(0), edge).astype(np.float32)
    codes = _round_away((g - bias[..., None]) / scale[..., None])
    codes = np.clip(codes, 0, 15).astype(np.uint8)
    return codes.reshape(N, K), f32_to_bf16(scale), f32_to_bf16(bias)


def dequant4(codes, S, B):
    N, K = codes.shape
    s = bf16_to_f32(S).reshape(N, K // 64, 1)
    b = bf16_to_f32(B).reshape(N, K // 64, 1)
    return (s * codes.reshape(N, K // 64, 64).astype(np.float32) + b).reshape(N, K)


def unpack4(W, K):
    """uint32 [N, K / 8] -> codes uint8 [N, K] (the engine's packing: code k in bits 4 (k % 8) of word k / 8)."""
    W = np.asarray(W, dtype=np.uint32)
    sh = (np.arange(8, dtype=np.uint32) * 4)
    return ((W[..., None] >> sh) & 15).astype(np.uint8).reshape(W.shape[0], K)


# ---- init ----


def cmd_init(a):
    rng = np.random.default_rng(a.seed)
    d, NL, nh, hd, I, H, mp = 1024, 2, 8, 128, 2816, a.hidden, 16
    t = {}

    def W(name, shape, std, mean=0.0):
        x = rng.standard_normal(shape, dtype=np.float32) * np.float32(std) + np.float32(mean)
        t[name] = ("BF16", f32_to_bf16(x))

    W("in_proj.weight", (d, H), 0.02)
    W("emb_norm.weight", (H,), 0.1, 1.0)
    W("emb_proj.weight", (d, H), 0.02)
    W("pos_embed", (mp, d), 0.02)
    for l in range(NL):
        p = f"layers.{l}."
        W(p + "input_layernorm.weight", (d,), 0.1, 1.0)
        W(p + "post_attention_layernorm.weight", (d,), 0.1, 1.0)
        for n in ("q", "k", "v", "o"):
            W(p + f"self_attn.{n}_proj.weight", (d, d), 0.02)
        W(p + "mlp.gate_proj.weight", (I, d), 0.02)
        W(p + "mlp.up_proj.weight", (I, d), 0.02)
        W(p + "mlp.down_proj.weight", (d, I), 0.02)
    W("norm.weight", (d,), 0.1, 1.0)
    W("out_proj.weight", (H, d), 0.0 if a.zero_out else 0.02)
    ls = np.ones(mp, np.float32) if a.zero_out else 1.0 + 0.1 * rng.standard_normal(mp, dtype=np.float32)
    t["logit_scale"] = ("F32", ls.astype(np.float32))
    acc = rng.standard_normal((2, d + H + 1), dtype=np.float32) * np.float32(0.02)
    acc[:, -1] = [1.0, 0.7]  # w_q
    t["acc_head.weight"] = ("F32", acc)
    t["acc_head.bias"] = ("F32", np.array([-1.0, -0.5], np.float32))
    os.makedirs(a.out, exist_ok=True)
    save_safetensors(os.path.join(a.out, "model.safetensors"), t, {"format": "dflash3-decoder-v1", "init": "random",
                                                                   "seed": str(a.seed)})
    cfg = {"hidden_size": d, "num_hidden_layers": NL, "num_attention_heads": nh, "head_dim": hd,
           "intermediate_size": I, "target_hidden_size": H, "vocab_size": a.vocab, "num_candidates": 32,
           "max_positions": mp, "rms_norm_eps": 1e-6, "block_size": 8, "architectures": ["DFlash3CandidateDecoder"],
           "format_version": 1, "emb_norm": True, "logit_scale": True, "acc_rows": ["greedy", "sampled"],
           "backbone": a.backbone, "init": "random" + (" (W_o = 0)" if a.zero_out else "")}
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    n = sum(v[1].size for v in t.values())
    print(f"{a.out}: {n / 1e6:.1f}M parameters, hidden {H}")


# ---- reference forward ----


class Decoder:
    def __init__(self, ckpt_dir, mimic=True):
        self.cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
        self.T = load_safetensors(os.path.join(ckpt_dir, "model.safetensors"))
        c = self.cfg
        self.d, self.NL, self.nh, self.hd, self.I, self.H = (c["hidden_size"], c["num_hidden_layers"],
                                                             c["num_attention_heads"], c["head_dim"],
                                                             c["intermediate_size"], c["target_hidden_size"])
        self.eps = np.float32(c.get("rms_norm_eps", 1e-6))
        self.mimic = mimic
        self.q = {}  # name -> (codes, S, B)
        T = self.T
        self.q["in"] = quantize4(T["in_proj.weight"])
        self.q["emb"] = quantize4(T["emb_proj.weight"])
        for l in range(self.NL):
            p = f"layers.{l}."
            self.q[p + "qkv"] = quantize4(np.concatenate([T[p + f"self_attn.{n}_proj.weight"] for n in "qkv"], axis=0))
            self.q[p + "o"] = quantize4(T[p + "self_attn.o_proj.weight"])
            self.q[p + "gate"] = quantize4(T[p + "mlp.gate_proj.weight"])
            self.q[p + "up"] = quantize4(T[p + "mlp.up_proj.weight"])
            self.q[p + "down"] = quantize4(T[p + "mlp.down_proj.weight"])
        self.q["out"] = quantize4(T["out_proj.weight"])
        self.W = {k: dequant4(*v) for k, v in self.q.items()}
        self.lscale = T.get("logit_scale", np.ones(16, np.float32)).reshape(-1)
        acc = T.get("acc_head.weight")
        if acc is not None:
            acc = acc.reshape(-1, self.d + self.H + 1)
            if acc.shape[0] == 1:
                acc = np.concatenate([acc, acc])
            self.wy, self.we, self.wq = acc[:, :self.d], acc[:, self.d:self.d + self.H], acc[:, -1]
            b = T["acc_head.bias"].reshape(-1)
            self.wb = np.concatenate([b, b]) if b.size == 1 else b
        else:
            self.wy = None

    def h16(self, x):
        return x.astype(np.float16).astype(np.float32) if self.mimic else x

    def mm(self, name, x):  # x [R, K] -> [R, N], fp64 accumulation of fp16-rounded inputs
        return (self.h16(x).astype(np.float64) @ self.W[name].T.astype(np.float64)).astype(np.float32)

    def rms(self, x, w):
        x64 = x.astype(np.float64)
        inv = 1.0 / np.sqrt((x64 * x64).mean(-1, keepdims=True) + float(self.eps))
        return (x64 * inv * w.astype(np.float64)).astype(np.float32)

    def branch(self, hrows, toks):
        """Rows 0..j of one branch: hrows [j+1, H] (backbone rows 1..j+1), toks [j+1] embedding rows E[t_i] [j+1, H].
        Returns per row: x, h after each layer, y, z (fp32)."""
        T, d = self.T, self.d
        x = T["pos_embed"][:len(toks)] + self.mm("in", hrows) + self.mm("emb", self.rms(toks, T["emb_norm.weight"]))
        out = {"x": x.copy()}
        R = x.shape[0]
        for l in range(self.NL):
            p = f"layers.{l}."
            n = self.rms(x, T[p + "input_layernorm.weight"])
            qkv = self.mm(p + "qkv", n)
            q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
            o = np.zeros((R, d), np.float32)
            for h in range(self.nh):
                sl = slice(h * self.hd, (h + 1) * self.hd)
                s = (q[:, sl].astype(np.float64) @ k[:, sl].T.astype(np.float64)) / np.sqrt(self.hd)
                s = np.where(np.tril(np.ones((R, R), bool)), s, -np.inf)
                s = np.exp(s - s.max(-1, keepdims=True))
                s /= s.sum(-1, keepdims=True)
                o[:, sl] = (s @ v[:, sl].astype(np.float64)).astype(np.float32)
            x = x + self.mm(p + "o", o)
            n2 = self.rms(x, T[p + "post_attention_layernorm.weight"])
            g, u = self.mm(p + "gate", n2), self.mm(p + "up", n2)
            hh = (g.astype(np.float64) / (1.0 + np.exp(-g.astype(np.float64))) * u).astype(np.float32)
            x = x + self.mm(p + "down", hh)
            out[f"h{l}"] = x.copy()
        y = self.rms(x, T["norm.weight"])
        out["y"], out["z"] = y, self.mm("out", y)
        return out

    def score(self, z, y, U, erows_out, j, qtemp, acc_row):
        eo = erows_out.astype(np.float64)
        S = float(self.lscale[j]) * U.astype(np.float64) + eo @ z.astype(np.float64)
        m = S.max()
        e = np.exp(S - m)
        ls = S - m - np.log(e.sum())
        q = np.exp((S - m) / qtemp)
        q /= q.sum()
        if self.wy is not None:
            r = acc_row
            inv = 1.0 / np.sqrt((eo * eo).mean(-1) + 1e-6)
            A = (float(self.wy[r] @ y.astype(np.float64)) + inv * (eo @ self.we[r].astype(np.float64))
                 + float(self.wq[r]) * ls + float(self.wb[r]))
            a = 1.0 / (1.0 + np.exp(-A))
        else:
            a = np.ones_like(S)
        return S, q, a, ls


def pcg_hash(v):
    state = (v * 747796405 + 2891336453) & 0xFFFFFFFF
    word = (((state >> ((state >> 28) + 4)) ^ state) * 277803737) & 0xFFFFFFFF
    return ((word >> 22) ^ word) & 0xFFFFFFFF


def dfl_uniform(seed, pos, salt):
    return float(pcg_hash((seed ^ pcg_hash(pos) ^ salt) & 0xFFFFFFFF) & 0xFFFFFF) / 16777216.0


SALT_WALK = 0x57A1C3E5


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.sqrt((b * b).mean()), 1e-30))


def cmd_check(a):
    dec = Decoder(a.ckpt, mimic=not a.no_mimic)
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    fails = 0
    # 1. the engine's quantized weights against this quantizer, code for code
    wdir = os.path.join(a.dump, "weights")
    if os.path.isdir(wdir):
        bad = 0
        tot = 0
        for name, (codes, S, B) in dec.q.items():
            fn = os.path.join(wdir, name)
            W = np.load(fn + ".w.npy")
            ec = unpack4(W, codes.shape[1])
            es, eb = np.load(fn + ".s.npy"), np.load(fn + ".b.npy")
            nb = int((ec != codes).sum() + (es != S).sum() + (eb != B).sum())
            bad += nb
            tot += codes.size
            if nb:
                print(f"  weight {name}: {nb} codes / scales / biases differ")
        ok = bad == 0
        print(f"[quant] engine 4-bit weights vs mx.quantize rounding here: {tot / 1e6:.1f}M codes, {bad} differ: "
              f"{'PASS' if ok else 'FAIL'}")
        fails += not ok
    for case in meta["cases"]:
        cd = os.path.join(a.dump, case["name"])
        h = np.load(os.path.join(cd, "h.npy"))            # [8, H] backbone rows
        ci = np.load(os.path.join(cd, "cand_i.npy"))      # [7, 32]
        cv = np.load(os.path.join(cd, "cand_v.npy"))      # [7, 32]
        ids = np.load(os.path.join(cd, "row_ids.npy"))    # token ids of the gathered rows
        erow = np.load(os.path.join(cd, "erows.npy"))     # [M, H] target embedding rows (fp32)
        orow = np.load(os.path.join(cd, "orows.npy"))     # [M, H] target LM-head rows (fp32)
        where = {int(t): i for i, t in enumerate(ids)}
        tr = {k: np.load(os.path.join(cd, f"tree_{k}.npy")) for k in
              ("parent", "token", "depth", "expanded", "ch_tok", "ch_s", "ch_q", "ch_a", "cand_k", "logp", "acc")}
        nn = int(case["n_nodes"])
        qtemp = float(case["qtemp"])
        cf = os.path.join(cd, "chain.npy")
        chain = np.load(cf) if os.path.exists(cf) else None

        def on_chain(n):  # n's root path spells the drafted chain
            toks = []
            while int(tr["parent"][n]) >= 0:
                toks.append(int(tr["token"][n]))
                n = int(tr["parent"][n])
            return chain is not None and toks[::-1] == [int(t) for t in chain[:len(toks)]]
        es, eq, ea, nexp = 0.0, 0.0, 0.0, 0
        dbg_done = False
        choice_bad, near = 0, 0
        for n in range(nn):
            if not tr["expanded"][n]:
                continue
            nexp += 1
            path = []
            m = n
            while m >= 0:
                path.append(m)
                m = int(tr["parent"][m])
            path = path[::-1]
            j = len(path) - 1
            toks = [int(tr["token"][m]) for m in path]
            out = dec.branch(h[1:j + 2], erow[[where[t] for t in toks]])
            if not dbg_done and j == 0 and os.path.exists(os.path.join(cd, "dbg_x.npy")):
                slot = int(case.get("dbg_slot", 0))
                msg = []
                for k in ("x", "h0", "h1", "y", "z"):
                    fn = os.path.join(cd, f"dbg_{k}.npy")
                    if os.path.exists(fn) and k in out:
                        msg.append(f"{k} {rel(np.load(fn)[slot], out[k][-1]):.1e}")
                print(f"  [{case['name']}] root row intermediates, max |d| / rms: " + ", ".join(msg))
                dbg_done = True
            cand = ci[j]
            S, q, aa, ls = dec.score(out["z"][-1], out["y"][-1], cv[j], orow[[where[int(c)] for c in cand]], j, qtemp,
                                     int(case.get("acc_row", 0)))
            if (tr["ch_tok"][n] != cand).any():
                print(f"  node {n}: children ids differ from the backbone's candidates of row {j + 1}")
                fails += 1
            es = max(es, rel(tr["ch_s"][n], S))
            eq = max(eq, float(np.abs(tr["ch_q"][n] - q).max()))
            ea = max(ea, float(np.abs(tr["ch_a"][n] - aa).max()))
            # the walk's choice at a node of the drafted chain (chain cases; with speculative passes the tree also
            # holds guessed nodes the walk did not take)
            kids = [c for c in range(nn) if int(tr["parent"][c]) == n and int(tr["token"][c]) == int(chain[j])] \
                if chain is not None and on_chain(n) else []
            if case["kind"] == "chain" and kids:
                k_eng = int(tr["cand_k"][kids[0]])
                if case["temp"] == 0:
                    order = np.argsort(-S, kind="stable")
                    k_ref = int(order[0])
                    gap = S[order[0]] - S[order[1]]
                else:
                    u = dfl_uniform(int(case["seed"]), int(case["P"]) + 1 + j, SALT_WALK)
                    cum = np.cumsum(q)
                    k_ref = int(np.argmax(cum > u)) if (cum > u).any() else int(np.nonzero(q > 0)[0][-1])
                    gap = float(np.abs(cum - u).min())
                if k_ref != k_eng:
                    if gap < 1e-4 * max(1.0, abs(float(S.max()))):
                        near += 1
                    else:
                        choice_bad += 1
                        print(f"  node {n} (depth {j}): engine child k={k_eng}, reference k={k_ref} (margin {gap:.3g})")
        ok = es < 2e-3 and eq < 2e-3 and ea < 2e-3 and choice_bad == 0
        extra = ""
        if case["kind"] == "chain":
            extra = f", walk choices: {choice_bad} differ" + (f" ({near} at near-ties)" if near else "")
        print(f"[{case['name']}] {nexp} expanded nodes: S max |d| / rms(S) {es:.2e}, q max |d| {eq:.2e}, "
              f"acceptance max |d| {ea:.2e}{extra}: {'PASS' if ok else 'FAIL'}")
        fails += not ok
    print("ALL PASSED" if not fails else f"FAILED: {fails}")
    return 1 if fails else 0


def main():
    np.seterr(all="ignore")  # Accelerate's float64 matmul raises spurious divide / overflow flags on macOS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("init")
    pi.add_argument("out")
    pi.add_argument("--hidden", type=int, required=True)
    pi.add_argument("--vocab", type=int, default=248320)
    pi.add_argument("--seed", type=int, default=0)
    pi.add_argument("--zero-out", action="store_true")
    pi.add_argument("--backbone", default="dflash2")
    pc = sub.add_parser("check")
    pc.add_argument("ckpt")
    pc.add_argument("dump")
    pc.add_argument("--no-mimic", action="store_true")
    a = ap.parse_args()
    if a.cmd == "init":
        cmd_init(a)
        return 0
    return cmd_check(a)


if __name__ == "__main__":
    sys.exit(main())

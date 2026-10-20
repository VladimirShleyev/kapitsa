"""KFlash candidate decoder (internal/KFLASH.md §1) in MLX: the training model and the reference forward for
the Metal implementation (M3). Checkpoint format: tools/kflash/README.md.

Per drafting pass the backbone (DFlash / DFlash2) gives, for block rows r = 1..7: h_r (its final normed hidden, the draft
head's input), C_r (top-32 ids of the draft head) and U_r (their raw draft logits). A tree node at depth j (the root =
the anchor, depth 0) is expanded by one decoder row j over its branch root..node:

    x_j = W_h h_{j+1} + W_e (RMSNorm(E[tok_j]) * emb_norm) + pos[j]       tok_0 = the anchor, tok_j = the node's token
    2 causal pre-norm layers (MHA 8 x 128, no RoPE, SwiGLU), y_j = RMSNorm(x, norm)
    z_j = W_o y_j
    S_j[k] = logit_scale[j] * U_{j+1}[k] + <z_j, E_out[C_{j+1}[k]]>           the scores of the node's children
    A_j[k, r] = <w_y[r], y_j> + <w_e[r], E_out[c] / rms(E_out[c])> + w_q[r] * log_softmax(S_j)[k] + b[r]
                r = 0: P(c = target argmax | path) (greedy), r = 1: min(1, p_target(c) / q(c)) at T = 1 (sampled)

E is the target's token embedding table and E_out its LM head (tied on the 4B), both 4-bit, dequantized rows, frozen.
"""
import json, math, os, struct

import mlx.core as mx
import mlx.nn as nn
import numpy as np

EPS = 1e-6


class Config:
    def __init__(self, **kw):
        self.hidden_size = 1024
        self.num_hidden_layers = 2
        self.num_attention_heads = 8
        self.head_dim = 128
        self.intermediate_size = 2816
        self.target_hidden_size = 2560
        self.vocab_size = 248320
        self.num_candidates = 32
        self.max_positions = 16
        self.rms_norm_eps = EPS
        self.block_size = 8
        self.__dict__.update(kw)

    def to_json(self):
        return dict(self.__dict__)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, self.nh, self.hd = c.hidden_size, c.num_attention_heads, c.head_dim
        self.q_proj = nn.Linear(d, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(d, self.nh * self.hd, bias=False)
        self.v_proj = nn.Linear(d, self.nh * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, d, bias=False)

    def __call__(self, x):
        B, L, _ = x.shape
        q, k, v = (p(x).reshape(B, L, self.nh, self.hd).transpose(0, 2, 1, 3) for p in (self.q_proj, self.k_proj, self.v_proj))
        # explicit causal softmax over the <= 8 rows of a branch (MLX's fused kernel has no gradient at this length)
        s = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(self.hd))
        causal = mx.triu(mx.full((L, L), -1e30, dtype=s.dtype), k=1)
        o = mx.softmax(s + causal, axis=-1) @ v
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.self_attn = Attention(c)
        self.post_attention_layernorm = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.mlp = MLP(c)

    def __call__(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Decoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        d, H = c.hidden_size, c.target_hidden_size
        self.in_proj = nn.Linear(H, d, bias=False)
        self.emb_norm = nn.RMSNorm(H, eps=c.rms_norm_eps)
        self.emb_proj = nn.Linear(H, d, bias=False)
        self.pos_embed = mx.zeros((c.max_positions, d))
        self.layers = [Layer(c) for _ in range(c.num_hidden_layers)]
        self.norm = nn.RMSNorm(d, eps=c.rms_norm_eps)
        self.out_proj = nn.Linear(d, H, bias=False)
        self.logit_scale = mx.ones((c.max_positions,))
        self.acc_head = nn.Linear(d + H + 1, 2)

    def init_weights(self, seed=0):
        """GPT-style: normal(0, 0.02) projections, residual outputs scaled by 1 / sqrt(2 layers), W_o = 0 (the decoder
        starts as the backbone: S = U), the acceptance head at zero with a prior bias."""
        mx.random.seed(seed)
        std = 0.02
        res = std / math.sqrt(2 * self.c.num_hidden_layers)
        for name, lin in [("in_proj", self.in_proj), ("emb_proj", self.emb_proj)]:
            lin.weight = mx.random.normal(lin.weight.shape) * std
        self.pos_embed = mx.random.normal(self.pos_embed.shape) * std
        for L in self.layers:
            for p in (L.self_attn.q_proj, L.self_attn.k_proj, L.self_attn.v_proj, L.mlp.gate_proj, L.mlp.up_proj):
                p.weight = mx.random.normal(p.weight.shape) * std
            for p in (L.self_attn.o_proj, L.mlp.down_proj):
                p.weight = mx.random.normal(p.weight.shape) * res
        self.out_proj.weight = mx.zeros(self.out_proj.weight.shape)
        self.acc_head.weight = mx.zeros(self.acc_head.weight.shape)
        self.acc_head.bias = mx.array([-1.0, -1.0])

    def trunk(self, h, tok_emb):
        """h [B, L, H] (rows 1..L of the block), tok_emb [B, L, H] (E rows of tok_0..tok_{L-1}) -> y [B, L, d], z [B, L, H]"""
        L = h.shape[1]
        x = self.in_proj(h) + self.emb_proj(self.emb_norm(tok_emb)) + self.pos_embed[:L]
        for layer in self.layers:
            x = layer(x)
        y = self.norm(x)
        return y, self.out_proj(y)

    def scores(self, z, u, cand_emb, L0=0):
        """S [B, L, K] = logit_scale[j] * U + <z, E_out[C]>; cand_emb [B, L, K, H]"""
        L = z.shape[1]
        corr = (cand_emb @ z[..., None])[..., 0]
        return self.logit_scale[L0:L0 + L][None, :, None] * u + corr

    def accept_logits(self, y, cand_emb, logq):
        """A [B, L, K, 2] (logits) from y [B, L, d], the candidates' E_out rows and log_softmax(S) [B, L, K]"""
        d = y.shape[-1]
        W = self.acc_head.weight  # [2, d + H + 1]
        wy, we, wq = W[:, :d], W[:, d:-1], W[:, -1]
        inv = mx.rsqrt(mx.mean(cand_emb.astype(mx.float32) ** 2, axis=-1, keepdims=True) + EPS)  # [B, L, K, 1]
        te = (cand_emb @ we.T) * inv                                                           # [B, L, K, 2]
        return (y @ wy.T)[:, :, None, :] + te + logq[..., None] * wq + self.acc_head.bias

    def __call__(self, h, tok_emb, u, cand_emb, cand_mask=None):
        """Teacher-forced (or any one branch): S [B, L, K], A logits [B, L, K, 2], y [B, L, d]."""
        y, z = self.trunk(h, tok_emb)
        S = self.scores(z, u, cand_emb)
        Sm = S if cand_mask is None else mx.where(cand_mask, S, -1e30)
        logq = Sm - mx.logsumexp(Sm, axis=-1, keepdims=True)
        A = self.accept_logits(y, cand_emb, mx.stop_gradient(logq))
        return S, A, y


# ---------------------------------------------------------------- the target's embedding table (4-bit, frozen)
def _st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _st_tensor(path, name, hdr=None):
    if hdr is None:
        hdr = _st_header(path)
    meta, base = hdr
    m = meta[name]
    a, b = m["data_offsets"]
    dt = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16, "F32": np.float32}[m["dtype"]]
    arr = np.fromfile(path, dtype=dt, count=(b - a) // np.dtype(dt).itemsize, offset=base + a).reshape(m["shape"])
    x = mx.array(arr)
    return x.view(mx.bfloat16) if m["dtype"] == "BF16" else x


class TargetEmbeddings:
    """Rows of the target's embedding table E and LM head E_out as plain fp32 vectors, whatever the target's
    precision: MLX affine-quantized tables (any bits / group size) are dequantized on the GPU, unquantized ones (bf16,
    f16, f32) read as stored. Tied models (the 4B) share one table. The decoder sees only the rows."""

    def __init__(self, model_dir):
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.H = cfg.get("text_config", cfg).get("hidden_size", cfg.get("hidden_size"))
        tied = cfg.get("tie_word_embeddings", cfg.get("text_config", {}).get("tie_word_embeddings", False))
        ip = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(ip):
            idx = json.load(open(ip))["weight_map"]
        else:  # a single-file checkpoint
            idx = {k: "model.safetensors" for k in _st_header(os.path.join(model_dir, "model.safetensors"))[0]}
        pre = next(k for k in idx if k.endswith("embed_tokens.weight"))[: -len("embed_tokens.weight")]
        self.E = self._load(model_dir, idx, pre + "embed_tokens")
        if tied:
            self.E_out = self.E
        else:
            head = next(k[: -len(".weight")] for k in idx if k.endswith("lm_head.weight"))
            self.E_out = self._load(model_dir, idx, head)

    def _load(self, model_dir, idx, base):
        if f"{base}.scales" not in idx:  # unquantized
            return (_st_tensor(os.path.join(model_dir, idx[f"{base}.weight"]), f"{base}.weight"),)
        w, s, b = (_st_tensor(os.path.join(model_dir, idx[f"{base}.{p}"]), f"{base}.{p}")
                   for p in ("weight", "scales", "biases"))
        bits, gs = w.shape[1] * 32 // self.H, self.H // s.shape[1]  # this table's own format (mixed-precision models)
        return (w, s, b, bits, gs)

    def rows(self, table, ids):
        flat = ids.reshape(-1)
        if len(table) == 1:
            r = table[0][flat].astype(mx.float32)
        else:
            w, s, b, bits, gs = table
            r = mx.dequantize(w[flat], s[flat], b[flat], group_size=gs, bits=bits).astype(mx.float32)
        return r.reshape(*ids.shape, -1)

    def emb(self, ids):
        return self.rows(self.E, ids)

    def out(self, ids):
        return self.rows(self.E_out, ids)


# ---------------------------------------------------------------- checkpoint
FP32_NAMES = ("logit_scale", "acc_head.weight", "acc_head.bias")


def save_checkpoint(model, out_dir, meta=None):
    """<out_dir>/config.json + model.safetensors (bf16; logit_scale and acc_head fp32)"""
    from mlx.utils import tree_flatten
    os.makedirs(out_dir, exist_ok=True)
    w = {}
    for k, v in tree_flatten(model.parameters()):
        w[k] = v.astype(mx.float32 if k in FP32_NAMES else mx.bfloat16)
    mx.save_safetensors(os.path.join(out_dir, "model.safetensors"), w, metadata={"format": "kflash-decoder-v1"})
    cfg = model.c.to_json()
    cfg.update({"architectures": ["KFlashCandidateDecoder"], "format_version": 1, "emb_norm": True,
                "logit_scale": True, "acc_rows": ["greedy", "sampled"]})
    if meta:
        cfg.update(meta)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)


def load_checkpoint(path):
    cfg = json.load(open(os.path.join(path, "config.json")))
    keep = {k: v for k, v in cfg.items() if k in Config().__dict__}
    m = Decoder(Config(**keep))
    w = mx.load(os.path.join(path, "model.safetensors"))
    m.load_weights([(k, v.astype(mx.float32)) for k, v in w.items()])
    return m, cfg

#!/usr/bin/env python3
"""Offline evaluation of KFlash candidate decoders on held-out sequences (internal/KFLASH.md M2).

The generator drafts at every answer position of each held-out sequence (`kflash_datagen data --anchors all`), so a
greedy decode can be replayed exactly: from the first answer position, a cycle at anchor P accepts the longest drafted
path whose tokens equal the target's argmax (known along the target's greedy path, which is the sequence itself), and
the next cycle starts at P + accepted + 1. Reported per method and domain: tokens per cycle (accepted + 1, the verify's
bonus token) over the replayed cycles; for the chains also the mean accepted length over all anchors.

Methods (V = verify rows: the anchor + V - 1 drafted nodes; today's chain of 7 is V = 8):
  v1-chain        each block row's top-1 (independent marginals): what Kapitsa runs today on the DFlash v1 backbone
  v1-tree-V       best-first tree over the backbone's marginals softmax(U_r) (path probability = their product)
  dec-chain       the decoder's greedy chain (argmax S along its own path)
  dec-tree-V      best-first over the decoder's path probability (product of q = softmax(S))
  acc-tree-V      best-first over the acceptance head's greedy row (product of sigmoid(A_0))
  v1-temp-tree-V  v1 trees with M1's per-depth temperature + coverage (dfl_tree_learn), learned online on past cycles
  v1-kf-tree-V    v1 trees with M1's shared per-depth logistic calibration (KfBudget), learned online on past cycles
  dec-kf-tree-V   the decoder's q with the same logistic calibration
Calibrations start warm: before the first group is measured they learn from a replay of it.
--q4 adds the same for each decoder quantized as the engine runs it (*_proj at 4-bit group 64). Trees are built only at
the anchors a replayed decode visits (sequences advance in lockstep, a batch of trees per cycle).

usage: eval.py --seqs heldout_seqs.jsonl --ckpt DIR [--ckpt DIR2 ...] [--q4] [--trees 8,16,32] [--out results.json]
GPU work holds ~/.cache/nace-gpu.lock in chunks of at most --chunk-min minutes (the generator stopped and the lock
released for --pause-s between them, as train.py does); NI_LOCK_HELD=1: the caller holds it.
"""
import argparse, collections, heapq, json, os, subprocess, sys, time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as dm  # noqa: E402
import stream as ds  # noqa: E402
from train import GpuLock  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
K, NPOS = 32, 7


# ---------------------------------------------------------------- the decoder on arbitrary paths
class Runner:
    def __init__(self, model, te):
        self.m, self.te = model, te

    def last_row(self, h, toks, depth, u, cand):
        """decoder outputs at the last row of a batch of paths: h [B, 7, H] f16 (the anchors' block rows 1..7), toks
        [B, Lmax] path tokens (t_0 = the anchor), depth [B] (= path length - 1), u / cand [B, 7, K]. Returns log q [B, K]
        and acceptance logits [B, K, 2] of the node's children (the candidates of block row depth + 1)."""
        Lm = toks.shape[1]
        y, z = self.m.trunk(mx.array(h[:, :Lm].astype(np.float32)), self.te.emb(mx.array(toks)))
        d = mx.array(depth)
        yj = mx.take_along_axis(y, d[:, None, None], axis=1)[:, 0]
        zj = mx.take_along_axis(z, d[:, None, None], axis=1)[:, 0]
        cj = mx.array(np.take_along_axis(cand, depth[:, None, None], 1)[:, 0])
        uj = mx.array(np.take_along_axis(u, depth[:, None, None], 1)[:, 0])
        ce = self.te.out(cj)
        S = self.m.logit_scale[d][:, None] * uj + (ce @ zj[..., None])[..., 0]
        logq = S - mx.logsumexp(S, axis=-1, keepdims=True)
        A = self.m.accept_logits(yj[:, None], ce[:, None], logq[:, None])[:, 0]
        mx.eval(logq, A)
        return np.array(logq), np.array(A)

    def teacher(self, exs):
        """teacher-forced S and A over the 32 candidates of every row (the decoder's greedy chain is exact this way)"""
        h = mx.array(np.stack([e["h"] for e in exs]).astype(np.float32))
        tin = mx.array(np.stack([e["tok_in"] for e in exs]))
        u = mx.array(np.stack([e["u"] for e in exs]))
        cand = mx.array(np.stack([e["cand"] for e in exs]))
        S, A, _ = self.m(h, self.te.emb(tin), u, self.te.out(cand))
        mx.eval(S, A)
        return np.array(S), np.array(A)


def chain_len(ok, valid):
    return int(np.cumprod(ok & valid).sum())


def accept_tree(tree, tgt, valid):
    """greedy acceptance of a tree: from the root follow the child that carries the target's argmax. Returns the
    accepted length and the nodes on the accepted path (the root first)."""
    children = collections.defaultdict(dict)
    for n in range(1, len(tree["tok"])):
        children[tree["par"][n]][tree["tok"][n]] = n
    node, a, path = 0, 0, [0]
    while a < NPOS and valid[a] and tgt[a] in children[node]:
        node = children[node][tgt[a]]
        path.append(node)
        a += 1
    return a, path


def best_first(exs, rows, expand, value=None):
    """best-first trees of `rows` verify rows (root + rows - 1 nodes) for a list of anchors. expand(list of (i, path))
    gives the raw log values of each path's children (the candidates of block row len(path)); value(depth, raw) maps
    them to the log values the priority uses (a calibration; default: raw). Returns per anchor a tree: tok, par, dep,
    raw (each node's own raw log value) and kids (expanded node -> its children's raw vector)."""
    n = len(exs)
    trees = [{"tok": [int(exs[i]["tok_in"][0])], "par": [-1], "dep": [0], "raw": [0.0], "val": [0.0], "kids": {}}
             for i in range(n)]
    heaps = [[] for _ in range(n)]

    def push(i, node, raw):
        t = trees[i]
        d = t["dep"][node]
        if d >= NPOS:
            return
        t["kids"][node] = raw
        vals = raw if value is None else value(d + 1, raw)
        keep = min(K, rows - 1)  # a node never gets more children than the budget
        top = np.argpartition(-vals, keep - 1)[:keep] if keep < K else np.arange(K)
        for k in top:
            heapq.heappush(heaps[i], (-(t["val"][node] + float(vals[k])), node, int(k)))

    def path(i, node):
        t = trees[i]
        p = []
        while node >= 0:
            p.append(t["tok"][node])
            node = t["par"][node]
        return p[::-1]

    for i, v in enumerate(expand([(i, path(i, 0)) for i in range(n)])):
        push(i, 0, v)
    for r in range(rows - 1):
        todo = []
        for i in range(n):
            if not heaps[i]:
                continue
            negv, par_node, k = heapq.heappop(heaps[i])
            t = trees[i]
            j = t["dep"][par_node]
            t["tok"].append(int(exs[i]["cand"][j][k]))
            t["par"].append(par_node)
            t["dep"].append(j + 1)
            t["raw"].append(float(t["kids"][par_node][k]))
            t["val"].append(-negv)
            if j + 1 < NPOS and r < rows - 2:
                todo.append((i, len(t["tok"]) - 1))
        if todo:
            for (i, node), v in zip(todo, expand([(i, path(i, node)) for i, node in todo])):
                push(i, node, v)
    return trees


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


class KfCalib:
    """M1's shared KFlash calibration (KfBudget, src/model/kflash_budget.cpp on M1's branch; tools/tree_sim.py
    KfCalib): P(the target accepts a node of depth d with drafter probability q) = sigmoid(alpha_d + beta_d logit(q)),
    an L2-regularized logistic fit toward (0, 1) per depth (Newton steps on the last 2048 records, every 16 cycles once a
    depth holds 32), from every cycle's tree nodes whose parent is on the accepted path; before any records p = q."""
    WIN, EVERY, MIN_REC, ITERS = 2048, 16, 32, 4

    def __init__(self):
        self.alpha, self.beta = np.zeros(8), np.ones(8)
        self.x, self.y = [collections.deque(maxlen=self.WIN) for _ in range(8)], [collections.deque(maxlen=self.WIN) for _ in range(8)]
        self.since = 0

    @staticmethod
    def logit_q(lq):
        q = np.minimum(np.exp(lq), 0.999999)
        return lq - np.log1p(-q)

    def value(self, d, raw):
        z = self.alpha[d] + self.beta[d] * self.logit_q(np.asarray(raw, np.float64))
        return -np.logaddexp(0, -z)  # log sigmoid

    @staticmethod
    def fit(x, y, a, b, iters):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(a + b * x)))
            w, r = p * (1.0 - p), p - y
            ga, gb = a + r.sum(), (b - 1.0) + (r * x).sum()
            haa, hab, hbb = 1.0 + w.sum(), (w * x).sum(), 1.0 + (w * x * x).sum()
            det = haa * hbb - hab * hab
            if not det > 1e-12:
                break
            da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
            a, b = float(np.clip(a - da, -8.0, 8.0)), float(np.clip(b - db, 0.1, 4.0))
            if abs(da) + abs(db) < 1e-4:
                break
        return a, b

    def learn(self, cycles):
        for tree, onpath, _tgt, _rows in cycles:
            on = set(onpath)
            for n in range(1, len(tree["tok"])):
                if tree["par"][n] in on:
                    d = tree["dep"][n]
                    self.x[d].append(float(self.logit_q(tree["raw"][n])))
                    self.y[d].append(1.0 if n in on else 0.0)
            self.since += 1
            if self.since >= self.EVERY:
                self.since = 0
                for d in range(1, 8):
                    if len(self.x[d]) >= self.MIN_REC:
                        self.alpha[d], self.beta[d] = self.fit(self.x[d], self.y[d], self.alpha[d], self.beta[d], self.ITERS)


class TempCalib:
    """M1's per-depth temperature and coverage (dfl_tree_learn; tools/tree_sim.py OnlineCalib): per children row p a
    log-likelihood over tau = 2^((g - 8) / 4), g = 0..16 (0.25 .. 4) and the coverage (how often the target's token is
    among the candidates), learned from the rows on each cycle's accepted path, exponential forgetting over 256 samples;
    node probability = coverage_p * softmax(scores / tau_p)[k], tau_p = argmax of the log-likelihood minus
    0.5 (log2 tau)^2; before any data tau 1 and coverage 1."""
    G = 17

    def __init__(self, window=256.0, prior=0.5):
        self.lam, self.prior = 1.0 - 1.0 / window, prior
        self.taus = np.array([2.0 ** ((g - 8) / 4.0) for g in range(self.G)])
        self.ll, self.hit, self.tot = np.zeros((NPOS, self.G)), np.zeros(NPOS), np.zeros(NPOS)

    def value(self, d, raw):
        p = d - 1
        tau = self.taus[int(np.argmax(self.ll[p] - self.prior * np.log2(self.taus) ** 2))]
        return np.log((self.hit[p] + 1.0) / (self.tot[p] + 1.0)) + log_softmax(np.asarray(raw, np.float64) / tau)

    def learn(self, cycles):
        for tree, onpath, tgt, rows in cycles:
            for j in range(min(len(onpath) - 1, NPOS - 1) + 1):  # the anchor's row and each accepted node's
                if tgt[j] < 0:  # past the sequence's end
                    continue
                raw, cand = rows(j)
                self.ll[j] *= self.lam
                self.hit[j] *= self.lam
                self.tot[j] *= self.lam
                self.tot[j] += 1.0
                k = np.nonzero(cand == tgt[j])[0]
                if len(k) == 0:
                    continue
                self.hit[j] += 1.0
                for g, t in enumerate(self.taus):
                    self.ll[j][g] += log_softmax(np.asarray(raw, np.float64) / t)[k[0]]


def tree_accepts(exs, runner, kind, rows, calib=None, learn_only=False):
    """accepted lengths of one best-first tree per anchor: kind v1 (backbone marginals softmax(U)), dec (decoder q) or
    acc (acceptance head); calib: an online calibration of the node values (KfCalib / TempCalib), taught by these
    cycles' outcomes afterwards"""
    tgt = [e["tgt_ids"][:, 0] for e in exs]
    valid = [e["valid"] for e in exs]
    if kind == "v1":
        lu = [log_softmax(e["u"].astype(np.float64)) for e in exs]

        def expand(items):
            return [lu[i][len(p) - 1] for i, p in items]
    else:
        h = np.stack([e["h"] for e in exs])
        u = np.stack([e["u"] for e in exs])
        cand = np.stack([e["cand"] for e in exs])

        def expand(items):
            out = []
            for b in range(0, len(items), 1024):
                part = items[b:b + 1024]
                L = max(len(p) for _, p in part)
                toks = np.zeros((len(part), L), np.int32)
                for r, (_, p) in enumerate(part):
                    toks[r, :len(p)] = p
                depth = np.array([len(p) - 1 for _, p in part], np.int64)
                idx = [i for i, _ in part]
                logq, A = runner.last_row(h[idx], toks, depth, u[idx], cand[idx])
                vals = logq if kind == "dec" else -np.logaddexp(0, -A[..., 0])  # log sigmoid
                out.extend(vals)
            return out
    trees = best_first(exs, rows, expand, calib.value if calib else None)
    res = [accept_tree(t, tgt[i], valid[i]) for i, t in enumerate(trees)]
    if calib is not None:
        cyc = []
        for i, (t, (a, onpath)) in enumerate(zip(trees, res)):
            # the rows TempCalib learns from: row j's raw children values (v1: the backbone's marginals) and candidates
            def rows_fn(j, i=i, t=t, onpath=onpath):
                return (lu[i][j] if kind == "v1" else t["kids"].get(onpath[j], np.zeros(K))), exs[i]["cand"][j]
            cyc.append((t, onpath, tgt[i], rows_fn))
        calib.learn(cyc)
    return [a for a, _ in res]


# ---------------------------------------------------------------- per group of sequences
class Totals:
    def __init__(self):
        self.t = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0, 0]))
        self.cal = collections.defaultdict(list)

    def add(self, method, dom, tokens, cycles, acc_sum=0, anchors=0):
        for key in (dom, "all"):
            b = self.t[method][key]
            b[0] += tokens
            b[1] += cycles
            b[2] += acc_sum
            b[3] += anchors

    def table(self):
        return {m: {k: {"tok_per_cycle": v[0] / max(v[1], 1), "cycles": v[1],
                        **({"mean_accepted": v[2] / v[3], "anchors": v[3]} if v[3] else {})} for k, v in d.items()}
                for m, d in self.t.items()}


def replay(group, accept_fn):
    """lockstep greedy replay over a group of sequences; accept_fn(list of examples) -> accepted lengths.
    Returns per sequence (tokens, cycles)."""
    cur = {s: g["n_prompt"] for s, g in group.items()}
    res = {s: [0, 0] for s in group}
    while True:
        act = [s for s in group if cur[s] in group[s]["E"]]
        if not act:
            return res
        a = accept_fn([group[s]["E"][cur[s]] for s in act])
        for s, x in zip(act, a):
            res[s][0] += x + 1
            res[s][1] += 1
            cur[s] += x + 1


CALIBS = {}  # method -> its online calibration, kept across groups (as the engine keeps it across requests)


def eval_group(group, runners, tree_rows, tot, meta):
    """every method on a group of complete sequences: {seq: {"n_prompt", "id", "E": {P: example}}}. Calibrated tree
    methods learn online across the whole replay; before the first group is measured, they learn from a replay of it
    (a warm calibration, as the engine's persisted one is; in-sample for that group, which can only favor them)."""
    dom = {s: meta.get(g["id"], {}).get("domain", g["id"].split("-")[0]) for s, g in group.items()}
    allx = [(s, P, e) for s, g in group.items() for P, e in g["E"].items()]
    exs = [e for _, _, e in allx]
    tgt = np.stack([e["tgt_ids"][:, 0] for e in exs])
    valid = np.stack([e["valid"] for e in exs])
    cand = np.stack([e["cand"] for e in exs])
    chains = {"v1-chain": cand[..., 0] == tgt}
    for name, rn in runners.items():
        SA = [rn.teacher(exs[b:b + 256]) for b in range(0, len(exs), 256)]
        S = np.concatenate([x[0] for x in SA])
        A = np.concatenate([x[1] for x in SA])
        top = S.argmax(-1)
        pick = np.take_along_axis(cand, top[..., None], -1)[..., 0]
        chains[f"{name}:dec-chain"] = pick == tgt
        # calibration of the greedy acceptance row at the chain's pick, on rows the chain reaches
        pa = 1 / (1 + np.exp(-np.take_along_axis(A[..., 0], top[..., None], -1)[..., 0]))
        hit = (pick == tgt) & valid
        reach = np.concatenate([np.ones((len(exs), 1), bool), np.cumprod(hit[:, :-1], 1).astype(bool)], 1) & valid
        for i, j in zip(*np.nonzero(reach)):
            tot.cal[name].append((int(j), float(pa[i, j]), bool(hit[i, j])))
    for method, ok in chains.items():
        a = np.cumprod(ok & valid, 1).sum(1)
        acc = {(s, P): int(x) for (s, P, _), x in zip(allx, a)}
        for s, (tk, cy) in replay(group, lambda xs, acc=acc: [acc[(x["seq"], x["P"])] for x in xs]).items():
            mine = [acc[(s, P)] for P in group[s]["E"]]
            tot.add(method, dom[s], tk, cy, sum(mine), len(mine))
    for V in tree_rows:
        kinds = [("v1-tree", None, "v1", None), ("v1-temp-tree", None, "v1", TempCalib), ("v1-kf-tree", None, "v1", KfCalib)]
        for n, rn in runners.items():
            kinds.append((f"{n}:dec-tree", rn, "dec", None))
            if not n.endswith("-q4"):
                kinds += [(f"{n}:acc-tree", rn, "acc", None), (f"{n}:dec-kf-tree", rn, "dec", KfCalib)]
        for label, rn, kind, cal in kinds:
            name = f"{label}-{V}"
            c = None
            if cal is not None:
                if name not in CALIBS:
                    CALIBS[name] = cal()
                    replay(group, lambda xs, rn=rn, kind=kind, c=CALIBS[name]: tree_accepts(xs, rn, kind, V, c))
                c = CALIBS[name]
            for s, (tk, cy) in replay(group, lambda xs, rn=rn, kind=kind, c=c: tree_accepts(xs, rn, kind, V, c)).items():
                tot.add(name, dom[s], tk, cy)


def calibration(cal):
    """mean predicted vs actual acceptance per block row, and the expected calibration error over 10 bins"""
    if not cal:
        return {}
    a = np.array([c[1] for c in cal])
    y = np.array([c[2] for c in cal], float)
    j = np.array([c[0] for c in cal])
    rows = {int(r): {"pred": float(a[j == r].mean()), "actual": float(y[j == r].mean()), "n": int((j == r).sum())}
            for r in range(NPOS) if (j == r).any()}
    bins = np.minimum((a * 10).astype(int), 9)
    ece = sum(abs(a[bins == b].mean() - y[bins == b].mean()) * (bins == b).sum() for b in range(10) if (bins == b).any())
    return {"rows": rows, "ece": float(ece / len(a)), "n": int(len(a))}


def quantize_q4(model):
    """the engine's decoder numerics: *_proj weights as MLX affine 4-bit group 64 (mx.quantize rounding)"""
    nn.quantize(model, group_size=64, bits=4, class_predicate=lambda p, m: isinstance(m, nn.Linear) and p.endswith("_proj"))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--model", default=os.path.expanduser("~/.cache/kapitsa/models/qwen3.5-4b-4bit"))
    ap.add_argument("--bin", default=os.path.join(REPO, "build/kflash_datagen"))
    ap.add_argument("--trees", default="8,16,32")
    ap.add_argument("--q4", action="store_true")
    ap.add_argument("--group", type=int, default=16, help="sequences replayed together")
    ap.add_argument("--max-answer", type=int, default=1024)
    ap.add_argument("--out", default="")
    ap.add_argument("--stream", default="", help="an existing stream of --seqs (anchors all) instead of running the generator")
    ap.add_argument("--chunk-min", type=float, default=18.0)
    ap.add_argument("--pause-s", type=int, default=30)
    a = ap.parse_args()
    tree_rows = [int(x) for x in a.trees.split(",") if x]
    lines = [l for l in open(a.seqs) if l.strip()]
    meta = {}
    for l in lines:
        r = json.loads(l)
        meta[r["id"]] = r.get("meta", {})
    lock = None
    if not os.environ.get("NI_LOCK_HELD") and not a.stream:
        lock = GpuLock(a.chunk_min, a.pause_s, None)
        lock.take()
    te = dm.TargetEmbeddings(a.model)
    runners = {}
    for c in a.ckpt:
        name = os.path.basename(os.path.normpath(c))
        runners[name] = Runner(dm.load_checkpoint(c)[0], te)
        if a.q4:
            runners[name + "-q4"] = Runner(quantize_q4(dm.load_checkpoint(c)[0]), te)
    t0 = time.time()
    if a.stream:
        p = None
        it = ds.read_stream(open(a.stream, "rb"))
    else:
        cmd = ["env", "NI_ANE=0", a.bin, "data", a.model, "--seqs", a.seqs, "--anchors", "all", "--max-answer",
               str(a.max_answer)]
        kw = dict(stdout=subprocess.PIPE, stderr=open(os.devnull, "w"))
        p = lock.spawn(cmd, **kw) if lock else subprocess.Popen(cmd, **kw)
        it = ds.read_stream(p.stdout)
    next(it)  # header
    tot = Totals()
    group, cur, nseq, nanc = {}, {}, 0, 0
    for kind, rec in it:
        if kind == "seq":
            cur[rec["seq"]] = {"rec": rec, "E": {}}
        elif kind == "anchor":
            c = cur[rec["seq"]]
            c["E"][rec["P"]] = ds.example(c["rec"], rec)
            nanc += 1
        elif kind == "end":
            c = cur.pop(rec["seq"])
            group[rec["seq"]] = {"n_prompt": c["rec"]["n_prompt"], "id": c["rec"]["id"], "E": c["E"]}
            nseq += 1
            if len(group) >= a.group:
                eval_group(group, runners, tree_rows, tot, meta)
                group = {}
                print(time.strftime("%H:%M:%S"), f"{nseq} sequences, {nanc} anchors evaluated", flush=True)
                if lock:
                    lock.tick()
    if group:
        eval_group(group, runners, tree_rows, tot, meta)
    if p is not None:
        p.wait()
    if lock is not None:
        lock.release()
    table = tot.table()
    doms = sorted({k for t in table.values() for k in t}, key=lambda d: (d == "all", d))
    print(f"\n{nseq} held-out sequences, {nanc} anchors, {time.time() - t0:.0f} s. Tokens per cycle (accepted + 1):")
    print(f"{'method':30s} " + " ".join(f"{d:>10s}" for d in doms))
    for meth in sorted(table):
        t = table[meth]
        print(f"{meth:30s} " + " ".join(f"{t[d]['tok_per_cycle']:10.3f}" if d in t else " " * 10 for d in doms))
    print("\nmean accepted per anchor (chains, every answer position):")
    for meth in sorted(table):
        if "mean_accepted" in table[meth]["all"]:
            print(f"{meth:30s} {table[meth]['all']['mean_accepted']:.3f}")
    cal = {name: calibration(c) for name, c in tot.cal.items()}
    for name, c in cal.items():
        if c:
            print(f"\n{name}: acceptance head (greedy row) at the chain's pick, ECE {c['ece']:.4f} over {c['n']} rows")
            for r, v in c["rows"].items():
                print(f"  block row {r + 1}: predicted {v['pred']:.3f}, actual {v['actual']:.3f} ({v['n']})")
    if a.out:
        json.dump({"table": table, "calibration": cal, "sequences": nseq, "anchors": nanc}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

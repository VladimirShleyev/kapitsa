#!/usr/bin/env python3
"""Trains the KFlash candidate decoder (internal/KFLASH.md §1, §3) on the stream `kflash_datagen data`
writes, with the backbone and the target frozen (they are only in the stream). MLX.

Data: the answers of gen_answers.py, split by meta.split. Train sequences stream from the generator into a replay
buffer, one generator process per job (jobs_for): the agent sequences (5-20k-token prompts, ~200-token answers: their
prefill dominates) once at every answer position, weighted x --agent-weight; the others --epochs times with --anchors
fresh anchors each. Every buffered anchor is used `--reuse` times on average. Held-out sequences give a fixed validation
set (32 anchors each).

Loss, teacher-forced along the target's path, per block position j (row j + 1), positions after the first one whose
path token is not among the backbone's 32 candidates masked (no tree reaches them):
  LK loss (lambda KL + (1 - lambda) TV, lambda = exp(-2 stopgrad(1 - TV))) between the target's distribution and the
  decoder's q = softmax(S) over the candidates plus the path token when it is outside them (its draft logit from the
  stream), + 0.1 x NLL of the target's argmax, + 0.5 x BCE of the two acceptance rows (32 candidates).

GPU: the trainer holds ~/.cache/nace-gpu.lock for at most --chunk-min minutes at a time (the generator is stopped with
SIGSTOP and the lock released for --pause-s between chunks). Checkpoints and a log go to --out.

usage: train.py --answers answers.jsonl --out DIR [--epochs 3] [--batch 256] [--lr 3e-4] [--reuse 3] [--resume]
"""
import argparse, fcntl, json, os, queue, random, signal, subprocess, sys, threading, time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as dm  # noqa: E402
import stream as ds  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
LOCK = os.path.expanduser("~/.cache/nace-gpu.lock")
K, KX, NPOS = 32, 33, 7  # candidates, + the path token's slot, block positions


def log(*a, f=None):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    if f:
        f.write(s + "\n")
        f.flush()


# ---------------------------------------------------------------- GPU lock in chunks
class GpuLock:
    def __init__(self, chunk_min, pause_s, logf):
        self.chunk_s, self.pause_s, self.logf = chunk_min * 60, pause_s, logf
        self.fd, self.t0, self.procs = None, 0, []
        self.mu = threading.Lock()  # held while a generator starts and while the lock changes hands

    def take(self):
        t = time.time()
        fd = os.open(LOCK, os.O_RDONLY | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
        with self.mu:
            self.fd = fd
            self.t0 = time.time()
            for p in self.procs:
                if p.poll() is None:
                    p.send_signal(signal.SIGCONT)
        if self.t0 - t > 5:
            log(f"GPU lock after {self.t0 - t:.0f} s", f=self.logf)

    def release(self):
        with self.mu:
            for p in self.procs:
                if p.poll() is None:
                    p.send_signal(signal.SIGSTOP)
            time.sleep(0.2)  # a stopped generator's last command buffer drains
            mx.synchronize()
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None

    def spawn(self, cmd, **kw):
        """a generator process, started only while the lock is held (stopped right away otherwise)"""
        with self.mu:
            p = subprocess.Popen(cmd, **kw)
            self.procs = [p]
            if self.fd is None:
                p.send_signal(signal.SIGSTOP)
            return p

    def tick(self):
        """between steps: give the GPU away once the chunk is used up"""
        if time.time() - self.t0 > self.chunk_s:
            self.release()
            log(f"GPU lock released for {self.pause_s} s", f=self.logf)
            time.sleep(self.pause_s)
            self.take()


# ---------------------------------------------------------------- data
AGENT = ("agent", "agent_syn")


def split_answers(path, out_dir):
    """train sequences split into the agent ones (long prompts: drafted at every answer position, once) and the rest
    (fresh anchors every epoch); held-out sequences for validation and eval.py"""
    tr_a, tr_r, ho = [], [], []
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        if len(r.get("answer", [])) < 2:
            continue
        l = l if l.endswith("\n") else l + "\n"
        m = r.get("meta", {})
        if m.get("split") == "heldout":
            ho.append(l)
        else:
            (tr_a if m.get("domain") in AGENT else tr_r).append(l)
    out = {}
    for name, ls in (("train_agent", tr_a), ("train_rest", tr_r), ("heldout", ho)):
        out[name] = os.path.join(out_dir, name + "_seqs.jsonl")
        open(out[name], "w").writelines(ls)
    return out, len(tr_a), len(tr_r), len(ho)


def anchors_of(path, anchors, max_answer):
    n = 0
    for l in open(path):
        m = min(len(json.loads(l)["answer"]), max_answer) - 1
        n += m if anchors == "all" else min(int(anchors), m)
    return n


def jobs_for(a, files):
    """generator runs: the agent sequences once at every answer position (weight a.agent_weight), the rest with
    a.anchors fresh anchors per epoch; shuffled, the agent runs spread over the training"""
    rest = [(files["train_rest"], str(a.anchors), 1000 * ep + s + 1, f"{s}/{a.shards}", 1.0, f"rest ep{ep} {s}")
            for ep in range(a.epochs) for s in range(a.shards)]
    random.Random(a.seed).shuffle(rest)
    agent = [(files["train_agent"], "all", 7, f"{s}/{a.agent_shards}", a.agent_weight, f"agent {s}")
             for s in range(a.agent_shards)]
    jobs, step = [], max(1, len(rest) // max(1, len(agent)))
    for i, j in enumerate(rest):
        if i % step == 0 and agent:
            jobs.append(agent.pop(0))
        jobs.append(j)
    return jobs + agent


def gen_cmd(a, seqs, anchors, seed, shard):  # noqa: D103
    return ["env", "NI_ANE=0", a.bin, "data", a.model, "--seqs", seqs, "--anchors", str(anchors), "--seed", str(seed),
            "--shard", shard, "--max-answer", str(a.max_answer)]


def read_all(cmd, logf):
    """runs a generator to completion (the caller holds the lock) and returns its examples"""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=open(os.path.join(os.path.dirname(logf.name), "gen.log"), "a"))
    hdr, ex = ds.examples(p.stdout)
    out = list(ex)
    if p.wait() != 0:
        raise SystemExit(f"generator failed: {cmd}")
    return hdr, out


class Producer(threading.Thread):
    """the generator runs (jobs_for) into a queue of (example, weight), blocking when it is full"""

    def __init__(self, a, jobs, lock, q, logf):
        super().__init__(daemon=True)
        self.a, self.jobs, self.lock, self.q, self.logf = a, jobs, lock, q, logf
        self.done = False
        self.error = None
        self.job = ""

    def run(self):
        try:
            a = self.a
            for seqs, anchors, seed, shard, weight, name in self.jobs[a.start_job:]:
                self.job = name
                cmd = gen_cmd(a, seqs, anchors, seed, shard)
                p = self.lock.spawn(cmd, stdout=subprocess.PIPE, stderr=open(os.path.join(a.out, "gen.log"), "a"))
                hdr, ex = ds.examples(p.stdout)
                n = 0
                for e in ex:
                    self.q.put((e, weight))
                    n += 1
                rc = p.wait()
                self.lock.procs = []
                if rc != 0:
                    raise RuntimeError(f"generator exited with {rc}: {' '.join(cmd)}")
                log(f"job {name}: {n} anchors", f=self.logf)
            self.done = True
        except Exception as e:  # noqa: BLE001
            self.error = e
            self.done = True


# ---------------------------------------------------------------- batches
def make_batch(exs):
    """numpy arrays for a list of examples (stream.example dicts)"""
    B = len(exs)
    h = np.stack([e["h"] for e in exs])                       # [B,7,H] f16
    cand = np.stack([e["cand"] for e in exs])                 # [B,7,32]
    u = np.stack([e["u"] for e in exs])
    u_next = np.stack([e["u_next"] for e in exs])             # [B,7]
    tok_in = np.stack([e["tok_in"] for e in exs])
    tok_next = np.stack([e["tok_next"] for e in exs])
    valid = np.stack([e["valid"] for e in exs])
    tgt_ids = np.stack([e["tgt_ids"] for e in exs])           # [B,7,64]
    tgt_p = np.stack([e["tgt_p"] for e in exs])
    in_cand = (cand == tok_next[..., None]).any(-1)           # path token among the 32
    extra = valid & ~in_cand & np.isfinite(u_next)
    ce = np.concatenate([cand, np.where(extra, tok_next, 0)[..., None]], -1)       # [B,7,33]
    ue = np.concatenate([u, np.where(extra, u_next, 0)[..., None]], -1).astype(np.float32)
    cmask = np.concatenate([np.ones_like(cand, bool), extra[..., None]], -1)
    eq = ce[..., :, None] == tgt_ids[..., None, :]                                   # [B,7,33,64]
    pt = (eq * tgt_p[..., None, :]).sum(-1).astype(np.float32) * cmask
    gi = np.where(cmask & (ce == tgt_ids[..., :1]), np.arange(KX), KX).min(-1)       # argmax slot, KX if none
    reach = np.concatenate([np.ones((B, 1), bool), np.cumprod(in_cand[:, :-1], 1).astype(bool)], 1)
    pmask = valid & reach
    yg = (ce == tgt_ids[..., :1]) & cmask
    return {"h": h, "ce": ce.astype(np.int32), "ue": ue, "cmask": cmask, "pt": pt, "gi": gi.astype(np.int32),
            "pmask": pmask, "yg": yg, "tok_in": tok_in.astype(np.int32), "cand": cand}


def to_mx(b):
    return {k: mx.array(v) for k, v in b.items()}


def loss_fn(model, te, b, w_greedy=0.1, w_bce=0.5):
    h = b["h"].astype(mx.float32)
    S, A, y = model(h, te.emb(b["tok_in"]), b["ue"], te.out(b["ce"]), cand_mask=b["cmask"])
    cm = b["cmask"]
    Sm = mx.where(cm, S, -1e30)
    logq = Sm - mx.logsumexp(Sm, axis=-1, keepdims=True)
    q = mx.where(cm, mx.exp(logq), 0.0)
    pt = b["pt"]
    ps = pt.sum(-1)
    tv = 0.5 * (mx.abs(pt - q).sum(-1) + (1.0 - ps))
    phat = pt / mx.maximum(ps, 1e-9)[..., None]
    kl = mx.where(phat > 0, phat * (mx.log(mx.maximum(phat, 1e-30)) - logq), 0.0).sum(-1)
    lam = mx.exp(-2.0 * mx.stop_gradient(1.0 - tv))
    lk = mx.where(ps > 1e-6, lam * kl + (1 - lam) * tv, tv)
    pm = b["pmask"].astype(mx.float32)
    npos = mx.maximum(pm.sum(), 1.0)
    gi = b["gi"]
    has_g = (gi < KX).astype(mx.float32) * pm
    lg = -mx.take_along_axis(logq, mx.minimum(gi, KX - 1)[..., None], axis=-1)[..., 0]
    # acceptance rows over the 32 real candidates of unmasked positions
    cm32 = cm[..., :K].astype(mx.float32) * pm[..., None]
    ys = mx.minimum(1.0, pt[..., :K] / mx.maximum(mx.stop_gradient(q[..., :K]), 1e-9))
    bce = nn.losses.binary_cross_entropy(A[..., :K, 0], b["yg"][..., :K].astype(mx.float32), with_logits=True,
                                         reduction="none") + \
        nn.losses.binary_cross_entropy(A[..., :K, 1], ys, with_logits=True, reduction="none")
    l_lk = (lk * pm).sum() / npos
    l_g = (lg * has_g).sum() / mx.maximum(has_g.sum(), 1.0)
    l_b = (bce * cm32).sum() / mx.maximum(cm32.sum(), 1.0)
    loss = l_lk + w_greedy * l_g + w_bce * l_b
    top1 = (mx.argmax(Sm, -1) == gi).astype(mx.float32)
    aux = {"lk": l_lk, "greedy": l_g, "bce": l_b, "acc_tv": ((1 - tv) * pm).sum() / npos,
           "top1": (top1 * has_g).sum() / mx.maximum(has_g.sum(), 1.0),
           "top1_v1": ((gi == 0).astype(mx.float32) * has_g).sum() / mx.maximum(has_g.sum(), 1.0)}
    return loss, aux


# ---------------------------------------------------------------- validation (teacher-forced, chain acceptance)
def chain_lengths(argmax_ok, valid):
    """accepted length of a chain per anchor: leading positions whose pick equals the target's argmax"""
    ok = argmax_ok & valid
    return np.cumprod(ok, 1).sum(1)


def validate(model, te, val, bs=256):
    sums = {}
    a_v1, a_dec = [], []
    for i in range(0, len(val), bs):
        nb = make_batch(val[i:i + bs])
        b = to_mx(nb)
        loss, aux = loss_fn(model, te, b)
        S, _, _ = model(b["h"].astype(mx.float32), te.emb(b["tok_in"]), b["ue"], te.out(b["ce"]), cand_mask=b["cmask"])
        S32 = np.array(S[..., :K])
        n = len(val[i:i + bs])
        for k, v in dict(aux, loss=loss).items():
            sums[k] = sums.get(k, 0.0) + float(v.item()) * n
        tgt = np.stack([e["tgt_ids"][:, 0] for e in val[i:i + bs]])
        valid = np.stack([e["valid"] for e in val[i:i + bs]])
        pick = np.take_along_axis(nb["cand"], S32.argmax(-1)[..., None], -1)[..., 0]
        a_dec.append(chain_lengths(pick == tgt, valid))
        a_v1.append(chain_lengths(nb["cand"][..., 0] == tgt, valid))
    out = {k: v / len(val) for k, v in sums.items()}
    out["chain_v1"] = float(np.concatenate(a_v1).mean())
    out["chain_dec"] = float(np.concatenate(a_dec).mean())
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.expanduser("~/.cache/kapitsa/models/qwen3.5-4b-4bit"))
    ap.add_argument("--bin", default=os.path.join(REPO, "build/kflash_datagen"))
    ap.add_argument("--epochs", type=int, default=3, help="passes over the non-agent sequences (fresh anchors each)")
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--anchors", type=int, default=128)
    ap.add_argument("--agent-shards", type=int, default=4)
    ap.add_argument("--agent-weight", type=float, default=2.0)
    ap.add_argument("--max-answer", type=int, default=768)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--reuse", type=float, default=3.0)
    ap.add_argument("--buffer", type=int, default=40000)
    ap.add_argument("--min-buffer", type=int, default=6000)
    ap.add_argument("--chunk-min", type=float, default=18.0)
    ap.add_argument("--pause-s", type=int, default=30)
    ap.add_argument("--val-every", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-opt", action="store_true", help="also write the optimizer state with the resume checkpoint")
    ap.add_argument("--start-job", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "train.log"), "a")
    log("args", json.dumps(vars(a)), f=logf)
    files, n_ag, n_rest, n_ho = split_answers(a.answers, a.out)
    jobs = jobs_for(a, files)
    n_agent = anchors_of(files["train_agent"], "all", a.max_answer)
    n_rest_ep = anchors_of(files["train_rest"], a.anchors, a.max_answer)
    weighted = n_agent * a.agent_weight + n_rest_ep * a.epochs
    total_steps = int(weighted * a.reuse / a.batch)
    log(f"train: {n_ag} agent sequences ({n_agent} anchors, weight {a.agent_weight}), {n_rest} others ({n_rest_ep} "
        f"anchors per epoch x {a.epochs}); {n_ho} held-out; {len(jobs)} generator runs; {total_steps} steps", f=logf)

    te = dm.TargetEmbeddings(a.model)
    model = dm.Decoder(dm.Config())
    model.init_weights(a.seed)
    sched = optim.join_schedules([optim.linear_schedule(1e-7, a.lr, a.warmup),
                                  optim.cosine_decay(a.lr, max(1, total_steps - a.warmup), a.lr * 0.1)], [a.warmup])
    opt = optim.AdamW(learning_rate=sched, betas=[0.9, 0.98], weight_decay=a.wd)
    step = 0
    ck = os.path.join(a.out, "ckpt")
    if a.resume and os.path.exists(os.path.join(ck, "model.safetensors")):
        model, _ = dm.load_checkpoint(ck)
        st = json.load(open(os.path.join(ck, "state.json")))
        step = st["step"]
        if os.path.exists(os.path.join(ck, "opt.safetensors")):
            from mlx.utils import tree_unflatten
            opt.init(model.trainable_parameters())
            opt.state = tree_unflatten(list(mx.load(os.path.join(ck, "opt.safetensors")).items()))
        log(f"resumed at step {step}", f=logf)
    mx.eval(model.parameters())

    lock = GpuLock(a.chunk_min, a.pause_s, logf)
    lock.take()
    # held-out validation set
    t = time.time()
    _, val = read_all(gen_cmd(a, files["heldout"], 32, 4242, "0/1"), logf)
    log(f"validation: {len(val)} anchors in {time.time() - t:.0f} s", f=logf)

    q = queue.Queue(maxsize=4096)
    prod = Producer(a, jobs, lock, q, logf)
    prod.start()

    def train_step(b):
        def lf(m):
            return loss_fn(m, te, b)
        (loss, aux), grads = nn.value_and_grad(model, lf)(model)
        grads, gnorm = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss, aux, gnorm

    buf = []
    received = int(step * a.batch / a.reuse)  # (a resumed run's budget)
    rng = random.Random(a.seed + step)
    t_last, hist = time.time(), []
    best, best_w, best_meta = None, None, None
    while True:
        # take what arrived (a random replacement once the buffer is full)
        try:
            while True:
                e, w = q.get_nowait()
                copies = int(w) + (rng.random() < w - int(w))
                for _ in range(copies):
                    received += 1
                    if len(buf) < a.buffer:
                        buf.append(e)
                    else:
                        buf[rng.randrange(a.buffer)] = e
        except queue.Empty:
            pass
        if prod.error:
            raise prod.error
        finished = prod.done and q.empty()
        # step budget: every received anchor is used `reuse` times on average
        if (step + 1) * a.batch > a.reuse * received or len(buf) < a.min_buffer:
            if finished:
                break
            time.sleep(0.05)
            lock.tick()
            continue
        b = to_mx(make_batch(rng.sample(buf, a.batch)))
        loss, aux, gnorm = train_step(b)
        mx.eval(model.parameters(), opt.state, loss)
        step += 1
        hist.append({k: float(v.item()) for k, v in dict(aux, loss=loss, gnorm=gnorm).items()})
        if step % 50 == 0:
            m = {k: np.mean([h[k] for h in hist]) for k in hist[0]}
            dt = time.time() - t_last
            log(f"step {step}/{total_steps} [{prod.job}] lr {float(sched(step)):.2e} " +
                " ".join(f"{k} {v:.4f}" for k, v in m.items()) +
                f" | {50 / dt:.2f} step/s, received {received}, buffer {len(buf)}", f=logf)
            hist, t_last = [], time.time()
        if step % a.val_every == 0:
            v = validate(model, te, val)
            log("VAL step %d " % step + " ".join(f"{k} {x:.4f}" for k, x in v.items()), f=logf)
            save(model, opt, ck, step, v, a.save_opt, logf)
            if best is None or v["chain_dec"] > best:
                best = v["chain_dec"]
                best_w = [(k, mx.array(x)) for k, x in tree_flatten(model.parameters())]  # kept in memory too
                mx.eval([x for _, x in best_w])
                best_meta = {"step": step, "val": v}
                try_save(lambda: dm.save_checkpoint(model, os.path.join(a.out, "best"), best_meta), "best", logf)
        lock.tick()
    v = validate(model, te, val)
    log("VAL final step %d " % step + " ".join(f"{k} {x:.4f}" for k, x in v.items()), f=logf)
    save(model, opt, ck, step, v, a.save_opt, logf)
    try_save(lambda: dm.save_checkpoint(model, os.path.join(a.out, "final"), {"step": step, "val": v}), "final", logf)
    if best is None or v["chain_dec"] > best:
        try_save(lambda: dm.save_checkpoint(model, os.path.join(a.out, "best"), {"step": step, "val": v}), "best", logf)
    elif best_w is not None:  # (again: a save that failed for lack of disk space)
        model.load_weights(best_w)
        try_save(lambda: dm.save_checkpoint(model, os.path.join(a.out, "best"), best_meta), "best", logf)
    lock.release()


def try_save(fn, what, logf):
    """a checkpoint write that may fail on a full disk (shared machine): logged, training goes on"""
    try:
        fn()
        return True
    except (OSError, RuntimeError, ValueError) as e:  # MLX reports write errors as RuntimeError / ValueError
        log(f"WARNING: could not write the {what} checkpoint: {e}", f=logf)
        return False


def save(model, opt, ck, step, v, save_opt, logf):
    if not try_save(lambda: dm.save_checkpoint(model, ck, {"step": step, "val": v}), "resume", logf):
        return
    if save_opt:  # 2 x 4 bytes per parameter (270 MB): only when asked
        try_save(lambda: mx.save_safetensors(os.path.join(ck, "opt.safetensors"), dict(tree_flatten(opt.state))), "optimizer", logf)
    try_save(lambda: json.dump({"step": step, "val": v}, open(os.path.join(ck, "state.json"), "w")), "state", logf)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A tree's size and shape must not change a greedy reply.

Runs build/ni_run on one prompt with KFlash's tree forced to different sizes (NI_KFLASH_FORCE: n rows every cycle,
-1 every count in turn, cycle by cycle) and compares the generated ids. Every run decodes the same path of tokens
through trees of other sizes and shapes (the path's rows sit in other slots, jump or follow in sequence, are joined by
other rows), so the ids must be equal; at the first token that differs, a row's logits depended on the tree.

  tools/kflash_invariance.py <model dir> [prompt ids file, default bench/prompts/code_35b.txt] [n_gen=192]
      [--sizes 1,3,5,8,-1] [--env NAME=VALUE ...] [--bin build/ni_run]

The switches that undo the invariance fixes of the 35B's tree verify (pass them with --env to bisect a regression;
each is on by default): NI_V8_TREE_ATTN=0 (the tree block's keys back inside the attention's MMA tiles),
NI_V8_TREE_PIN=0 (the GDN step's update back to the compiler's contraction), NI_DFLASH_EXACT_MOE=0 (the experts'
numerics follow how many rows share an expert).
"""
import argparse
import os
import subprocess
import sys


def run(binary, model, prompt, n_gen, force, extra):
    env = dict(os.environ)
    env["NI_KFLASH_FORCE"] = str(force)
    env.update(extra)
    cmd = [binary, model, "@" + prompt, str(n_gen), "0", "--dflash", "auto"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-2000:])
        sys.exit("ni_run failed (force %s)" % force)
    for line in r.stdout.splitlines():
        if line.startswith("ids:"):
            return [int(x) for x in line[4:].split()]
    sys.exit("no ids line in ni_run's output (force %s)" % force)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prompt", nargs="?", default="bench/prompts/code_35b.txt")
    ap.add_argument("n_gen", nargs="?", type=int, default=192)
    ap.add_argument("--sizes", default="1,3,5,8,-1")
    ap.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--bin", default="build/ni_run")
    a = ap.parse_args()
    extra = dict(kv.split("=", 1) for kv in a.env)
    sizes = [int(x) for x in a.sizes.split(",")]
    outs = {}
    for n in sizes:
        outs[n] = run(a.bin, a.model, a.prompt, a.n_gen, n, extra)
        print("force %2d: %d tokens" % (n, len(outs[n])))
    ref_n = sizes[0]
    ref = outs[ref_n]
    bad = 0
    for n in sizes[1:]:
        o = outs[n]
        k = next((i for i in range(min(len(ref), len(o))) if ref[i] != o[i]), None)
        if k is None and len(ref) == len(o):
            print("force %2d == force %2d" % (n, ref_n))
        else:
            bad += 1
            k = min(len(ref), len(o)) if k is None else k
            print("force %2d != force %2d: first difference at generated token %d (%s vs %s)" %
                  (n, ref_n, k, ref[k] if k < len(ref) else "end", o[k] if k < len(o) else "end"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

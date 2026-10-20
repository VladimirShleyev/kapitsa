#!/usr/bin/env python3
"""The target's own answers to a request set (prompts.py): runs `kflash_datagen gen` under ~/.cache/nace-gpu.lock
in chunks of at most --chunk-s seconds (the lock is released between chunks so other sessions can measure), until
every request has an answer in --out (appended; requests already answered there are skipped). Stdlib only.

usage: gen_answers.py --requests requests.jsonl --out answers.jsonl [--model DIR] [--chunk-s 1050] [--batch 8]
"""
import argparse, json, os, random, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
LOCK = os.path.expanduser("~/.cache/nace-gpu.lock")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.expanduser("~/.cache/kapitsa/models/qwen3.5-4b-4bit"))
    ap.add_argument("--bin", default=os.path.join(REPO, "build/kflash_datagen"))
    ap.add_argument("--chunk-s", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-pos", type=int, default=98304)
    ap.add_argument("--pause-s", type=int, default=20, help="lock-free gap between chunks")
    a = ap.parse_args()
    reqs = [json.loads(l) for l in open(a.requests) if l.strip()]
    random.Random(1).shuffle(reqs)  # long agent prompts spread among short ones
    while True:
        done = set()
        if os.path.exists(a.out):
            for l in open(a.out):
                try:
                    done.add(json.loads(l)["id"])
                except ValueError:
                    pass  # a line cut short by an interrupted run
        left = [r for r in reqs if r["id"] not in done]
        print(time.strftime("%H:%M:%S"), f"{len(done)} answered, {len(left)} left", flush=True)
        if not left:
            return
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for r in left:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp = f.name
        env = dict(os.environ, NI_ANE="0")
        cmd = ["lockf", "-k", "-t", "7200", LOCK, a.bin, "gen", a.model, "--requests", tmp, "--batch", str(a.batch),
               "--max-pos", str(a.max_pos), "--time-limit", str(a.chunk_s), "--grace", "180"]
        with open(a.out, "a") as out:
            rc = subprocess.call(cmd, stdout=out, env=env)
        os.unlink(tmp)
        if rc != 0:
            sys.exit(f"kflash_datagen gen exited with {rc}")
        time.sleep(a.pause_s)


if __name__ == "__main__":
    main()

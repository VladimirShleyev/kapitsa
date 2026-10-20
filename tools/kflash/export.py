#!/usr/bin/env python3
"""Exports a trained candidate decoder to <model>/kflash/{config.json, model.safetensors} (format: README.md), next
to the drafter it was trained on (<model>/dflash2). The weights are rewritten from the checkpoint (bf16; logit_scale and
the acceptance head fp32) with the backbone, target and evaluation recorded in config.json.

usage: export.py --ckpt DIR [--model DIR] [--eval results.json] [--out DIR]
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as dm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default=os.path.expanduser("~/.cache/kapitsa/models/qwen3.5-4b-4bit"))
    ap.add_argument("--eval", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(a.model, "kflash")
    m, cfg = dm.load_checkpoint(a.ckpt)
    bb = json.load(open(os.path.join(a.model, "dflash2", "config.json")))
    meta = {
        "backbone": "dflash" if bb.get("architectures") == ["DFlashDraftModel"] else "dflash2",
        "backbone_dir": "dflash2",
        "target": os.path.basename(os.path.normpath(a.model)),
        "training": {k: cfg[k] for k in ("step", "val") if k in cfg},
    }
    if a.eval:
        meta["eval"] = json.load(open(a.eval))
    dm.save_checkpoint(m, out, meta)
    print("exported ->", out)


if __name__ == "__main__":
    main()

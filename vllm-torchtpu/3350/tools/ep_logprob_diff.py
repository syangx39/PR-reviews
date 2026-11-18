#!/usr/bin/env python3
"""Quantify switch-phase differences between two arms: which rows differ, whether token ids match, and how much the
chosen-token log-probs move.

    python3 ep_logprob_diff.py RESULT_DIR --ref base_off_e1 --arm final_on_e1

Loads <label>.json(.gz) written by mskip_engine.py (switch phase). Standard library only.
"""
import argparse
import gzip
import json
import os


def load(result_dir, label):
    for name in (f"{label}.json.gz", f"{label}.json"):
        path = os.path.join(result_dir, name)
        if os.path.exists(path):
            opener = gzip.open if name.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as fh:
                return json.load(fh)
    raise FileNotFoundError(f"{label} not found in {result_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arm", required=True)
    args = ap.parse_args()
    ref, arm = load(args.result_dir, args.ref), load(args.result_dir, args.arm)
    print(f"{args.arm} vs {args.ref}: ep={arm['meta'].get('ep')}, sequence {arm['meta'].get('switch_seq')}")
    n_rows = n_diff = 0
    for br, ba in zip(ref["switch"], arm["switch"]):
        for ri, (a, b) in enumerate(zip(br["rows"], ba["rows"])):
            n_rows += 1
            if a["token_ids"] == b["token_ids"] and a["logprobs"] == b["logprobs"]:
                continue
            n_diff += 1
            same_tokens = a["token_ids"] == b["token_ids"]
            diffs = [(i, abs(x - y)) for i, (x, y) in enumerate(zip(a["logprobs"] or [], b["logprobs"] or [])) if x != y]
            first = diffs[0][0] if diffs else None
            worst = max((d for _, d in diffs), default=0.0)
            print(f"  batch {br['i']} (n={br['n']}) row {ri}: token ids identical={same_tokens}, length {len(a['token_ids'])}, "
                  f"chosen-token log-prob differs at {len(diffs)} positions from token {first}, max |diff| {worst:.4g}")
    print(f"  {n_diff} of {n_rows} rows differ")


if __name__ == "__main__":
    main()

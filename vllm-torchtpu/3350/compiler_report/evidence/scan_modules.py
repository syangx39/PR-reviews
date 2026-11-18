#!/usr/bin/env python3
"""Map the dumped forward-module files (t4 g0 HLO dumps, already extracted locally) to their bucket size.

For every *.tt_jit_ir_ops_L89C0_fused_add_rms_norm_to_copy_copy_*.{before,after}_optimizations.txt file of each arm:
module number, kind, number of ENTRY instructions, number of router_topk custom-calls in ENTRY and the row count of
their result (the token bucket), plus file size and sha256. Read-only; prints a TSV.
"""
import glob
import hashlib
import os
import re
import sys

ROOT = sys.argv[1]
ARMS = sys.argv[2].split(",")
TOPK_RE = re.compile(r"=\s*\(?\s*f32\[(\d+),8\]")


def scan(path):
    entry = False
    n_entry = 0
    rows = set()
    n_topk = 0
    bal = 0
    with open(path, errors="replace") as f:
        for line in f:
            if not entry:
                if line.startswith("ENTRY"):
                    entry = True
                continue
            if line.strip() == "}" and bal == 0:
                break
            if re.match(r"^\s+(ROOT\s+)?%", line):
                n_entry += 1
                if "router_topk" in line and "custom-call(" in line:
                    n_topk += 1
                    m = TOPK_RE.search(line)
                    if m:
                        rows.add(int(m.group(1)))
    return n_entry, n_topk, sorted(rows)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


print("arm\tmodule\tkind\tentry_instrs\trouter_topk\trows\tbytes\tsha256\tfile")
for arm in ARMS:
    d = os.path.join(ROOT, arm, arm)
    for p in sorted(glob.glob(os.path.join(d, "module_*.tt_jit_ir_ops_L89C0_fused_add_rms_norm_to_copy_copy_.*"))):
        b = os.path.basename(p)
        if b.endswith(".before_optimizations.txt"):
            kind = "before_optimizations"
        elif b.endswith(".after_optimizations.txt"):
            kind = "after_optimizations"
        else:
            continue
        mod = b.split(".")[0]
        n_entry, n_topk, rows = scan(p)
        print(f"{arm}\t{mod}\t{kind}\t{n_entry}\t{n_topk}\t{rows}\t{os.path.getsize(p)}\t{sha256(p)}\t{b}", flush=True)

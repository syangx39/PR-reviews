#!/usr/bin/env python3
"""Numbers for the overlay-wait checks (t5 / t6 / t7: pairs of arms traced with --xla_tpu_show_overlay_waits_in_profiler).

    t5_summary.py RAW_DIR OUT.json [--pairs A:B,...] [--ns 16,32]

Default pairs: base_on_w1:final_on_w1,base_on_w2:final_on_w2 (t5); each pair is A (reference) and B on one host.

For each pair and N: overlay-wait events per decode step and their time
(prof_overlay.analyze), the same-host op alignment of final_on against base_on (t4_summary.compare), and, per
op kind, the change of overlay-wait time next to the change of op duration.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t4_summary as t4  # noqa: E402  (module-level code needs argv: ROOT OUT)
from prof_overlay import analyze  # noqa: E402

import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("out")
ap.add_argument("--pairs", default="base_on_w1:final_on_w1,base_on_w2:final_on_w2")
ap.add_argument("--ns", default="16,32")
args = ap.parse_args()
ROOT, OUT = args.root, args.out
t4.ROOT = ROOT
res = []
for pair in args.pairs.split(","):
    la, lb = pair.split(":")
    for n in [int(x) for x in args.ns.split(",")]:
        pa, pb = t4.path(la, n), t4.path(lb, n)
        if not pa or not pb:
            continue
        A = analyze(pa, "/device:TPU:0", "overlay-wait", 2)
        B = analyze(pb, "/device:TPU:0", "overlay-wait", 2)
        cmp = t4.compare(la, lb, n)
        kinds = dict(cmp["top_kinds"])
        per_kind = []
        for k in sorted(set(A["by_op"]) | set(B["by_op"]),
                                        key=lambda k: -abs(B["by_op"].get(k, 0) - A["by_op"].get(k, 0)))[:8]:
            per_kind.append(dict(kind=k, waits_A=round(A["by_op_n"].get(k, 0), 1), waits_B=round(B["by_op_n"].get(k, 0), 1),
                                                      wait_delta_us=round(B["by_op"].get(k, 0) - A["by_op"].get(k, 0), 1),
                                                      dur_delta_us=kinds.get(k)))
        r = dict(A=la, B=lb, n=n, waits_per_step_A=A["wait_count"], waits_per_step_B=B["wait_count"],
                          wait_us_A=round(A["wait_mean_us"], 1), wait_us_B=round(B["wait_mean_us"], 1),
                          wait_delta_us=round(B["wait_mean_us"] - A["wait_mean_us"], 1),
                          aligned_dur_delta_us=cmp["aligned_dur_delta_us"], module_mean_delta_us=cmp["module_mean_delta_us"],
                          module_us_A=cmp["module_us_A"], module_us_B=cmp["module_us_B"], module_pct=cmp["module_pct"],
                          share_of_dur_delta=round((B["wait_mean_us"] - A["wait_mean_us"]) / cmp["aligned_dur_delta_us"], 3),
                          per_kind=per_kind)
        res.append(r)
        print(f"{lb} vs {la} n={n}: overlay waits/step {r['waits_per_step_A']} -> {r['waits_per_step_B']}, "
                    f"{r['wait_us_A']} -> {r['wait_us_B']} us ({r['wait_delta_us']:+.1f}); existing-op duration "
                    f"{r['aligned_dur_delta_us']:+.1f} us (waits = {100 * r['share_of_dur_delta']:.0f}%); forward "
                    f"{r['module_us_A']} -> {r['module_us_B']} us ({r['module_pct']:+.2f}%)")
        for p in per_kind:
            print(f"    waits {p['waits_A']:5.1f} -> {p['waits_B']:5.1f}/step, wait {p['wait_delta_us']:+7.1f} us, "
                        f"duration {p['dur_delta_us'] if p['dur_delta_us'] is not None else 'n/a'}  {p['kind'][:80]}")
json.dump(res, open(OUT, "w"), indent=1)

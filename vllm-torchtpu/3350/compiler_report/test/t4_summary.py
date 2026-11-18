#!/usr/bin/env python3
"""Numbers for the n=16 investigation (t4 device traces), written as JSON + printed tables.

    t4_summary.py T4RAW_DIR OUT.json

T4RAW_DIR holds <label>/n<N>/plugins/profile/<ts>/rank0_*.xplane.pb for base_on_r{1,2}, final_on_r{1,2},
base_off_r{1,2} (rank 0 owns /device:TPU:0). For each host (r1, r2) and N, pairs on the same host are aligned op
by op (prof_align.load): per-step module time, aligned-op duration / idle-gap deltas, the delta of new or
removed ops, and the GMM share. For N=16 and 32 the per-instance durations of the three op kinds with the
largest extra time are classified as slow (> 1.15 x the median instance of A) to show the bimodal pattern.
"""
import difflib
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prof_align import load  # noqa: E402

ROOT = OUT = None  # set by main()
KINDS = ["distributed_all_reduce", "select_reduce_fusion", "fused_conv1d_gdn_batched"]


def path(label, n):
    p = glob.glob(os.path.join(ROOT, label, f"n{n}", "plugins", "profile", "*", "rank0_*.xplane.pb"))
    return p[0] if p else None


cache = {}


def steps(label, n):
    if (label, n) not in cache:
        p = path(label, n)
        cache[(label, n)] = load(p, "/device:TPU:0")[1][2:] if p else None  # skip 2 warm steps
    return cache[(label, n)]


def compare(la, lb, n):
    A, B = steps(la, n), steps(lb, n)
    if not A or not B:
        return None
    m = min(len(A), len(B))
    ps = lambda v: v / 1e6 / m  # noqa: E731  ps summed over m steps -> us per step
    dur, gap, extra, gmm = 0, 0, 0, 0
    by_kind = defaultdict(float)
    for k in range(m):
        sa, sb = A[k]["seq"], B[k]["seq"]
        ka, kb = [o[0] for o in sa], [o[0] for o in sb]
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ka, b=kb, autojunk=False).get_opcodes():
            if tag == "equal":
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    dd = sb[j][1] - sa[i][1]
                    dur += dd
                    gap += sb[j][2] - sa[i][2]
                    by_kind[ka[i]] += dd
                    if ka[i].startswith("gmm_v2"):
                        gmm += dd
            else:
                extra += sum(o[1] + o[2] for o in sb[j1:j2]) - sum(o[1] + o[2] for o in sa[i1:i2])
    med = lambda X, f: statistics.median(f(x) for x in X[:m])  # noqa: E731
    r = dict(
            A=la, B=lb, n=n, steps=m,
            module_us_A=round(med(A, lambda x: x["dur"] / 1e6), 1),
            module_us_B=round(med(B, lambda x: x["dur"] / 1e6), 1),
            module_mean_delta_us=round(statistics.mean(b["dur"] for b in B[:m]) / 1e6 -
                                                                  statistics.mean(a["dur"] for a in A[:m]) / 1e6, 1),
            aligned_dur_delta_us=round(ps(dur), 1), aligned_gap_delta_us=round(ps(gap), 1),
            unaligned_delta_us=round(ps(extra), 1), gmm_delta_us=round(ps(gmm), 1),
            non_gmm_aligned_dur_delta_us=round(ps(dur - gmm), 1),
            top_kinds=[(k, round(ps(v), 1)) for k, v in sorted(by_kind.items(), key=lambda kv: -abs(kv[1]))[:20]],
    )
    r["module_pct"] = round(100 * (r["module_us_B"] / r["module_us_A"] - 1), 2)
    return r


def instances(la, lb, n):
    A, B = steps(la, n), steps(lb, n)
    out = {}
    for kind in KINDS:
        def per(X):
            v = [[o[1] / 1e6 for o in x["seq"] if o[0].startswith(kind)] for x in X]
            c = min(len(p) for p in v)
            return [statistics.median(p[i] for p in v) for i in range(c)], \
                          [max(p[i] for p in v) - min(p[i] for p in v) for i in range(c)]
        a, sa = per(A)
        b, sb = per(B)
        thr = 1.15 * statistics.median(a)
        slow_a = [x for x in a if x > thr]
        slow_b = [x for x in b if x > thr]
        fast = [x for x in a + b if x <= thr]
        out[kind] = dict(
                instances=len(a), threshold_us=round(thr, 2), slow_A=len(slow_a), slow_B=len(slow_b),
                fast_median_us=round(statistics.median(fast), 2) if fast else None,
                slow_median_us=round(statistics.median(slow_a + slow_b), 2) if slow_a + slow_b else None,
                sum_A_us=round(sum(a), 1), sum_B_us=round(sum(b), 1),
                max_step_spread_us=round(max(sa + sb), 2),
                both_slow=sum(1 for x, y in zip(a, b) if x > thr and y > thr),
        )
    return out


def main():
    global ROOT, OUT
    ROOT, OUT = sys.argv[1], sys.argv[2]
    res = {"pairs": [], "instances": {}}
    for host in ("r1", "r2"):
        for n in (16, 32, 8):
            for la, lb in (("base_on", "final_on"), ("base_off", "base_on"), ("base_off", "final_on")):
                r = compare(f"{la}_{host}", f"{lb}_{host}", n)
                if r:
                    res["pairs"].append(r)
                    print(f"{host} n={n:2d} {lb:9s} - {la:9s}: module {r['module_us_A']:7.1f} -> {r['module_us_B']:7.1f} us "
                                f"({r['module_pct']:+.2f}%), aligned dur {r['aligned_dur_delta_us']:+7.1f} (GMM {r['gmm_delta_us']:+7.1f}, "
                                f"rest {r['non_gmm_aligned_dur_delta_us']:+6.1f}), gaps {r['aligned_gap_delta_us']:+6.1f}, "
                                f"new/removed ops {r['unaligned_delta_us']:+5.1f}")
        for n in (16, 32):
            key = f"{host}_n{n}"
            res["instances"][key] = instances(f"base_on_{host}", f"final_on_{host}", n)
            for kind, v in res["instances"][key].items():
                print(f"  {key} {kind:26s} slow instances: base_on {v['slow_A']:2d}/{v['instances']}, final_on "
                            f"{v['slow_B']:2d}/{v['instances']} (both {v['both_slow']}); fast {v['fast_median_us']} us, "
                            f"slow {v['slow_median_us']} us; sum {v['sum_A_us']} -> {v['sum_B_us']} us/step; "
                            f"max spread over steps {v['max_step_spread_us']} us")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()

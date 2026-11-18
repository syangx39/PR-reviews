#!/usr/bin/env python3
"""Align the decode-step op timelines of two profiles (same host) and locate the extra device time.

    prof_align.py A.xplane.pb B.xplane.pb [--device /device:TPU:0] [--top 25]

For each decode step (one event of the forward module = the per-step module with the largest total time) the
XLA Ops events inside the module window are listed in start order; each op gets its duration and the idle gap
before it (time since the previous op ended, clipped at 0). The op-name sequences of A and B (instruction name
without numeric suffixes + result shape) are aligned with difflib (B may have extra ops); for aligned ops the
per-step duration and gap differences are aggregated by op kind and by layer (router_topk count so far).
"""
import argparse
import bisect
import difflib
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prof_mskip import XSpace, op_kind  # noqa: E402


def load(path, device):
    xs = XSpace()
    with open(path, "rb") as f:
        xs.ParseFromString(f.read())
    pl = next(p for p in xs.planes if p.name == device)
    lines = {ln.name: ln for ln in pl.lines}
    name = {k: v.name for k, v in pl.event_metadata.items()}
    mods, ops = lines["XLA Modules"], lines["XLA Ops"]
    mev = [(mods.timestamp_ns * 1000 + e.offset_ps, e.duration_ps, name[e.metadata_id]) for e in mods.events]
    tot = defaultdict(int)
    cnt = defaultdict(int)
    for _, d, n in mev:
        tot[n] += d
        cnt[n] += 1
    # the decode forward: the module with the most device time among those run in (almost) every step; at large n
    # some steps of the profiled batch still mix in prefill chunks and run another program
    anchor = max((n for n in tot if cnt[n] >= 20), key=lambda n: tot[n])
    steps = sorted((t0, d) for t0, d, n in mev if n == anchor)
    oev = sorted((ops.timestamp_ns * 1000 + e.offset_ps, e.duration_ps, name[e.metadata_id]) for e in ops.events)
    starts = [o[0] for o in oev]
    out = []
    for t0, d in steps:
        i, j = bisect.bisect_left(starts, t0), bisect.bisect_right(starts, t0 + d)
        seq = []
        prev_end = t0
        for s, dd, n in oev[i:j]:
            gap = max(0, s - prev_end)
            seq.append((op_kind(n), dd, gap))
            prev_end = max(prev_end, s + dd)
        out.append(dict(t0=t0, dur=d, tail=max(0, t0 + d - prev_end), seq=seq))
    return anchor, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--device", default="/device:TPU:0")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--skip", type=int, default=2, help="ignore the first steps")
    ap.add_argument("--end", type=int, default=None,
                    help="last step index (exclusive), e.g. to drop the partial-padding tail at n=256")
    a = ap.parse_args()
    an_a, A = load(a.a, a.device)
    an_b, B = load(a.b, a.device)
    print("anchor A:", an_a[:90], "steps", len(A))
    print("anchor B:", an_b[:90], "steps", len(B))
    A, B = A[a.skip:a.end], B[a.skip:a.end]
    us = lambda ps: ps / 1e6  # noqa: E731
    for lab, X in (("A", A), ("B", B)):
        dur = [us(x["dur"]) for x in X]
        busy = [us(sum(o[1] for o in x["seq"])) for x in X]
        gaps = [us(sum(o[2] for o in x["seq"]) + x["tail"]) for x in X]
        print(f"{lab}: steps {len(X)}  module us/step median {statistics.median(dur):.1f}  ops-busy {statistics.median(busy):.1f}"
                    f"  idle-in-module {statistics.median(gaps):.1f}  ops/step {statistics.median(len(x['seq']) for x in X)}")
    # align step k of A with step k of B
    dk_dur, dk_gap, dk_cnt = defaultdict(float), defaultdict(float), defaultdict(int)
    layer_dur, layer_gap = defaultdict(float), defaultdict(float)
    extra_b = defaultdict(float)
    n = min(len(A), len(B))
    for k in range(n):
        sa, sb = A[k]["seq"], B[k]["seq"]
        ka, kb = [o[0] for o in sa], [o[0] for o in sb]
        sm = difflib.SequenceMatcher(a=ka, b=kb, autojunk=False)
        layer = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    if ka[i].startswith("router_topk"):
                        layer += 1
                    dk_dur[ka[i]] += sb[j][1] - sa[i][1]
                    dk_gap[ka[i]] += sb[j][2] - sa[i][2]
                    dk_cnt[ka[i]] += 1
                    layer_dur[layer] += sb[j][1] - sa[i][1]
                    layer_gap[layer] += sb[j][2] - sa[i][2]
            else:
                for j in range(j1, j2):
                    extra_b["+" + kb[j]] += sb[j][1] + sb[j][2]
                for i in range(i1, i2):
                    extra_b["-" + ka[i]] -= sa[i][1] + sa[i][2]
    tot_dur = sum(dk_dur.values())
    tot_gap = sum(dk_gap.values())
    tot_extra = sum(extra_b.values())
    print(f"\nper step, B - A: aligned-op duration {us(tot_dur) / n:+.1f} us, aligned-op idle gap {us(tot_gap) / n:+.1f} us, "
                f"unaligned ops {us(tot_extra) / n:+.1f} us (incl. their gaps)")
    print("module time per step B - A: %+.1f us" % (statistics.mean(us(b["dur"]) for b in B[:n]) -
                                                                                                    statistics.mean(us(x["dur"]) for x in A[:n])))
    print("\nby op kind (us/step): duration delta, gap delta, count/step")
    keys = sorted(dk_dur, key=lambda k: -abs(dk_dur[k] + dk_gap[k]))
    for k in keys[:a.top]:
        print(f"  dur {us(dk_dur[k]) / n:+8.2f}  gap {us(dk_gap[k]) / n:+8.2f}  x{dk_cnt[k] / n:5.1f}  {k[:110]}")
    print("\nunaligned ops (us/step):")
    for k, v in sorted(extra_b.items(), key=lambda kv: -abs(kv[1]))[:10]:
        print(f"  {us(v) / n:+8.2f}  {k[:110]}")
    print("\nby layer (us/step): duration delta, gap delta")
    for l in sorted(layer_dur):
        print(f"  layer {l:2d}: dur {us(layer_dur[l]) / n:+7.2f}  gap {us(layer_gap[l]) / n:+7.2f}")


if __name__ == "__main__":
    main()

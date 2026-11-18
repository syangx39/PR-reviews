#!/usr/bin/env python3
"""Overlay waits in the decode steps of two profiles taken with --xla_tpu_show_overlay_waits_in_profiler=true.

    prof_overlay.py A.xplane.pb B.xplane.pb [--device /device:TPU:0] [--match overlay]

With that libtpu flag the overlayer's wait-for-active-DMA routine brackets every wait for an instruction-memory
(IMEM) overlay load with on-device trace markers. This script
    - lists, per trace line of the device plane, the events that fall inside the decode steps (forward-module
        windows, first 2 steps skipped), to show where the markers land;
    - sums the events whose name matches --match (case-insensitive) per step: count and time;
    - attributes each wait to the XLA op running when it starts, per op kind;
and prints A, B and B - A next to the per-step forward-module time.
"""
import argparse
import bisect
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prof_mskip import XSpace, op_kind  # noqa: E402


def analyze(path, device, match, skip):
    xs = XSpace()
    with open(path, "rb") as f:
        xs.ParseFromString(f.read())
    pl = next(p for p in xs.planes if p.name == device)
    name = {k: v.name or v.display_name for k, v in pl.event_metadata.items()}
    lines = {ln.name: ln for ln in pl.lines}
    t = lambda ln, e: ln.timestamp_ns * 1000 + e.offset_ps  # noqa: E731
    mods = lines["XLA Modules"]
    tot, cnt = defaultdict(int), defaultdict(int)
    for e in mods.events:
        tot[name[e.metadata_id]] += e.duration_ps
        cnt[name[e.metadata_id]] += 1
    anchor = max((n for n in tot if cnt[n] >= 20), key=lambda n: tot[n])
    wins = sorted((t(mods, e), e.duration_ps) for e in mods.events if name[e.metadata_id] == anchor)[skip:]
    wstart = [w[0] for w in wins]

    def step_of(ts):
        i = bisect.bisect_right(wstart, ts) - 1
        return i if i >= 0 and ts < wins[i][0] + wins[i][1] else None

    ops = lines["XLA Ops"]
    oev = sorted((t(ops, e), e.duration_ps, name[e.metadata_id]) for e in ops.events)
    ostart = [o[0] for o in oev]
    per_line = {}
    waits = [[] for _ in wins]
    rx = re.compile(match, re.I)
    for ln in pl.lines:
        if ln.name in ("XLA Ops", "XLA Modules"):
            continue
        c = Counter()
        for e in ln.events:
            ts = t(ln, e)
            s = step_of(ts)
            if s is None:
                continue
            nm = name.get(e.metadata_id, str(e.metadata_id))
            c[nm[:70]] += 1
            if rx.search(nm) or rx.search(ln.name):
                waits[s].append((ts, e.duration_ps, ln.name, nm))
        if c:
            per_line[ln.name] = c
    by_op = defaultdict(float)
    by_op_n = defaultdict(int)
    for ws in waits:
        for ts, d, _, _ in ws:
            i = bisect.bisect_right(ostart, ts) - 1
            k = op_kind(oev[i][2]) if i >= 0 and ts < oev[i][0] + oev[i][1] else "(between ops)"
            by_op[k] += d
            by_op_n[k] += 1
    n = len(wins)
    return dict(
            anchor=anchor, steps=n, per_line=per_line,
            module_us=statistics.median(w[1] / 1e6 for w in wins),
            module_mean_us=statistics.mean(w[1] / 1e6 for w in wins),
            wait_us=statistics.median(sum(d for _, d, _, _ in ws) / 1e6 for ws in waits),
            wait_mean_us=statistics.mean(sum(d for _, d, _, _ in ws) / 1e6 for ws in waits),
            wait_count=statistics.median(len(ws) for ws in waits),
            by_op={k: v / 1e6 / n for k, v in by_op.items()}, by_op_n={k: v / n for k, v in by_op_n.items()},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--device", default="/device:TPU:0")
    ap.add_argument("--match", default="overlay")
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    R = {}
    for lab, p in (("A", a.a), ("B", a.b)):
        r = analyze(p, a.device, a.match, a.skip)
        R[lab] = r
        print(f"{lab}: {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(p)))))} "
                    f"steps {r['steps']}; module us/step median {r['module_us']:.1f} mean {r['module_mean_us']:.1f}; "
                    f"overlay waits/step median {r['wait_count']} totalling {r['wait_us']:.1f} us (mean {r['wait_mean_us']:.1f})")
        for ln, c in r["per_line"].items():
            print(f"   line {ln!r}: {sum(c.values())} events in steps; top {c.most_common(4)}")
    A, B = R["A"], R["B"]
    print(f"\nB - A per step: module (mean) {B['module_mean_us'] - A['module_mean_us']:+.1f} us, "
                f"overlay waits (mean) {B['wait_mean_us'] - A['wait_mean_us']:+.1f} us")
    keys = set(A["by_op"]) | set(B["by_op"])
    print("overlay-wait time by the op it falls in (us/step): A, B, B - A, waits/step A, B")
    for k in sorted(keys, key=lambda k: -abs(B["by_op"].get(k, 0) - A["by_op"].get(k, 0)))[:a.top]:
        print(f"  {A['by_op'].get(k, 0):8.2f} {B['by_op'].get(k, 0):8.2f} {B['by_op'].get(k, 0) - A['by_op'].get(k, 0):+8.2f}"
                    f"   {A['by_op_n'].get(k, 0):5.1f} {B['by_op_n'].get(k, 0):5.1f}  {k[:100]}")


if __name__ == "__main__":
    main()

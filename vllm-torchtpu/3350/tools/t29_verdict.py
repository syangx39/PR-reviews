#!/usr/bin/env python3
"""t29 performance verdict (plan rev3 section 6.2 / 6.3): FP8 MoE row-count (RC) vs zero-weight (ZW) vs skip off (S0),
steady-state decode per step = (t(545) - t(33)) / 512, KV pinned, against the run's frozen manifest.

    python3 t29_verdict.py RESULTS_DIR --manifest FROZEN.json --mode f4|f3|anchor [--numerics PASS|FAIL|INCOMPLETE]
                           [--accept-pod-boundary] [--out report.json]

Modes:
  f4      F4-A: two hosts, 18 points, S0 / ZW / RC with two sessions each per host (orders frozen in the manifest).
  f3      F3 one-host screen, 7 points; every result is labelled SCREEN (no final conclusion).
  anchor  F4-A parent-vs-candidate check on one host, n = 1 / 16 / 256: upbase_off vs S0 and upbase_on vs ZW.
Integrity first (t29_integrity.arm_problems per arm + the frozen session order per host + no same-host group spanning
hosts or pods + no conflicting duplicate session files): missing evidence -> INCOMPLETE (exit 2), contract mismatch ->
FAIL (exit 1); no verdict is printed as final then.
Pod boundary (rev3 6.2, "a session after a host change or preemption starts a new group"): a frozen group whose sessions
ran in more than one pod is INCOMPLETE. --accept-pod-boundary records that as an explicit protocol deviation instead,
computes the comparisons anyway and labels the report SUPPLEMENTARY (never FINAL); every pair and A/A that crosses the
boundary is flagged per point.
Per host h and point n: med = median of a session's 7 per-step times; A/A_h(X) = 100 * (med X#2 / med X#1 - 1)
(sessions in frozen order); band_h = max(0.1 %, |A/A_h(X)|, |A/A_h(Y)|). Pairs within a host: X#1 vs Y#1 and X#2 vs Y#2
(first occurrence with first occurrence); delta_h = mean of the two paired deltas; the host is "stable" when both pairs
have the sign of delta_h.
Outcome per comparison and n, evaluated in this order (v2: the rev3 6.2 rows overlap; the conservative order is fixed
here and applied to every report):
  1. repeatable gain / regression    every host beyond its band in the same direction, every host stable
  2. INCONCLUSIVE (direction)        the hosts' mean deltas have opposite signs, whether or not they are inside the bands
  3. no difference detected at X %   every host within its band; X = the largest band (NOT proof of no regression)
  4. INCONCLUSIVE                    anything else (one host beyond band and another not, an unstable host)
  A stable beyond-band slowdown on one host is flagged to investigate.
Decisions (f4): D2 = every padded point (n = 1,2,4,8,10,12,14,15,17,24,31,250,255) and D3 = every full bucket
(n = 16,32,64,128,256), RC vs ZW: a repeatable regression is FAIL, an unresolved point INCONCLUSIVE; padded vs full is
checked against the measured n -> T bucket counts. No rule turns "not shown slower" into PASS. RC vs S0 is disclosed.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t29_integrity import arm_problems, load_arms, pod_of  # noqa: E402

POINTS = {"f4": [1, 2, 4, 8, 10, 12, 14, 15, 16, 17, 24, 31, 32, 64, 128, 250, 255, 256],
          "f3": [1, 15, 16, 17, 64, 255, 256], "anchor": [1, 16, 256]}
COMPARE = {"f4": [("RC", "ZW"), ("RC", "S0"), ("ZW", "S0")], "f3": [("RC", "ZW"), ("RC", "S0"), ("ZW", "S0")],
           "anchor": [("upbase_off", "S0"), ("upbase_on", "ZW")]}
ROLES = {"f4": ("S0", "ZW", "RC"), "f3": ("S0", "ZW", "RC"), "anchor": ("upbase_off", "S0", "upbase_on", "ZW")}
D2 = [1, 2, 4, 8, 10, 12, 14, 15, 17, 24, 31, 250, 255]
D3 = [16, 32, 64, 128, 256]
FLOOR = 0.10


def outcome_reason(deltas, bands, stable):
    """(outcome, reason) per the rule order in the module docstring."""
    beyond = [abs(d) > b for d, b in zip(deltas, bands)]
    if all(beyond) and all(stable):
        if all(d > 0 for d in deltas):
            return "repeatable regression", "every host beyond its band, same direction, stable"
        if all(d < 0 for d in deltas):
            return "repeatable gain", "every host beyond its band, same direction, stable"
    if any(d > 0 for d in deltas) and any(d < 0 for d in deltas):
        return "INCONCLUSIVE", "direction conflict between hosts"
    if not any(beyond):
        return f"no difference detected at {max(bands):.2f}%", "every host within its band"
    if any(beyond) and not all(beyond):
        return "INCONCLUSIVE", "one host beyond its band, another within"
    return "INCONCLUSIVE", "unstable host (paired deltas disagree in sign)"


def outcome(deltas, bands, stable):
    return outcome_reason(deltas, bands, stable)[0]


def measured_bucket(arms, labels, n):
    """T for decode steps with n valid tokens, from the workers' bucket counts at the end of every listed arm."""
    ts = {}
    for lab in labels:
        for r in ((arms[lab][1].get("workers") or {}).get("at_end") or []):
            for key, cnt in ((r or {}).get("buckets") or {}).items():
                v, _, t = key.partition("->")
                if v == str(n):
                    ts[int(t)] = ts.get(int(t), 0) + cnt
    return max(ts, key=ts.get) if ts else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--manifest", required=False)
    ap.add_argument("--mode", choices=sorted(POINTS), required=True)
    ap.add_argument("--numerics", choices=("PASS", "FAIL", "INCOMPLETE"), help="D1 from t29_numerics.py")
    ap.add_argument("--accept-pod-boundary", action="store_true",
                    help="protocol deviation: pair sessions across a pod rebuild (report labelled SUPPLEMENTARY)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    manifest = json.load(open(a.manifest)) if a.manifest else None
    points, roles = POINTS[a.mode], ROLES[a.mode]
    conflicts = []
    arms = {lab: v for lab, v in load_arms(a.results, conflicts).items() if v[0] in roles and "steady" in v[1]}
    inc, fail, deviations = [], list(conflicts), []
    if manifest is None:
        inc.append("no --manifest: identities and the frozen order cannot be checked")
    for lab, (role, d, _p) in sorted(arms.items()):
        i, f = arm_problems(lab, role, d, manifest, phases=["steady"])
        inc += [f"{lab}: {x}" for x in i]
        fail += [f"{lab}: {x}" for x in f]
        got = [int(x) for x in (d.get("meta") or {}).get("steady_ns") or []]
        if sorted(got) != sorted(points):
            fail.append(f"{lab}: steady points {got} != {points}")
        if (d.get("meta") or {}).get("steady_reps") != 7:
            fail.append(f"{lab}: steady_reps {(d.get('meta') or {}).get('steady_reps')} != 7")
    # frozen session order per host group (manifest["orders"][mode] = {group: [labels in order]})
    groups = ((manifest or {}).get("orders") or {}).get(a.mode) or {}
    if not groups:
        inc.append(f"manifest has no frozen order for mode {a.mode}")
    by_host = {}
    for g, order in groups.items():
        missing = [lab for lab in order if lab not in arms]
        if missing:
            inc.append(f"group {g}: missing sessions {missing}")
            continue
        nodes = sorted({str((arms[lab][1].get("meta") or {}).get("node")) for lab in order})
        if len(nodes) != 1:
            fail.append(f"group {g}: same-host sessions span hosts {nodes} (a new group is needed after a host change)")
            continue
        pods = {lab: pod_of(arms[lab][1]) for lab in order}
        if any(v is None for v in pods.values()):
            msg = f"group {g}: no pod identity for {[lab for lab, v in pods.items() if v is None]}; preemption " \
                  f"boundaries cannot be checked"
            (deviations if a.accept_pod_boundary else inc).append(msg)
        elif len(set(pods.values())) > 1:
            segs, cur = [], None
            for lab in order:
                if pods[lab] != cur:
                    segs.append([pods[lab], []])
                    cur = pods[lab]
                segs[-1][1].append(lab)
            msg = f"group {g}: sessions span {len(set(pods.values()))} pods " + \
                  " | ".join(f"{pod}: {', '.join(labs)}" for pod, labs in segs) + \
                  " (rev3 6.2: a session after a preemption / pod rebuild starts a new group)"
            (deviations if a.accept_pod_boundary else inc).append(msg)
        extra = [lab for lab in arms if lab not in order and (arms[lab][1].get("meta") or {}).get("node") == nodes[0]]
        if extra:
            inc.append(f"group {g}: sessions outside the frozen order on the same host: {extra}")
        starts = [(arms[lab][1].get("meta") or {}).get("start_utc") or "" for lab in order]
        if starts != sorted(starts):
            fail.append(f"group {g}: sessions did not run in the frozen order ({list(zip(order, starts))})")
        sess = {r: [lab for lab in order if arms[lab][0] == r] for r in roles}
        if any(len(v) != 2 for v in sess.values()):
            fail.append(f"group {g}: expected two sessions per arm, got {sess}")
            continue
        by_host[g] = dict(node=nodes[0], sessions=sess, order=order, pods=pods)
    want_hosts = 2 if a.mode == "f4" else 1
    if len(groups) != want_hosts:
        inc.append(f"{len(groups)} host group(s) in the manifest for {a.mode}, expected {want_hosts}")
    label = "SCREEN" if a.mode == "f3" else "FINAL"
    if deviations:
        label = "SUPPLEMENTARY"
    report = dict(mode=a.mode, label=label, hosts=by_host, points={}, decisions={}, protocol_deviations=deviations)
    if inc or fail or len(by_host) != want_hosts:
        for x in inc:
            print(f"INCOMPLETE: {x}")
        for x in fail:
            print(f"FAIL: {x}")
        report.update(incomplete=inc, fail=fail, verdict="FAIL" if fail else "INCOMPLETE")
        print(f"\nverdict: {report['verdict']} (no performance conclusion)")
        if a.out:
            json.dump(report, open(a.out, "w"), indent=1)
        return 1 if fail else 2

    def med(lab, n):
        return statistics.median(arms[lab][1]["steady"][str(n)]["ms_per_step"])

    hosts = sorted(by_host)
    for g in hosts:
        h = by_host[g]
        print(f"\n## host group {g} ({h['node']}): " + " -> ".join(h["order"]))
    for n in points:
        row = dict(hosts={}, compare={})
        for g in hosts:
            s = by_host[g]["sessions"]
            m = {r: [med(lab, n) for lab in s[r]] for r in roles}
            aa = {r: (v[1] / v[0] - 1) * 100 for r, v in m.items()}
            row["hosts"][g] = dict(medians_ms=m, aa_pct=aa)
        for x, y in COMPARE[a.mode]:
            deltas, bands, stable, pairs, cross = [], [], [], [], []
            for g in hosts:
                hm, aa = row["hosts"][g]["medians_ms"], row["hosts"][g]["aa_pct"]
                p = [(hm[x][i] / hm[y][i] - 1) * 100 for i in (0, 1)]
                dlt = sum(p) / 2
                deltas.append(dlt)
                bands.append(max(FLOOR, abs(aa[x]), abs(aa[y])))
                stable.append(all((q > 0) == (dlt > 0) for q in p) if dlt else False)
                pairs.append(p)
                s, pods = by_host[g]["sessions"], by_host[g]["pods"]
                cross.append(dict(pairs=[pods[s[x][i]] != pods[s[y][i]] for i in (0, 1)],
                                  aa={x: pods[s[x][0]] != pods[s[x][1]], y: pods[s[y][0]] != pods[s[y][1]]}))
            out, why = outcome_reason(deltas, bands, stable)
            flag = any(d > b and st for d, b, st in zip(deltas, bands, stable)) and out == "INCONCLUSIVE"
            row["compare"][f"{x} vs {y}"] = dict(outcome=out, reason=why, deltas_pct=deltas, bands_pct=bands,
                                                 stable=stable, pairs_pct=pairs, investigate_one_host_slowdown=flag,
                                                 crosses_pod_boundary=cross)
        report["points"][n] = row
    print(f"\n## per point ({report['label']}); delta = mean of the paired deltas (#1 vs #1, #2 vs #2), band = "
          f"max(0.1%, |A/A|)")
    for n in points:
        row = report["points"][n]
        ms = " | ".join(f"{g}: " + " ".join(f"{r} {statistics.mean(row['hosts'][g]['medians_ms'][r]):.4f}"
                                             for r in roles) for g in hosts)
        print(f"  n={n:>3}  {ms} ms")
        for key, c in row["compare"].items():
            hostpart = "; ".join(f"{g} {d:+.3f}% (pairs {p[0]:+.3f}/{p[1]:+.3f}, band {b:.3f})"
                                 for g, d, p, b in zip(hosts, c["deltas_pct"], c["pairs_pct"], c["bands_pct"]))
            print(f"         {key:22s} {c['outcome']}{'  [investigate]' if c['investigate_one_host_slowdown'] else ''}"
                  f"  {hostpart}")
    if a.mode == "f4":
        tmap = {n: measured_bucket(arms, list(arms), n) for n in points}
        mis = [n for n in D2 if tmap.get(n) is not None and tmap[n] == n] + \
              [n for n in D3 if tmap.get(n) is not None and tmap[n] != n]
        unknown = [n for n in points if tmap.get(n) is None]
        dec = {}
        for name, pts in (("D2", D2), ("D3", D3)):
            res = {n: report["points"][n]["compare"]["RC vs ZW"]["outcome"] for n in pts}
            fails = [n for n, o in res.items() if o == "repeatable regression"]
            unres = [n for n, o in res.items() if o == "INCONCLUSIVE"]
            within = [n for n, o in res.items() if o.startswith("no difference")]
            gains = [n for n, o in res.items() if o == "repeatable gain"]
            dec[name] = dict(per_point=res, FAIL=fails, INCONCLUSIVE=unres, within_band=within, gain=gains,
                             status="FAIL" if fails else ("INCONCLUSIVE" if unres else "no repeatable regression "
                                                          "detected at the listed resolutions"))
        dec["n_to_T"] = tmap
        dec["classification_mismatch"] = mis
        dec["n_to_T_unknown"] = unknown
        dec["D1"] = a.numerics or "not given"
        report["decisions"] = dec
        print("\n## decisions (plan rev3 section 6.3)")
        print(f"  measured n->T: {tmap}")
        if mis or unknown:
            print(f"  padded / full classification mismatch {mis}, unknown {unknown}")
        print(f"  D1 numerics gate: {dec['D1']}")
        for name in ("D2", "D3"):
            dd = dec[name]
            print(f"  {name}: {dd['status']}; regression {dd['FAIL']}, inconclusive {dd['INCONCLUSIVE']}, "
                  f"within band {dd['within_band']}, gain {dd['gain']}")
        if a.numerics != "PASS":
            print("  D1 is not PASS: no performance conclusion.")
    if deviations:
        print("\n## protocol deviations (report is SUPPLEMENTARY, not a strict rev3 confirmation)")
        for x in deviations:
            print(f"  {x}")
    report.update(incomplete=[], fail=[], verdict="REPORTED" if not deviations else "REPORTED_WITH_DEVIATIONS")
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())

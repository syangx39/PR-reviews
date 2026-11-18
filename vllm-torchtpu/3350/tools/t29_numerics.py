#!/usr/bin/env python3
"""t29 numerics gate (plan rev3 section 6.1): N1 engine numerics of RC / ZW / S0 and the parent sanity arms, with the
N2 direct check, against the run's frozen manifest.

    python3 t29_numerics.py RESULTS_DIR --manifest RESOLVED.json [--direct FP8D_JSON] [--out report.json]

Engine arms <tree>[k<blocks>]_<off|zw|rc|on>_<rep> (mskip_engine.py, phases numerics + switch): S0 = fp8rc*_off,
ZW = fp8rc*_zw, RC = fp8rc*_rc, upbase*_off, upbase*_on. Every arm must pass t29_integrity.arm_problems (selectors,
MoE method / branch / filter / count / kernels per rank, model / image / KV / source vs the manifest, complete phases,
finite logprobs, all ranks); S0 must have exactly two sessions (A/A) and every other role exactly one.
Per unit (cases n1 / n10 / n40; switch batches of the first pass) and row: token ids, chosen-token logprobs, lengths and
finish reasons. Gate (all must hold):
    S0 A/A (session 1 vs 2)                      bit-identical
    RC vs S0, upbase_off vs S0, upbase_on vs ZW  bit-identical
    switch pass 2 vs pass 1 inside each arm      bit-identical, and no new compiled program per rank in pass 2
Reported, not waived (plan rev3): ZW vs S0 and RC vs ZW output differences with count and positions; any such
difference pauses the replacement decision (localize routing -> scale -> GMM -> combine -> logits).
N2 (--direct): the fp8_direct_check.py JSON must be ok; its ZW output differences are listed.
Warm-switch compile check: after each of the two switch passes the worker report must hold exactly the expected rank
set (check_arm.expected_rank_set, {0,1,2,3} for TP=4) with no duplicate rank (duplicate = FAIL, missing = INCOMPLETE);
only then are the per-rank program counts compared.
Exit 0 = gate PASS, 1 = FAIL (a contract or a bit-identity check failed), 2 = INCOMPLETE (missing evidence).
(v2, review of PR #134: manifest identity, duplicate session labels and the rank set are enforced.)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_arm import expected_rank_set  # noqa: E402
from t29_integrity import arm_problems, load_arms, pod_of, spans  # noqa: E402

GATE = (("S0#2", "S0#1"), ("RC", "S0#1"), ("upbase_off", "S0#1"), ("upbase_on", "ZW"))
REPORT = (("ZW", "S0#1"), ("RC", "ZW"))


def units(d, pass_index=0):
    if pass_index == 0:
        out = [(f"case {c}", rows) for c, rows in sorted((d.get("cases") or {}).items())]
        out += [(f"switch {b['i']} (n={b['n']})", b["rows"]) for b in d.get("switch") or []]
        return out
    p = (d.get("switch_passes") or [])[pass_index]
    return [(f"switch {b['i']} (n={b['n']})", b["rows"]) for b in p["batches"]]


def compare(ux, uy):
    ux, uy = dict(ux), dict(uy)
    if not ux or set(ux) != set(uy):
        return dict(error=f"units differ or are empty: {sorted(set(ux) ^ set(uy))}")
    diffs, rows, tokens = [], 0, 0
    for name in ux:
        if len(ux[name]) != len(uy[name]) or not ux[name]:
            return dict(error=f"{name}: {len(ux[name])} vs {len(uy[name])} rows")
        for i, (a, b) in enumerate(zip(ux[name], uy[name])):
            rows += 1
            tokens += len(b["token_ids"])
            if a["prompt_sha"] != b["prompt_sha"]:
                return dict(error=f"{name} row {i}: different prompts")
            ta, tb, la, lb = a["token_ids"], b["token_ids"], a["logprobs"] or [], b["logprobs"] or []
            if ta == tb and la == lb and a["finish_reason"] == b["finish_reason"]:
                continue
            ft = next((k for k, (p, q) in enumerate(zip(ta, tb)) if p != q), None)
            if ft is None and len(ta) != len(tb):
                ft = min(len(ta), len(tb))
            fl = next((k for k, (p, q) in enumerate(zip(la, lb)) if p != q), None)
            upto = ft if ft is not None else min(len(la), len(lb))
            dl = [abs(p - q) for p, q in zip(la[:upto], lb[:upto])]
            diffs.append(dict(unit=name, row=i, first_token_diff=ft, first_logprob_diff=fl, len=[len(ta), len(tb)],
                              finish=[a["finish_reason"], b["finish_reason"]],
                              max_abs_dlogprob_before_divergence=max(dl) if dl else 0.0))
    return dict(identical=not diffs, rows=rows, tokens=tokens, rows_differ=len(diffs), diffs=diffs)


def programs_check(d):
    """Switch pass 2 must not compile: per rank, programs after pass 2 == after pass 1.

    Returns (ok, message): ok None = INCOMPLETE (missing report / ranks / counts), False = FAIL (duplicate rank records
    or a new program), True = every expected rank reported once after both passes with unchanged program counts."""
    ps = d.get("switch_passes") or []
    if len(ps) < 2:
        return None, "fewer than 2 switch passes"
    want = expected_rank_set(d.get("meta") or {}, d.get("workers") or {})
    if not want:
        return None, "expected worker rank set unknown (no torch_tpu_topology / after_start report)"
    counts = []
    for k, p in enumerate(ps[:2], start=1):
        rep = p.get("workers_after")
        if not isinstance(rep, list) or not rep:
            return None, f"no worker report after switch pass {k}"
        ranks = [w.get("rank") if isinstance(w, dict) else None for w in rep]
        if any(not isinstance(r, int) or isinstance(r, bool) for r in ranks):
            return None, f"invalid rank entries after switch pass {k}: {ranks}"
        dup = sorted({r for r in ranks if ranks.count(r) > 1})
        if dup:
            return False, f"duplicate rank records after switch pass {k}: {dup} (ranks {ranks})"
        if set(ranks) != want:
            return None, f"ranks after switch pass {k} {sorted(ranks)} != expected {sorted(want)}"
        pc = {w["rank"]: w.get("programs") for w in rep}
        if any(not isinstance(v, int) or isinstance(v, bool) for v in pc.values()):
            return None, f"program counts unavailable after switch pass {k}: {pc}"
        counts.append(pc)
    pa, pb = counts
    new = {r: pb[r] - pa[r] for r in sorted(want) if pb[r] != pa[r]}
    return not new, (f"new programs in pass 2: {new}" if new else f"programs per rank after pass 1 / 2: {pa} / {pb}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--manifest", help="resolved run manifest JSON (model, image, kv_blocks, src_tree_sha256 per tree)")
    ap.add_argument("--direct", help="fp8_direct_check.py JSON (N2)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    manifest = json.load(open(a.manifest)) if a.manifest else None
    conflicts = []
    arms = load_arms(a.results, conflicts)
    by_role = {}
    for lab, (role, d, _p) in sorted(arms.items()):
        by_role.setdefault(role, []).append(lab)
    inc, fail, report = [], list(conflicts), dict(arms={}, pairs={}, reported={})
    if manifest is None:
        inc.append("no --manifest: identities cannot be checked")
    for role, k in {"S0": 2, "ZW": 1, "RC": 1, "upbase_off": 1, "upbase_on": 1}.items():
        got = by_role.get(role, [])
        if len(got) != k:
            inc.append(f"{role}: {len(got)} session(s) {got}, expected {k}")
    for lab, (role, d, _p) in sorted(arms.items()):
        i, f = arm_problems(lab, role, d, manifest, phases=["numerics", "switch"])
        inc += [f"{lab}: {x}" for x in i]
        fail += [f"{lab}: {x}" for x in f]
        ok_prog, msg = programs_check(d)
        if ok_prog is None:
            inc.append(f"{lab}: {msg}")
        elif not ok_prog:
            fail.append(f"{lab}: {msg}")
        report["arms"][lab] = dict(role=role, node=(d.get("meta") or {}).get("node"), pod=pod_of(d), programs=msg)
    report["hosts"] = spans(arms, list(arms))
    print("## arms")
    for lab, info in report["arms"].items():
        print(f"  {lab:26s} {info['role']:10s} node={info['node']}  {info['programs']}")
    roles_ok = all(len(by_role.get(r, [])) == k for r, k in
                   {"S0": 2, "ZW": 1, "RC": 1, "upbase_off": 1, "upbase_on": 1}.items())
    if roles_ok:
        s0 = sorted(by_role["S0"], key=lambda lab: ((arms[lab][1].get("meta") or {}).get("start_utc") or "", lab))
        pick = {"S0#1": s0[0], "S0#2": s0[1], "ZW": by_role["ZW"][0], "RC": by_role["RC"][0],
                "upbase_off": by_role["upbase_off"][0], "upbase_on": by_role["upbase_on"][0]}
        print("\n## pairs (cases n1 / n10 / n40 + switch pass 1: ids, chosen-token logprobs, lengths, finish)")
        for x, y in GATE + REPORT:
            gate = (x, y) in GATE
            try:
                c = compare(units(arms[pick[x]][1]), units(arms[pick[y]][1]))
            except (KeyError, TypeError) as e:
                c = dict(error=f"malformed result: {e!r}")
            report["pairs" if gate else "reported"][f"{x} vs {y}"] = c
            if "error" in c:
                (fail if gate else inc).append(f"{x} vs {y}: {c['error']}")
                print(f"  {x:10s} vs {y:5s}: ERROR {c['error']}")
                continue
            st = "bit-identical" if c["identical"] else f"{c['rows_differ']} of {c['rows']} rows differ"
            tag = "" if c["identical"] else ("  <-- FAIL" if gate else "  <-- difference: pause, localize")
            print(f"  {x:10s} vs {y:5s}: {st} ({c['tokens']} tokens){tag}")
            for dd in c["diffs"][:20]:
                print(f"      {dd['unit']} row {dd['row']}: first token diff {dd['first_token_diff']}, first logprob "
                      f"diff {dd['first_logprob_diff']}, lens {dd['len']}, finish {dd['finish']}, max |dlp| before "
                      f"divergence {dd['max_abs_dlogprob_before_divergence']:.3g}")
            if gate and not c["identical"]:
                fail.append(f"{x} vs {y}: {c['rows_differ']} rows differ")
        print("\n## warm switch: pass 2 vs pass 1 inside each arm")
        for lab, (role, d, _p) in sorted(arms.items()):
            try:
                c = compare(units(d, 0)[len(d.get("cases") or {}):], units(d, 1))
            except (IndexError, KeyError, TypeError) as e:
                c = dict(error=f"no second pass: {e!r}")
            st = "bit-identical" if c.get("identical") else c.get("error") or f"{c['rows_differ']} rows differ"
            print(f"  {lab:26s} {st}")
            if not c.get("identical"):
                (inc if "error" in c else fail).append(f"{lab}: switch pass 2 vs pass 1: {st}")
    if a.direct:
        dj = json.load(open(a.direct)) if os.path.exists(a.direct) else None
        if dj is None:
            inc.append(f"N2 result {a.direct} missing")
        else:
            report["direct"] = dict(ok=dj.get("ok"), cases=dj.get("cases"), failed=len(dj.get("failed") or []),
                                    zw_output_differences=dj.get("zw_output_differences"), weights=dj.get("weights"))
            print(f"\n## N2 direct check: {'PASS' if dj.get('ok') is True else 'FAIL'} ({dj.get('cases')} configs, "
                  f"{len(dj.get('failed') or [])} failed); runtime layout "
                  f"{json.dumps((dj.get('weights') or {}).get('layout'))}")
            for zd in dj.get("zw_output_differences") or []:
                print(f"      ZW output != S0: T={zd['T']} n={zd['n']} seed={zd['seed']} {zd['gates']} rows "
                      f"{zd['rows']} (zero-gate rows {zd['zero_gate_rows']}; without {zd['rows_without_zero_gate']})")
            if dj.get("ok") is not True:
                fail.append("N2 direct check failed")
            if (dj.get("cases") or 0) < 98:
                inc.append(f"N2 has {dj.get('cases')} configs, expected 90 + 8 boundaries")
    else:
        inc.append("no --direct: N2 not checked")
    for x in inc:
        print(f"INCOMPLETE: {x}")
    for x in fail:
        print(f"FAIL: {x}")
    verdict = "FAIL" if fail else ("INCOMPLETE" if inc else "PASS")
    report.update(incomplete=inc, fail=fail, verdict=verdict)
    print(f"\nD1 numerics gate: {verdict}")
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())

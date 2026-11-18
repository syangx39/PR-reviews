#!/usr/bin/env python3
"""t29 F0 tool self-test (plan rev3 section 4): t29_numerics.py, t29_verdict.py and t29_resolve.py must report FAIL or
INCOMPLETE, never PASS / RESOLVED, on broken inputs, and PASS / REPORTED / RESOLVED on a consistent synthetic run.
The synthetic manifests use the archived frozen-manifest structure (identity block, expected_arms, orders, commands).

    python3 selftest_t29_tools.py        # exit 0 when every expectation holds

v2 (review of PR #134) adds: wrong identities against the archived manifest structure, missing identity blocks,
conflicting / identical duplicate session files, incomplete or duplicated rank reports in the warm-switch compile
check, pod rebuilds inside a frozen group, the rule order for direction conflicts, and resolver cases (empty results,
one session missing, combined manifest without expected sessions, frozen orders preserved).
"""
import copy
import gzip
import json
import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t29_numerics  # noqa: E402
import t29_resolve  # noqa: E402
import t29_verdict  # noqa: E402

SRC = {"fp8rc": "a" * 64, "upbase": "b" * 64}
MODEL, IMAGE, KV = "/data/assets/hf/Qwen3.5-35B-A3B-FP8-text", "img@sha256:x", 11000
SEQ = [16, 10, 1, 17]
ENV = {"S0": ("0", "0"), "ZW": ("1", "0"), "RC": ("1", "1"), "upbase_off": ("0", None), "upbase_on": ("1", None)}
TREE = {"S0": "fp8rc", "ZW": "fp8rc", "RC": "fp8rc", "upbase_off": "upbase", "upbase_on": "upbase"}
FLAG = {"S0": "off", "ZW": "zw", "RC": "rc", "upbase_off": "off", "upbase_on": "on"}


def rows(n, tok=7):
    return [dict(prompt_sha=f"p{i}", token_ids=[tok, 3, 5], logprobs=[-0.1, -0.2, -0.3], finish_reason="length")
            for i in range(n)]


def worker(rank, role, n_layers=40):
    skip = ENV[role][0] == "1"
    filt = {"RC": "row_count"}.get(role, "zero_weight" if TREE[role] == "fp8rc" else "None")
    keys = ["(8, 'silu', False, None, %s, <PaddedRowFilter.%s: 'x'>)" % (
        skip, "ROW_COUNT" if role == "RC" else "ZERO_WEIGHT")]
    pol = None if TREE[role] == "upbase" else dict(enabled=role == "RC", fallback_reason=None)
    return dict(rank=rank, kv_num_blocks=KV, programs=200, track_valid_count=role == "RC",
                buckets={"1->16": 5, "16->16": 5, "17->32": 5, "10->16": 5},
                fp8_moe=dict(moe_methods={"VllmFp8MoEMethodTPU": n_layers}, fp8_paths={"fused_moe_gmm": n_layers},
                             fp8_padded_row_filter={filt: n_layers}, skip_flag=skip, dense_expert_threshold=0,
                             fused_moe_kernels=keys, fp8_row_count_policy=pol))


def arm(role, rep, node="host-a", phases=("numerics", "switch"), start="2026-11-18T07:00:00Z", steady_ns=None, ms=10.0,
        pod=None):
    lab = f"{TREE[role]}k{KV}_{FLAG[role]}_{rep}"
    meta = dict(label=lab, node=node, pod=pod or f"{node}-pod1", model=MODEL, image=IMAGE, src_tree_sha256=SRC[TREE[role]],
                moe_skip_env=ENV[role][0], fp8_env={"TPU_MOE_SKIP_FP8_ROW_COUNT": ENV[role][1]}, phases=list(phases),
                torch_tpu_topology="1,2,1,2", start_utc=start, switch_seq=SEQ, steady_ns=steady_ns or [],
                steady_reps=7, steady_lens=[33, 545])
    ws = [worker(r, role) for r in range(4)]
    d = dict(meta=meta, done=True, workers=dict(after_start=copy.deepcopy(ws), at_end=copy.deepcopy(ws)))
    if "numerics" in phases:
        d["cases"] = {"n1": rows(1), "n10": rows(10), "n40": rows(40)}
    if "switch" in phases:
        d["switch"] = [dict(i=i, n=n, rows=rows(n)) for i, n in enumerate(SEQ)]
        d["switch_passes"] = [dict(pass_index=0, batches="see switch", workers_after=copy.deepcopy(ws)),
                              dict(pass_index=1, batches=copy.deepcopy(d["switch"]), workers_after=copy.deepcopy(ws))]
    if "steady" in phases:
        d["steady"] = {str(n): dict(ms_per_step=[ms] * 7, t_short_s=[1.0] * 7, t_long_s=[2.0] * 7, lens=[33, 545])
                       for n in steady_ns}
    return lab, d


IDENTITY = dict(model=MODEL, image=IMAGE, kv_blocks=KV, src_tree_sha256=SRC)
MANIFEST = dict(identity=IDENTITY)          # the archived frozen-manifest structure
LEGACY_MANIFEST = dict(IDENTITY)            # identity fields at the top level only (must not be accepted silently)


def write(dirpath, arms, partial=(), extra=()):
    for lab, d in arms:
        name = f"{lab}.partial.json" if lab in partial else f"{lab}.json"
        json.dump(d, open(os.path.join(dirpath, name), "w"))
    for rel, d in extra:
        os.makedirs(os.path.dirname(os.path.join(dirpath, rel)), exist_ok=True)
        json.dump(d, open(os.path.join(dirpath, rel), "w"))


def numerics_set():
    return [arm("S0", "n1", start="2026-11-18T07:00:00Z"), arm("S0", "n2", start="2026-11-18T07:10:00Z"),
            arm("ZW", "n1", start="2026-11-18T07:20:00Z"), arm("RC", "n1", start="2026-11-18T07:30:00Z"),
            arm("upbase_off", "n1", start="2026-11-18T07:40:00Z"), arm("upbase_on", "n1", start="2026-11-18T07:50:00Z")]


DIRECT = dict(ok=True, cases=98, failed=[], zw_output_differences=[], weights={})


def run_numerics(arms, partial=(), manifest=MANIFEST, direct=DIRECT, extra=()):
    d = tempfile.mkdtemp()
    try:
        write(d, arms, partial, extra)
        mpath, dpath = os.path.join(d, "m.json"), os.path.join(d, "direct.json")
        argv = [d]
        if manifest is not None:
            json.dump(manifest, open(mpath, "w"))
            argv += ["--manifest", mpath]
        if direct is not None:
            json.dump(direct, open(dpath, "w"))
            argv += ["--direct", dpath]
        devnull = open(os.devnull, "w")
        old, sys.stdout = sys.stdout, devnull
        try:
            return t29_numerics.main(argv)
        finally:
            sys.stdout = old
    finally:
        shutil.rmtree(d)


P18 = t29_verdict.POINTS["f4"]


def f4_set(slow_rc=0.0, nodes=("host-a", "host-b")):
    out, orders = [], {}
    for g, node, order in (("A", nodes[0], ["S0", "ZW", "RC", "RC", "ZW", "S0"]),
                           ("B", nodes[1], ["RC", "ZW", "S0", "S0", "ZW", "RC"])):
        seen, labs = {}, []
        for i, role in enumerate(order):
            seen[role] = seen.get(role, 0) + 1
            rep = f"{g.lower()}{seen[role]}"
            lab, d = arm(role, rep, node=node, phases=("steady",), start=f"2026-11-18T08:{10 * i:02d}:00Z",
                         steady_ns=P18, ms=10.0 * (1 + (slow_rc if role == "RC" else 0.0)))
            out.append((lab, d))
            labs.append(lab)
        orders[g] = labs
    man = dict(MANIFEST, orders={"f4": orders})
    return out, man


def run_verdict(arms, manifest, mode="f4", partial=(), extra=(), flags=(), report=None):
    d = tempfile.mkdtemp()
    try:
        write(d, arms, partial, extra)
        mpath, out = os.path.join(d, "m.json"), os.path.join(d, "r.json")
        json.dump(manifest, open(mpath, "w"))
        devnull = open(os.devnull, "w")
        old, sys.stdout = sys.stdout, devnull
        try:
            rc = t29_verdict.main([d, "--manifest", mpath, "--mode", mode, "--numerics", "PASS", "--out", out,
                                   *flags])
        finally:
            sys.stdout = old
        if report is not None and os.path.exists(out):
            report.update(json.load(open(out)))
        return rc
    finally:
        shutil.rmtree(d)


def frozen_for(arms, mode="f2"):
    labs = [lab for lab, _ in arms]
    return dict(identity=IDENTITY, expected_arms=labs, orders={mode: {mode: labs}}, commands=["launch"],
                session_contract=dict(fresh_process=True), template_head={}, run_id="selftest", plan_revision="rev3")


def run_resolve(arms, frozen, result=None):
    d = tempfile.mkdtemp()
    try:
        res = os.path.join(d, "res")
        os.makedirs(res)
        write(res, arms)
        fpath, out = os.path.join(d, "frozen.json"), os.path.join(d, "resolved.yaml")
        json.dump(frozen, open(fpath, "w"))
        old, sys.stdout = sys.stdout, open(os.devnull, "w")
        try:
            rc = t29_resolve.main(["--frozen", fpath, "--results", res, "--out", out])
        finally:
            sys.stdout = old
        if result is not None:
            result.update(json.load(open(os.path.splitext(out)[0] + ".json")))
        return rc
    finally:
        shutil.rmtree(d)


def mutate(arms, label_part, fn):
    out = []
    for lab, d in arms:
        d = copy.deepcopy(d)
        if label_part in lab:
            fn(d)
        out.append((lab, d))
    return out


def main():
    cases = []
    base = numerics_set()
    cases.append(("numerics: consistent run", run_numerics(base), {0}))
    cases.append(("numerics: no manifest", run_numerics(base, manifest=None), {2}))
    cases.append(("numerics: no direct check", run_numerics(base, direct=None), {2}))
    cases.append(("numerics: direct check failed", run_numerics(base, direct=dict(DIRECT, ok=False)), {1}))
    cases.append(("numerics: missing reference S0", run_numerics([a for a in base if "_off_" not in a[0] or "upbase" in a[0]]), {2}))
    cases.append(("numerics: missing candidate RC", run_numerics([a for a in base if "_rc_" not in a[0]]), {2}))
    cases.append(("numerics: RC only as .partial", run_numerics(base, partial={base[3][0]}), {2}))
    cases.append(("numerics: empty arm", run_numerics(mutate(base, "_rc_", lambda d: (d.pop("cases"), d.pop("switch")))), {1, 2}))
    cases.append(("numerics: done=false", run_numerics(mutate(base, "_zw_", lambda d: d.update(done=False))), {2}))
    cases.append(("numerics: missing rank", run_numerics(mutate(base, "_rc_", lambda d: d["workers"]["at_end"].pop())), {2}))
    cases.append(("numerics: duplicate rank", run_numerics(mutate(base, "_rc_", lambda d: d["workers"]["at_end"][3].update(rank=2))), {2}))
    cases.append(("numerics: model mismatch", run_numerics(mutate(base, "_rc_", lambda d: d["meta"].update(model="/x"))), {1}))
    cases.append(("numerics: KV mismatch", run_numerics(mutate(base, "_zw_", lambda d: [w.update(kv_num_blocks=KV - 1) for w in d["workers"]["after_start"]])), {1}))
    cases.append(("numerics: source mismatch", run_numerics(mutate(base, "_rc_", lambda d: d["meta"].update(src_tree_sha256=SRC["upbase"]))), {1}))
    cases.append(("numerics: image mismatch", run_numerics(mutate(base, "upbasek", lambda d: d["meta"].update(image="other"))), {1}))
    cases.append(("numerics: FP8 fell back to BF16", run_numerics(mutate(base, "_rc_", lambda d: [w["fp8_moe"].update(moe_methods={"VllmUnquantizedFusedMoEMethod": 40}) for w in d["workers"]["at_end"]])), {1}))
    cases.append(("numerics: RC silently ZERO_WEIGHT", run_numerics(mutate(base, "_rc_", lambda d: [w["fp8_moe"].update(fp8_padded_row_filter={"zero_weight": 40}) for w in d["workers"]["at_end"]])), {1}))
    cases.append(("numerics: RC without ROW_COUNT kernel", run_numerics(mutate(base, "_rc_", lambda d: [w["fp8_moe"].update(fused_moe_kernels=["(8, 'silu', False, None, True, <PaddedRowFilter.ZERO_WEIGHT: 'x'>)"]) for w in d["workers"]["at_end"]])), {1}))
    cases.append(("numerics: RC on another branch", run_numerics(mutate(base, "_rc_", lambda d: [w["fp8_moe"].update(fp8_paths={"fused_moe_decode": 40}) for w in d["workers"]["at_end"]])), {1}))
    cases.append(("numerics: ZW selector wrong", run_numerics(mutate(base, "_zw_", lambda d: d["meta"]["fp8_env"].update(TPU_MOE_SKIP_FP8_ROW_COUNT="1"))), {1}))
    cases.append(("numerics: S0 count buffer allocated", run_numerics(mutate(base, "fp8rck11000_off_n1", lambda d: [w.update(track_valid_count=True) for w in d["workers"]["at_end"]])), {1}))
    cases.append(("numerics: truncated logprobs", run_numerics(mutate(base, "_rc_", lambda d: d["cases"]["n10"][3].update(logprobs=[-0.1]))), {1, 2}))
    cases.append(("numerics: NaN logprob", run_numerics(mutate(base, "_rc_", lambda d: d["cases"]["n40"][0].update(logprobs=[-0.1, math.nan, -0.3]))), {1, 2}))
    cases.append(("numerics: RC output differs", run_numerics(mutate(base, "_rc_", lambda d: d["cases"]["n10"][2].update(token_ids=[9, 3, 5]))), {1}))
    cases.append(("numerics: S0 A/A differs", run_numerics(mutate(base, "fp8rck11000_off_n2", lambda d: d["switch"][1]["rows"][0].update(finish_reason="stop"))), {1}))
    cases.append(("numerics: upbase_on != ZW", run_numerics(mutate(base, "upbasek11000_on", lambda d: d["cases"]["n1"][0].update(logprobs=[-0.1, -0.2, -0.31]))), {1}))
    cases.append(("numerics: new program in switch pass 2", run_numerics(mutate(base, "_zw_", lambda d: d["switch_passes"][1]["workers_after"][0].update(programs=201))), {1}))
    cases.append(("numerics: no second switch pass", run_numerics(mutate(base, "_zw_", lambda d: d["switch_passes"].pop())), {2}))
    # v2: identity against the archived manifest structure, duplicates, rank sets
    cases.append(("numerics: legacy top-level identity only", run_numerics(base, manifest=LEGACY_MANIFEST), {2}))
    cases.append(("numerics: identity block without source hashes", run_numerics(base, manifest=dict(identity={k: v for k, v in IDENTITY.items() if k != "src_tree_sha256"})), {2}))
    cases.append(("numerics: RC model+image+source wrong", run_numerics(mutate(base, "_rc_", lambda d: d["meta"].update(model="/x", image="other", src_tree_sha256="c" * 64))), {1}))
    rc_lab, rc_d = next(a for a in base if "_rc_" in a[0])
    rc_alt = copy.deepcopy(rc_d)
    rc_alt["cases"]["n10"][2]["token_ids"] = [9, 3, 5]
    cases.append(("numerics: duplicate RC label, other tokens", run_numerics(base, extra=[(f"copy/{rc_lab}.json", rc_alt)]), {1}))
    cases.append(("numerics: identical duplicate copy", run_numerics(base, extra=[(f"copy/{rc_lab}.json", rc_d)]), {0}))
    cases.append(("numerics: switch reports only rank 0", run_numerics(mutate(base, "_zw_", lambda d: [p.update(workers_after=p["workers_after"][:1]) for p in d["switch_passes"]])), {2}))
    cases.append(("numerics: new compile masked by duplicate rank", run_numerics(mutate(base, "_zw_", lambda d: (d["switch_passes"][1]["workers_after"][0].update(programs=201), d["switch_passes"][1]["workers_after"].append(dict(d["switch_passes"][1]["workers_after"][1], rank=0, programs=200))))), {1}))
    f4, man = f4_set()
    cases.append(("verdict f4: consistent run", run_verdict(f4, man), {0}))
    cases.append(("verdict f4: RC identity wrong (archived manifest)", run_verdict(mutate(f4, "_rc_a1", lambda d: d["meta"].update(model="/x")), man), {1}))
    cases.append(("verdict f4: legacy manifest without identity", run_verdict(f4, dict(LEGACY_MANIFEST, orders=man["orders"])), {2}))
    a1_lab, a1_d = next(a for a in f4 if a[0].endswith("_rc_a1"))
    a1_alt = copy.deepcopy(a1_d)
    a1_alt["steady"]["1"]["ms_per_step"] = [12.0] * 7
    cases.append(("verdict f4: duplicate session, other timings", run_verdict(f4, man, extra=[(f"copy/{a1_lab}.json", a1_alt)]), {1}))
    cases.append(("verdict f4: pod rebuild inside a group", run_verdict(mutate(f4, "_off_a2", lambda d: d["meta"].update(pod="host-a-pod2")), man), {2}))
    rep_dev = {}
    rc_dev = run_verdict(mutate(f4, "_off_a2", lambda d: d["meta"].update(pod="host-a-pod2")), man,
                         flags=("--accept-pod-boundary",), report=rep_dev)
    cases.append(("verdict f4: pod rebuild accepted -> SUPPLEMENTARY", 0 if (rc_dev == 0 and rep_dev.get("label") == "SUPPLEMENTARY" and rep_dev.get("protocol_deviations")) else 1, {0}))
    cases.append(("verdict f4: no pod identity", run_verdict(mutate(f4, "_zw_b1", lambda d: d["meta"].pop("pod")), man), {2}))
    out_c, why_c = t29_verdict.outcome_reason([-0.086, +0.066], [0.10, 0.10], [True, True])
    cases.append(("verdict rule: direction conflict inside bands", 0 if out_c == "INCONCLUSIVE" and "direction" in why_c else 1, {0}))
    out_w, _ = t29_verdict.outcome_reason([-0.05, -0.04], [0.10, 0.10], [True, True])
    cases.append(("verdict rule: same direction inside bands", 0 if out_w.startswith("no difference") else 1, {0}))
    # v2: resolver
    fz = frozen_for(base)
    res_ok = {}
    cases.append(("resolve: consistent run", run_resolve(base, fz, res_ok), {0}))
    orders_kept = (res_ok.get("session_contract") or {}).get("frozen_orders") == fz["orders"] and \
        (res_ok.get("session_contract") or {}).get("host_session_command_manifest")
    cases.append(("resolve: frozen orders and commands preserved", 0 if orders_kept else 1, {0}))
    cases.append(("resolve: empty results", run_resolve([], fz), {2}))
    cases.append(("resolve: one session missing", run_resolve(base[1:], fz), {2}))
    cases.append(("resolve: unexpected extra session", run_resolve(base, frozen_for(base[1:])), {2}))
    combined = dict(identity=IDENTITY, session_contract=dict(frozen_orders={"f2": [a[0] for a in base]}), run_id="x")
    cases.append(("resolve: combined manifest without expected_arms", run_resolve(base, combined), {2}))
    cases.append(("resolve: wrong identity", run_resolve(mutate(base, "_rc_", lambda d: d["meta"].update(image="other")), fz), {2}))
    res_pod = {}
    run_resolve(mutate(base, "_rc_", lambda d: d["meta"].update(pod="host-a-pod2")), fz, res_pod)
    cases.append(("resolve: pod rebuild recorded as deviation", 0 if res_pod.get("protocol_deviations") else 1, {0}))
    cases.append(("verdict f4: missing session", run_verdict(f4[1:], man), {2}))
    cases.append(("verdict f4: .partial session", run_verdict(f4, man, partial={f4[2][0]}), {2}))
    cases.append(("verdict f4: same-host group spans hosts", run_verdict(mutate(f4, "_a2", lambda d: d["meta"].update(node="host-c")), man), {1}))
    cases.append(("verdict f4: out of frozen order", run_verdict(mutate(f4, "fp8rck11000_off_a1", lambda d: d["meta"].update(start_utc="2026-11-18T09:59:00Z")), man), {1}))
    cases.append(("verdict f4: 5 pairs instead of 7", run_verdict(mutate(f4, "_zw_b1", lambda d: d["meta"].update(steady_reps=5)), man), {1}))
    cases.append(("verdict f4: KV mismatch", run_verdict(mutate(f4, "_rc_b2", lambda d: [w.update(kv_num_blocks=1) for w in d["workers"]["after_start"]]), man), {1}))
    cases.append(("verdict f4: RC fell back to ZERO_WEIGHT", run_verdict(mutate(f4, "_rc_a1", lambda d: [w["fp8_moe"].update(fp8_padded_row_filter={"zero_weight": 40}) for w in d["workers"]["at_end"]]), man), {1}))
    cases.append(("verdict f4: done=false", run_verdict(mutate(f4, "_zw_a2", lambda d: d.update(done=False)), man), {2}))
    bad = 0
    for name, rc, want in cases:
        ok = rc in want
        bad += not ok
        print(f"{'ok  ' if ok else 'BAD '} {name:45s} exit {rc} (expected {sorted(want)})")
    # the verdict logic itself: a slow RC on both hosts must be a repeatable regression, not PASS
    f4s, mans = f4_set(slow_rc=0.05)
    d = tempfile.mkdtemp()
    write(d, f4s)
    json.dump(mans, open(os.path.join(d, "m.json"), "w"))
    out = os.path.join(d, "r.json")
    old, sys.stdout = sys.stdout, open(os.devnull, "w")
    try:
        t29_verdict.main([d, "--manifest", os.path.join(d, "m.json"), "--mode", "f4", "--numerics", "PASS", "--out", out])
    finally:
        sys.stdout = old
    rep = json.load(open(out))
    reg = rep["decisions"]["D3"]["status"] == "FAIL" and rep["decisions"]["D2"]["status"] == "FAIL"
    print(f"{'ok  ' if reg else 'BAD '} verdict f4: 5% slower RC -> D2 / D3 FAIL ({rep['decisions']['D2']['status']}, "
          f"{rep['decisions']['D3']['status']})")
    bad += not reg
    shutil.rmtree(d)
    print(f"\n{len(cases) + 1 - bad} / {len(cases) + 1} expectations hold")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

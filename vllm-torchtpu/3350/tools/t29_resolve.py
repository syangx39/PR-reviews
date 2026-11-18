#!/usr/bin/env python3
"""t29: fill resolved_run.yaml (plan rev3 run.yaml template) from the frozen run manifests and the run's results.

    python3 t29_resolve.py --frozen manifests/frozen_t29-f1.json [--frozen ...] --results <dir> [--results ...]
                           [--host-map host_map_private.json] --out resolved_run.yaml

Each frozen manifest (written before its run) holds the plan revision, source SHAs, the image digest, the expected
source-tree hashes and KV pin (identity), the expected sessions (expected_arms), the frozen session orders per mode and
group (orders) and the launch commands (commands). A combined manifest may carry the orders and commands nested under
session_contract (frozen_orders / host_session_command_manifest) instead; they are read from there and never replaced
by null.

Status is RESOLVED only when
  (a) the frozen manifests list a non-empty set of expected sessions,
  (b) every expected session has a result (engine sessions by meta.label; non-engine evidence: fp8d_fp8rc.json* for
      "fp8d-fp8rc" / "fp8d_fp8rc", unittest_status.txt for "unittest"),
  (c) no engine result is unexpected or a conflicting duplicate of another file with the same label,
  (d) every engine session passes t29_integrity.arm_problems against the identity of the manifest that expects it,
  (e) every manifest carries its frozen orders, and model / image / source-tree identities agree across manifests.
Otherwise the status is INCOMPLETE and the reasons are listed under missing_evidence (exit 2).
Run-order facts are recorded per frozen group under session_contract.group_checks: sessions out of the frozen order,
and sessions of one group that ran in different pods (preemption / pod rebuild; rev3 6.2 asks for a new group then) are
listed as protocol deviations. They do not block RESOLVED (the evidence exists) but every comparison that relies on
the strict pairing must disclose them.
Hosts (and pods) are pseudonymized with --host-map (private, never published).
(v2, review of PR #134: expected-session check, nested orders preserved, duplicates, multiple frozen manifests.)
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t29_integrity import LABEL, arm_problems, load_arms, manifest_identity, pod_of  # noqa: E402

OTHER_EVIDENCE = {"fp8d_fp8rc": "fp8d_fp8rc.json*", "unittest": "unittest_status.txt"}


def yaml_dump(o, ind=0):
    """Minimal YAML writer (JSON-compatible scalars), enough for the manifest."""
    sp = "  " * ind
    if isinstance(o, dict):
        if not o:
            return "{}"
        out = []
        for k, v in o.items():
            if isinstance(v, (dict, list)) and v:
                out.append(f"{sp}{k}:\n{yaml_dump(v, ind + 1)}")
            else:
                out.append(f"{sp}{k}: {yaml_dump(v, 0) if isinstance(v, (dict, list)) else json.dumps(v)}")
        return "\n".join(out)
    if isinstance(o, list):
        if not o:
            return "[]"
        out = []
        for v in o:
            if isinstance(v, (dict, list)) and v:
                body = yaml_dump(v, ind + 1).lstrip()
                out.append(f"{sp}- {body}")
            else:
                out.append(f"{sp}- {json.dumps(v)}")
        return "\n".join(out)
    return json.dumps(o)


def norm_expected(name):
    return name.replace("-", "_") if name.startswith("fp8d") else name


def frozen_orders_of(fz):
    """mode -> group -> [labels]; a combined manifest's nested group -> [labels] is returned under mode '*'."""
    orders = fz.get("orders")
    if isinstance(orders, dict) and orders:
        return orders
    nested = (fz.get("session_contract") or {}).get("frozen_orders")
    if isinstance(nested, dict) and nested:
        if all(isinstance(v, list) for v in nested.values()):
            return {"*": nested}
        return nested
    return None


def commands_of(fz):
    cmds = fz.get("commands")
    if cmds:
        return cmds
    return (fz.get("session_contract") or {}).get("host_session_command_manifest")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", action="append", required=True)
    ap.add_argument("--results", action="append", required=True)
    ap.add_argument("--host-map", default=None, help="private real -> pseudonym map (never published)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    hmap = json.load(open(a.host_map)) if a.host_map else {}
    missing = []

    frozen = [(os.path.basename(p), json.load(open(p))) for p in a.frozen]
    expect_by, orders, commands, identities = {}, {}, [], []
    for name, fz in frozen:
        exp = [norm_expected(x) for x in (fz.get("expected_arms") or [])]
        if not exp:
            missing.append(f"{name}: no expected session list (expected_arms)")
        for lab in exp:
            if lab in expect_by:
                missing.append(f"{lab}: expected by both {expect_by[lab][0]} and {name}")
            expect_by[lab] = (name, fz)
        fo = frozen_orders_of(fz)
        if not fo:
            missing.append(f"{name}: no frozen session orders")
        else:
            for mode, groups in fo.items():
                orders.setdefault(mode, {}).update(groups)
        cmds = commands_of(fz)
        if not cmds:
            missing.append(f"{name}: no host / session command manifest")
        else:
            commands.append({"manifest": name, "commands": cmds})
        ident, probs = manifest_identity(fz)
        missing += [f"{name}: {x}" for x in probs]
        if ident:
            identities.append((name, ident))
    if not expect_by:
        missing.append("the frozen manifests list no expected sessions at all")
    for key in ("model", "image", "src_tree_sha256"):
        vals = {json.dumps(i.get(key), sort_keys=True) for _n, i in identities}
        if len(vals) > 1:
            missing.append(f"identity '{key}' differs between frozen manifests: {sorted(vals)}")

    conflicts, arms = [], {}
    for r in a.results:
        for lab, v in load_arms(r, conflicts).items():
            if lab in arms:
                if json.dumps(arms[lab][1], sort_keys=True) != json.dumps(v[1], sort_keys=True):
                    conflicts.append(f"duplicate session label {lab}: {arms[lab][2]} and {v[2]} differ")
                continue
            arms[lab] = v
    missing += conflicts

    # pod pseudonyms: per host, numbered in session-start order
    pod_alias, per_host = {}, {}
    for lab, (_role, d, _p) in sorted(arms.items(), key=lambda kv: ((kv[1][1].get("meta") or {}).get("start_utc") or "")):
        meta = d.get("meta") or {}
        pod, node = pod_of(d), meta.get("node")
        if pod is None or pod in pod_alias:
            continue
        host = hmap.get(node, node)
        per_host[host] = per_host.get(host, 0) + 1
        pod_alias[pod] = pod if not hmap else f"{host}-pod{per_host[host]}"

    per_arm = {}
    versions, kv, kv_shapes, compile_sizes, layouts, buckets = set(), set(), set(), set(), set(), {}
    for lab, (role, d, path) in sorted(arms.items()):
        meta, w = d.get("meta") or {}, d.get("workers") or {}
        owner = expect_by.get(lab)
        if owner is None:
            missing.append(f"{lab}: result not expected by any frozen manifest")
            inc, fail = [], ["not expected by any frozen manifest"]
        else:
            inc, fail = arm_problems(lab, role, d, owner[1], phases=None)
        ranks = []
        for rep in (w.get("at_end") or []) if isinstance(w.get("at_end"), list) else []:
            f8 = (rep or {}).get("fp8_moe") or {}
            ranks.append(dict(rank=rep.get("rank"), kv_num_blocks=rep.get("kv_num_blocks"), programs=rep.get("programs"),
                              track_valid_tokens=rep.get("track_valid_count"), moe_methods=f8.get("moe_methods"),
                              fp8_paths=f8.get("fp8_paths"), padded_row_filter=f8.get("fp8_padded_row_filter"),
                              skip_flag=f8.get("skip_flag"), kernels=f8.get("fused_moe_kernels"),
                              policy=f8.get("fp8_row_count_policy"), layer0=f8.get("fp8_layer0"),
                              layouts=f8.get("fp8_layouts"), requant_env=f8.get("requant_env")))
            kv.add(rep.get("kv_num_blocks"))
            kv_shapes.add(json.dumps(rep.get("kv_shapes"), sort_keys=True))
            compile_sizes.add(json.dumps(f8.get("compile_sizes")))
            layouts.add(json.dumps((f8.get("fp8_layer0") or {}).get("layout"), sort_keys=True))
            for k, c in (rep.get("buckets") or {}).items():
                buckets[k] = buckets.get(k, 0) + c
        versions.add(json.dumps(meta.get("runtime_versions"), sort_keys=True))
        node = meta.get("node")
        per_arm[lab] = dict(role=role, file=os.path.relpath(path, os.path.dirname(os.path.dirname(path))),
                            expected_by=owner[0] if owner else None,
                            host=hmap.get(node, node), pod=pod_alias.get(pod_of(d)),
                            session_start_utc=meta.get("start_utc"),
                            session_id=f"{meta.get('group')}/{lab}@{meta.get('start_utc')}",
                            selectors={"TPU_MOE_SKIP_PADDED_TOKENS": meta.get("moe_skip_env"),
                                       "TPU_MOE_SKIP_FP8_ROW_COUNT": (meta.get("fp8_env") or {}).get(
                                           "TPU_MOE_SKIP_FP8_ROW_COUNT")},
                            other_env=meta.get("fp8_env"), image=meta.get("image"), model=meta.get("model"),
                            src_tree_sha256=meta.get("src_tree_sha256"), chat_template_kwargs=meta.get(
                                "chat_template_kwargs"), prompts_sha256=meta.get("prompts_sha256"),
                            num_gpu_blocks_override=meta.get("num_gpu_blocks_override"), phases=meta.get("phases"),
                            done=d.get("done"), per_rank=ranks, integrity_incomplete=inc, integrity_fail=fail)
        if inc or fail:
            missing.append(f"{lab}: {len(inc)} incomplete / {len(fail)} contract problems")

    other_found = {}
    for lab in sorted(expect_by):
        if LABEL.match(lab):
            if lab not in arms:
                missing.append(f"{lab}: no result")
            continue
        pat = OTHER_EVIDENCE.get(lab)
        hits = sorted(h for r in a.results for h in glob.glob(os.path.join(r, "**", pat), recursive=True)) if pat else []
        if not hits:
            missing.append(f"{lab}: no evidence file" + (f" ({pat})" if pat else " (unknown evidence kind)"))
        other_found[lab] = [os.path.relpath(h, os.path.dirname(os.path.dirname(h))) for h in hits]

    group_checks = {}
    for mode, groups in sorted(orders.items()):
        for g, order in sorted(groups.items()):
            eng = [lab for lab in order if LABEL.match(norm_expected(lab)) and lab in arms]
            starts = [(arms[lab][1].get("meta") or {}).get("start_utc") or "" for lab in eng]
            pods = [pod_alias.get(pod_of(arms[lab][1])) for lab in eng]
            segs = []
            for lab, pod in zip(eng, pods):
                if not segs or segs[-1]["pod"] != pod:
                    segs.append(dict(pod=pod, sessions=[]))
                segs[-1]["sessions"].append(lab)
            deviations = []
            if starts != sorted(starts):
                deviations.append("sessions did not start in the frozen order: " +
                                  ", ".join(f"{lab}@{s}" for lab, s in sorted(zip(eng, starts), key=lambda x: x[1])))
            if len({p for p in pods}) > 1:
                deviations.append("sessions span pods (rev3 6.2: a session after a preemption / pod rebuild starts a "
                                  "new group): " + " | ".join(f"{s['pod']}: {', '.join(s['sessions'])}" for s in segs))
            group_checks[f"{mode}/{g}"] = dict(frozen_order=order, engine_sessions=len(eng), pods=segs,
                                               protocol_deviations=deviations)

    fz0 = frozen[0][1]
    res = dict(fz0.get("template_head") or {})
    res.update(status="RESOLVED" if not missing else "INCOMPLETE", ready_to_run=False,
               run_id=fz0.get("run_id"), plan_revision=fz0.get("plan_revision"),
               frozen_manifests=[n for n, _ in frozen])
    res["sources"] = dict(fz0.get("sources") or {})
    res["sources"]["runtime_versions"] = [json.loads(v) for v in sorted(versions)]
    res["identity_per_manifest"] = {n: i for n, i in identities}
    res["inputs"] = fz0.get("inputs")
    res["topology"] = fz0.get("topology")
    res["engine"] = dict(fz0.get("engine") or {}, shared_kv_blocks=sorted(kv, key=str),
                         actual_kv_shapes=[json.loads(v) for v in sorted(kv_shapes)],
                         resolved_compile_sizes=[json.loads(v) for v in sorted(compile_sizes)],
                         actual_n_to_bucket_mapping=dict(sorted(buckets.items())))
    res["runtime_quantization"] = dict(source_expectation="default per-channel runtime scales; confirm after loading",
                                       observed_layout_layer0_all_ranks_all_arms=[json.loads(v) for v in sorted(layouts)])
    res["expected_sessions"] = {lab: owner[0] for lab, owner in sorted(expect_by.items())}
    res["non_engine_evidence"] = other_found
    res["arms"] = per_arm
    base_contract = dict(fz0.get("session_contract") or {})
    base_contract.pop("frozen_orders", None)
    base_contract.pop("host_session_command_manifest", None)
    res["session_contract"] = dict(base_contract, frozen_orders=orders or None,
                                   host_session_command_manifest=commands or None, group_checks=group_checks)
    res["protocol_deviations"] = [f"{g}: {x}" for g, c in group_checks.items() for x in c["protocol_deviations"]]
    res["missing_evidence"] = missing
    with open(a.out, "w") as f:
        f.write("# resolved from " + ", ".join(n for n, _ in frozen) + " and the run results; see tools/t29_resolve.py\n")
        f.write(yaml_dump(res) + "\n")
    json.dump(res, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(f"wrote {a.out}: {len(per_arm)} engine sessions, status {res['status']}, missing {len(missing)}, "
          f"protocol deviations {len(res['protocol_deviations'])}")
    for x in missing:
        print(f"  MISSING: {x}")
    for x in res["protocol_deviations"]:
        print(f"  DEVIATION: {x}")
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())

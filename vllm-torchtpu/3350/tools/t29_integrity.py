#!/usr/bin/env python3
"""t29 integrity checks shared by t29_numerics.py and t29_verdict.py (plan rev3 section 4).

An arm is checked against its role's selector / dispatch contract and against the run's frozen manifest. Manifest
schema (frozen_*.json as archived): manifest["identity"] = {model, image, kv_blocks, src_tree_sha256: {tree: sha256}};
kv_blocks may be null (no KV pin, F1 only) but the key must exist. A manifest without an identity block, or an identity
block without one of these keys, makes every arm INCOMPLETE (identities cannot be checked); nothing is skipped silently.
Problems come in two kinds:
    INCOMPLETE  evidence missing: .partial / done=false / incomplete phases / missing worker ranks / no probe /
                missing identity in the manifest
    FAIL        evidence contradicts the contract: selectors, MoE method or branch, padded-row filter, count buffer,
                kernel keys, model / KV / source / image vs the manifest, a same-host group spanning hosts,
                two different result files with the same session label
Nothing here returns PASS by default: callers pass only when both lists are empty.
(v2, review of PR #134: identity read from manifest["identity"]; duplicate labels detected; pod identity exposed.)
"""
import glob
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_arm import _check_worker_ranks, problems as phase_problems  # noqa: E402

LABEL = re.compile(r"^(fp8rc|upbase)(k(\d{4,}))?_(off|zw|rc|on)_([a-z0-9]+)$")
ROLE = {("fp8rc", "off"): "S0", ("fp8rc", "zw"): "ZW", ("fp8rc", "rc"): "RC", ("upbase", "off"): "upbase_off",
        ("upbase", "on"): "upbase_on"}
CONTRACT = {  # plan rev3 section 2.1 / run.yaml arms
    "S0": dict(tree="fp8rc", env=("0", "0"), skip=False, filter="zero_weight", count=False, policy=False),
    "ZW": dict(tree="fp8rc", env=("1", "0"), skip=True, filter="zero_weight", count=False, policy=False),
    "RC": dict(tree="fp8rc", env=("1", "1"), skip=True, filter="row_count", count=True, policy=True),
    "upbase_off": dict(tree="upbase", env=("0", None), skip=False, filter="None", count=False, policy=None),
    "upbase_on": dict(tree="upbase", env=("1", None), skip=True, filter="None", count=False, policy=None),
}
FP8_METHOD = "VllmFp8MoEMethodTPU"


def load_arms(root, conflicts=None):
    """label -> (role, dict, path) for every complete-looking engine result below root; .partial files are skipped
    (an arm that only has a .partial file is therefore missing).

    Two files with the same session label are never merged silently: a byte-identical copy (same canonical JSON) is
    ignored, any other duplicate is a contradiction and is appended to `conflicts` (callers report it as FAIL); without
    a `conflicts` list a ValueError is raised instead."""
    arms, canon = {}, {}
    for p in sorted(glob.glob(os.path.join(root, "**", "*.json*"), recursive=True)):
        base = os.path.basename(p)
        if ".partial." in base or not (base.endswith(".json") or base.endswith(".json.gz")):
            continue
        try:
            d = json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))
        except (OSError, ValueError):
            continue
        lab = (d.get("meta") or {}).get("label", "") if isinstance(d, dict) else ""
        m = LABEL.match(lab)
        if not m:
            continue
        c = json.dumps(d, sort_keys=True)
        if lab in arms:
            if canon[lab] == c:
                continue  # an identical archived copy of the same session
            msg = f"duplicate session label {lab}: {arms[lab][2]} and {p} differ"
            if conflicts is None:
                raise ValueError(msg)
            conflicts.append(msg)
            continue
        arms[lab], canon[lab] = (ROLE[(m.group(1), m.group(4))], d, p), c
    return arms


IDENTITY_KEYS = ("model", "image", "kv_blocks", "src_tree_sha256")


def manifest_identity(manifest):
    """(identity, problems) of a frozen / resolved manifest; identity is manifest["identity"] or None."""
    ident = manifest.get("identity") if isinstance(manifest, dict) else None
    if not isinstance(ident, dict):
        return None, ["manifest has no identity block (model / image / kv_blocks / src_tree_sha256)"]
    return ident, [f"manifest identity has no '{k}'" for k in IDENTITY_KEYS if k not in ident]


def pod_of(d):
    """Pod (session container) identity of a result; a pod rebuild after preemption changes it."""
    return (d.get("meta") or {}).get("pod")


def _kernel_filter_keys(keys):
    rc = [k for k in keys if "ROW_COUNT" in k]
    zw_skip = [k for k in keys if "ZERO_WEIGHT" in k and ", True, " in k]
    return rc, zw_skip


def arm_problems(label, role, d, manifest=None, phases=None):
    """(incomplete, fail) problem lists for one arm."""
    inc, fail = [], []
    meta = d.get("meta") or {}
    if d.get("done") is not True:
        inc.append("done is not true")
    inc += [f"phase: {p}" for p in phase_problems(d, phases)]
    w = d.get("workers")
    if not isinstance(w, dict):
        inc.append("no workers probe")
        w = {}
    from check_arm import expected_rank_set
    want = expected_rank_set(meta, w)
    for when in ("after_start", "at_end"):
        inc += [f"workers.{when}: {p}" for p in _check_worker_ranks(w.get(when), want, label)]
    c = CONTRACT[role]
    m = LABEL.match(label)
    # selectors (set or cleared before import, recorded by the engine)
    skip_env = meta.get("moe_skip_env")
    rc_env = (meta.get("fp8_env") or {}).get("TPU_MOE_SKIP_FP8_ROW_COUNT")
    if (skip_env, rc_env) != c["env"]:
        fail.append(f"selectors TPU_MOE_SKIP_PADDED_TOKENS={skip_env!r} TPU_MOE_SKIP_FP8_ROW_COUNT={rc_env!r}, "
                    f"expected {c['env']}")
    # dispatch per rank (at_end: after every phase compiled and ran)
    for r in (w.get("at_end") or []) if isinstance(w.get("at_end"), list) else []:
        if not isinstance(r, dict):
            continue
        rk, f8 = r.get("rank"), r.get("fp8_moe") or {}
        if not f8 or f8.get("error"):
            inc.append(f"rank {rk}: no FP8 MoE probe ({f8.get('error') if f8 else 'missing'})")
            continue
        methods = f8.get("moe_methods") or {}
        if set(methods) != {FP8_METHOD} or not methods.get(FP8_METHOD):
            fail.append(f"rank {rk}: MoE methods {methods} (expected only {FP8_METHOD})")
        n = methods.get(FP8_METHOD, 0)
        if f8.get("fp8_paths") != {"fused_moe_gmm": n}:
            fail.append(f"rank {rk}: FP8 MoE branches {f8.get('fp8_paths')} (expected all fused_moe_gmm)")
        if bool(f8.get("skip_flag")) != c["skip"]:
            fail.append(f"rank {rk}: skip flag {f8.get('skip_flag')} (expected {c['skip']})")
        if f8.get("fp8_padded_row_filter") != {c["filter"]: n}:
            fail.append(f"rank {rk}: padded-row filter {f8.get('fp8_padded_row_filter')} (expected {c['filter']})")
        if bool(r.get("track_valid_count")) != c["count"]:
            fail.append(f"rank {rk}: valid-token count buffer {r.get('track_valid_count')} (expected {c['count']})")
        pol = f8.get("fp8_row_count_policy")
        if c["policy"] is not None and bool((pol or {}).get("enabled")) != c["policy"]:
            fail.append(f"rank {rk}: FP8 row-count policy {pol} (expected enabled={c['policy']})")
        rc_keys, zw_skip_keys = _kernel_filter_keys(f8.get("fused_moe_kernels") or [])
        if role == "RC" and (not rc_keys or zw_skip_keys):
            fail.append(f"rank {rk}: kernels {f8.get('fused_moe_kernels')} (RC must build ROW_COUNT and no "
                        f"skip/ZERO_WEIGHT kernel)")
        if role != "RC" and rc_keys:
            fail.append(f"rank {rk}: a ROW_COUNT kernel was built in {role}")
        if (f8.get("dense_expert_threshold") or 0) > 0:
            fail.append(f"rank {rk}: dense-expert threshold {f8.get('dense_expert_threshold')}")
    # identity vs the frozen manifest (manifest["identity"]; missing identity = INCOMPLETE, never skipped)
    if manifest is not None:
        ident, probs = manifest_identity(manifest)
        inc += probs
        ident = ident or {}
        if "model" in ident:
            if not ident["model"]:
                inc.append("manifest identity model is empty")
            elif meta.get("model") != ident["model"]:
                fail.append(f"model {meta.get('model')!r} != manifest {ident['model']!r}")
        if "image" in ident:
            if not ident["image"]:
                inc.append("manifest identity image is empty")
            elif meta.get("image") != ident["image"]:
                fail.append(f"image {meta.get('image')!r} != manifest {ident['image']!r}")
        if "src_tree_sha256" in ident:
            want_src = (ident["src_tree_sha256"] or {}).get(c["tree"]) if isinstance(ident["src_tree_sha256"], dict) else None
            if not want_src:
                inc.append(f"manifest identity has no source-tree hash for {c['tree']}")
            elif meta.get("src_tree_sha256") != want_src:
                fail.append(f"source tree {str(meta.get('src_tree_sha256'))[:16]} != manifest {c['tree']} {want_src[:16]}")
        if "kv_blocks" in ident and ident["kv_blocks"] is not None:  # null = this manifest pins no KV size (F1)
            kv_want = ident["kv_blocks"]
            label_kv = int(m.group(3)) if m and m.group(3) else None
            kvs = {r.get("kv_num_blocks") for r in (w.get("after_start") or []) if isinstance(r, dict)}
            if kvs != {kv_want} or (label_kv is not None and label_kv != kv_want):
                fail.append(f"KV blocks {sorted(kvs, key=str)} (label {label_kv}) != manifest {kv_want}")
    return inc, fail


def host_groups(arms, labels):
    """Group same-host sessions: {host: [labels in session order]}; a group whose labels expect one host but whose
    meta.node differ is reported by the caller via spans()."""
    out = {}
    for lab in labels:
        node = (arms[lab][1].get("meta") or {}).get("node")
        out.setdefault(node, []).append(lab)
    return out


def spans(arms, group_labels):
    """Hosts of a set of labels that must share one host (a same-host group)."""
    return sorted({str((arms[lab][1].get("meta") or {}).get("node")) for lab in group_labels if lab in arms})

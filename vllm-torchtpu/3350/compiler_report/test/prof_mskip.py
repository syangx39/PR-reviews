#!/usr/bin/env python3
"""Device-trace comparison for the MoeSkip decode steps (the t4 profile runs).

  python3 prof_mskip.py RESULTS_DIR [--n 16 32 8] [--ops 25]

RESULTS_DIR holds <label>.prof.tgz archives from run_arm.sh (or the extracted <label>/ directories); each contains
n<N>/plugins/profile/<ts>/*.xplane.pb, one file per TPU worker, written by vLLM's start_profile() / stop_profile()
around one batch of N requests with --profile-steps decode steps (mskip_engine.py, profile phase).

For every label and N, from the worker that owns /device:TPU:0:
  - the XLA modules on the device ("XLA Modules" line): name, count, mean duration;
  - the decode steps: events of the modules that run once per step (count >= steps - 2); device time per step (sum of
    those modules), step interval (start to start) and the device-busy share of the interval;
  - the XLA ops inside the decode-step modules ("XLA Ops" line), per step, grouped by op name and by category
    (the op's hlo_category stat when the trace has it, otherwise the name without its numeric suffix).
Then, for each N, the per-step difference of every op group between labels that share a host (group).

Needs only the protobuf package: the XSpace subset is declared inline (tensorflow/tsl/profiler/protobuf/xplane.proto).
"""
import argparse
import glob
import json
import os
import re
import statistics
import tarfile
import tempfile
from collections import defaultdict

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def _xspace_class():
    T = descriptor_pb2.FieldDescriptorProto
    fdp = descriptor_pb2.FileDescriptorProto(name="xplane_subset2.proto", package="xprof_subset", syntax="proto3")

    def msg(name, fields, map_entry=False):
        m = fdp.message_type.add(name=name)
        if map_entry:
            m.options.map_entry = True
        for fname, num, ftype, tname, rep in fields:
            f = m.field.add(name=fname, number=num, type=ftype)
            if tname:
                f.type_name = ".xprof_subset." + tname
            if rep:
                f.label = T.LABEL_REPEATED

    msg("XStat", [("metadata_id", 1, T.TYPE_INT64, None, False), ("double_value", 2, T.TYPE_DOUBLE, None, False),
                  ("uint64_value", 3, T.TYPE_UINT64, None, False), ("int64_value", 4, T.TYPE_INT64, None, False),
                  ("str_value", 5, T.TYPE_STRING, None, False), ("bytes_value", 6, T.TYPE_BYTES, None, False),
                  ("ref_value", 7, T.TYPE_UINT64, None, False)])
    msg("XEvent", [("metadata_id", 1, T.TYPE_INT64, None, False), ("offset_ps", 2, T.TYPE_INT64, None, False),
                   ("duration_ps", 3, T.TYPE_INT64, None, False), ("stats", 4, T.TYPE_MESSAGE, "XStat", True)])
    msg("XLine", [("id", 1, T.TYPE_INT64, None, False), ("name", 2, T.TYPE_STRING, None, False),
                  ("timestamp_ns", 3, T.TYPE_INT64, None, False), ("events", 4, T.TYPE_MESSAGE, "XEvent", True)])
    msg("XEventMetadata", [("id", 1, T.TYPE_INT64, None, False), ("name", 2, T.TYPE_STRING, None, False),
                           ("display_name", 4, T.TYPE_STRING, None, False),
                           ("stats", 5, T.TYPE_MESSAGE, "XStat", True)])
    msg("XStatMetadata", [("id", 1, T.TYPE_INT64, None, False), ("name", 2, T.TYPE_STRING, None, False)])
    msg("EventMetadataEntry", [("key", 1, T.TYPE_INT64, None, False),
                               ("value", 2, T.TYPE_MESSAGE, "XEventMetadata", False)], map_entry=True)
    msg("StatMetadataEntry", [("key", 1, T.TYPE_INT64, None, False),
                              ("value", 2, T.TYPE_MESSAGE, "XStatMetadata", False)], map_entry=True)
    msg("XPlane", [("id", 1, T.TYPE_INT64, None, False), ("name", 2, T.TYPE_STRING, None, False),
                   ("lines", 3, T.TYPE_MESSAGE, "XLine", True),
                   ("event_metadata", 4, T.TYPE_MESSAGE, "EventMetadataEntry", True),
                   ("stat_metadata", 5, T.TYPE_MESSAGE, "StatMetadataEntry", True)])
    msg("XSpace", [("planes", 1, T.TYPE_MESSAGE, "XPlane", True)])
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("xprof_subset.XSpace"))


XSpace = _xspace_class()


def _stat_value(st, smeta):
    # ref_value points at a stat-metadata entry whose name is the (interned) string value
    if st.ref_value:
        return smeta.get(st.ref_value)
    return st.str_value or st.int64_value or st.uint64_value or st.double_value or None


def op_kind(name):
    """Group the op instances of all layers: the HLO instruction name without numeric suffixes, plus the result
    shape, e.g. '%gather_fusion.102 = bf16[128,2048]{...} fusion(...)' -> 'gather_fusion bf16[128,2048]'."""
    head, _, rest = name.partition(" = ")
    inst = re.sub(r"\.\d+", "", head.strip().lstrip("%"))
    shape = re.match(r"\(?([a-z0-9]+\[[0-9,]*\])", rest.strip())
    return f"{inst} {shape.group(1) if shape else ''}".strip()


def device_plane(path):
    xs = XSpace()
    with open(path, "rb") as f:
        xs.ParseFromString(f.read())
    for pl in xs.planes:
        if pl.name == "/device:TPU:0":
            return pl
    return None


def summarize(path, steps):
    pl = device_plane(path)
    if pl is None:
        return None
    smeta = {k: v.name for k, v in pl.stat_metadata.items()}
    emeta = {}
    for k, v in pl.event_metadata.items():
        stats = {smeta.get(s.metadata_id, str(s.metadata_id)): _stat_value(s, smeta) for s in v.stats}
        emeta[k] = (v.name or v.display_name, v.display_name or v.name, stats)
    lines = {ln.name: ln for ln in pl.lines}
    mods, ops = lines.get("XLA Modules"), lines.get("XLA Ops")
    if mods is None or ops is None:
        return {"error": f"lines {sorted(lines)}"}
    base_ps = lambda ln, ev: ln.timestamp_ns * 1000 + ev.offset_ps  # noqa: E731
    mod_ev = [(emeta[e.metadata_id][0], base_ps(mods, e), e.duration_ps) for e in mods.events]
    by_name = defaultdict(list)
    for name, t0, d in mod_ev:
        by_name[name].append((t0, d))
    per_step_names = [n for n, v in by_name.items() if len(v) >= steps - 2]
    # decode steps only: the forward module of the prefill step has a different name (other shapes)
    step_mods = sorted((t0, d, n) for n in per_step_names for t0, d in by_name[n])
    # one decode step = the per-step modules between consecutive starts of the most frequent of them
    # the model forward: the per-step module with the most device time (it runs exactly once per decode step)
    anchor = max(per_step_names, key=lambda n: sum(d for _, d in by_name[n]), default=None)
    if anchor is None:
        return {"modules": {n: len(v) for n, v in by_name.items()}, "error": "no per-step module"}
    starts = sorted(t0 for t0, _ in by_name[anchor])
    n_steps = len(starts)
    interval = (starts[-1] - starts[0]) / (n_steps - 1) / 1e6 if n_steps > 1 else None  # us
    windows = [(t0, t0 + d) for t0, d, _ in step_mods]
    win_starts = [w[0] for w in windows]
    dev_ps = sum(d for _, d, _ in step_mods)
    op_name, op_cat, op_cnt = defaultdict(float), defaultdict(float), defaultdict(int)
    import bisect
    for e in ops.events:
        t0 = base_ps(ops, e)
        i = bisect.bisect_right(win_starts, t0) - 1
        if i < 0 or t0 >= windows[i][1]:
            continue
        name, disp, stats = emeta[e.metadata_id]
        cat = stats.get("hlo_category") or re.sub(r"[._]\d+$", "", name)
        k = op_kind(name)
        op_name[k] += e.duration_ps
        op_cnt[k] += 1
        op_cat[str(cat)] += e.duration_ps
    us = lambda ps: ps / 1e6 / n_steps  # noqa: E731
    return {
        "modules": {n: dict(count=len(v), mean_us=round(statistics.mean(d for _, d in v) / 1e6, 2)) for n, v in
                    sorted(by_name.items(), key=lambda kv: -sum(d for _, d in kv[1]))},
        "per_step_modules": per_step_names, "anchor": anchor, "steps": n_steps,
        "device_us_per_step": round(us(dev_ps), 2), "interval_us": round(interval, 2) if interval else None,
        "busy_pct": round(100 * us(dev_ps) / interval, 1) if interval else None,
        "op_us_per_step": {k: round(us(v), 3) for k, v in sorted(op_name.items(), key=lambda kv: -kv[1])},
        "op_count_per_step": {k: round(op_cnt[k] / n_steps, 2) for k in op_name},
        "cat_us_per_step": {k: round(us(v), 3) for k, v in sorted(op_cat.items(), key=lambda kv: -kv[1])},
        "op_example": {},
    }


def find_runs(root):
    """label -> {n: xplane path of the worker with /device:TPU:0}"""
    runs = defaultdict(dict)
    tmp = tempfile.mkdtemp(prefix="mskip_prof_")
    for tgz in glob.glob(os.path.join(root, "**", "*.prof.tgz"), recursive=True):
        with tarfile.open(tgz) as tf:
            tf.extractall(tmp)
    for base in (root, tmp):
        for p in glob.glob(os.path.join(base, "**", "plugins", "profile", "*", "*.xplane.pb"), recursive=True):
            rel = os.path.relpath(p, base).split(os.sep)
            # <label>/n<N>/plugins/profile/<ts>/<file> (possibly below prof/)
            try:
                i = rel.index("plugins")
            except ValueError:
                continue
            label, nn = rel[i - 2], rel[i - 1]
            if not nn.startswith("n"):
                continue
            runs[label].setdefault(int(nn[1:]), []).append(p)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--ops", type=int, default=25)
    ap.add_argument("--json", help="write the summaries here")
    a = ap.parse_args()
    runs = find_runs(a.results)
    out = {}
    for label in sorted(runs):
        for n, paths in sorted(runs[label].items()):
            summ = None
            for p in sorted(paths):
                summ = summarize(p, a.steps)
                if summ is not None:
                    break
            out.setdefault(label, {})[n] = summ
            if not summ or "error" in summ:
                print(f"{label} n={n}: {summ and summ.get('error')}")
                continue
            print(f"\n## {label} n={n}: {summ['steps']} steps, device {summ['device_us_per_step']} us/step, "
                  f"interval {summ['interval_us']} us, busy {summ['busy_pct']}%  (per-step modules: "
                  f"{', '.join(summ['per_step_modules'])})")
            print("  categories (us/step): " + ", ".join(f"{k} {v}" for k, v in list(summ["cat_us_per_step"].items())[:10]))
    # pairwise per-step differences for labels on the same host (same rep suffix)
    reps = defaultdict(dict)
    for label in out:
        reps[label.rsplit("_", 1)[-1]][label.rsplit("_", 1)[0]] = label
    for rep, cfgs in sorted(reps.items()):
        for n in sorted({n for lab in cfgs.values() for n in out[lab]}):
            for test, ref in (("final_on", "base_on"), ("final_on", "base_off"), ("base_on", "base_off")):
                if test not in cfgs or ref not in cfgs:
                    continue
                A, B = out[cfgs[test]].get(n), out[cfgs[ref]].get(n)
                if not A or not B or "error" in A or "error" in B:
                    continue
                print(f"\n### host {rep}, n={n}: {cfgs[test]} - {cfgs[ref]}: device/step "
                      f"{A['device_us_per_step'] - B['device_us_per_step']:+.1f} us, interval "
                      f"{(A['interval_us'] or 0) - (B['interval_us'] or 0):+.1f} us")
                cats = set(A["cat_us_per_step"]) | set(B["cat_us_per_step"])
                dc = sorted(((A["cat_us_per_step"].get(c, 0) - B["cat_us_per_step"].get(c, 0), c) for c in cats),
                            key=lambda x: -abs(x[0]))
                print("  by category: " + ", ".join(f"{c} {d:+.1f}" for d, c in dc[:8]))
                names = set(A["op_us_per_step"]) | set(B["op_us_per_step"])
                dn = sorted(((A["op_us_per_step"].get(k, 0) - B["op_us_per_step"].get(k, 0), k) for k in names),
                            key=lambda x: -abs(x[0]))
                for d, k in dn[:a.ops]:
                    ca, cb = A["op_count_per_step"].get(k, 0), B["op_count_per_step"].get(k, 0)
                    print(f"  {d:+8.2f} us  [{A['op_us_per_step'].get(k, 0):8.2f} vs {B['op_us_per_step'].get(k, 0):8.2f}]"
                          f"  x{ca:g} vs x{cb:g}/step  {k[:150]}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

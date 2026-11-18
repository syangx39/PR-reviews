#!/usr/bin/env python3
"""MoeSkip (TPU_MOE_SKIP_PADDED_TOKENS in the bf16 MoE) upstream-readiness test: one single-host vLLM engine (TP=4).

One engine start runs three phases and writes <out>/<label>.json:
  numerics  GSM8K test prompts 0 / 1-10 / 11-50 as three batches (n1 / n10 / n40), greedy, 512 new tokens, token ids
            and chosen-token logprobs (the PR #101 harness).
  sweep     decode time per step vs batch size: the first n GSM8K prompts, greedy, ignore_eos;
            (t(max_tokens=129) - t(max_tokens=1)) / 128, --reps repeats (default 3) after a warm-up of each n.
  steady    steady-state decode time per step at a constant n: the first n GSM8K prompts, greedy, ignore_eos, run
            with max_tokens L1 and L2 (--steady-lens); (t(L2) - t(L1)) / (L2 - L1). Both runs share the prefill
            steps, the mixed prefill/decode steps and the tail where early requests finish (partial padding), so
            the difference is exactly L2 - L1 decode steps on which all n requests decode. --steady-reps pairs per
            n (order alternates), after one warm-up of each length.
  profile   device traces (vLLM profiler) of --profile-steps decode steps for each n in --profile-ns
  rollout   256 GSM8K prompts with per-request max_tokens from rollout_lengths.json (one replica's rollout in Round 3:
            lengths 1-1,999, mean 341), ignore_eos, greedy, so every arm does the same work; wall time of 3 repeats.
  switch    N1: padding-state transitions in one warm engine. Batches of the first n prompts in the order of
            --switch-seq (full 16-row bucket, padded, full again, different padding, other buckets), greedy, natural
            EOS, --switch-max-tokens new tokens; token ids, chosen-token logprobs and finish reasons per request.
            Requests finish at different steps, so the number of active rows also changes inside a batch.
  graphsel  TPU_MOE_SKIP_STOCK_GRAPH (dual graph): which graph each step ran on and whether anything compiled.
            Batches of the first n prompts in the order of --graphsel-seq, --graphsel-steps decode steps each
            (ignore_eos, so every decode step has n valid rows), the whole sequence twice. Before and after every
            batch each worker reports the runner's per-(bucket, valid rows, graph) step counters and the number of
            programs in the torch_tpu compile cache (collective_rpc). Also times the selection itself on the host.

With TPU_MOE_SKIP_STOCK_GRAPH set (or the graphsel phase) every worker reports after engine start and at the end:
the stock graph's buckets, its trace + compile + warm-up time, the HBM in use it added, the step counters and the
number of compiled programs (result["workers"]).

  python3 mskip_engine.py --label base_off_a --out /out [--phases numerics,sweep,rollout]
"""
import argparse
import hashlib
import inspect
import json
import os
import random
import time

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--out", default="/out")
ap.add_argument("--model", default="/data/assets/hf/Qwen3.5-35B-A3B-text")
ap.add_argument("--data", default="/data/data/gsm8k/test.parquet")
ap.add_argument("--lengths", default="/opt/measure/rollout_lengths.json")
ap.add_argument("--tp", type=int, default=4)
ap.add_argument("--phases", default="numerics,sweep,rollout")
ap.add_argument("--sweep-ns", default="1,2,4,8,10,12,16,24,32,64,128,256")
ap.add_argument("--reps", type=int, default=3, help="timed repeats per sweep point")
ap.add_argument("--steady-ns", default="1,15,16,255,256", help="request counts of the steady phase")
ap.add_argument("--steady-lens", default="33,545", help="max_tokens of the short and the long run (steady phase)")
ap.add_argument("--steady-reps", type=int, default=7, help="timed short/long pairs per steady point")
ap.add_argument("--profile-ns", default="16,32,8", help="request counts profiled in the profile phase")
ap.add_argument("--profile-steps", type=int, default=32, help="decode steps per profiled batch")
ap.add_argument("--profile-dir", help="trace directory (default <out>/prof/<label>)")
ap.add_argument("--switch-seq", default="16,10,16,1,16,15,12,16,3,17,32,16",
                help="request counts of the switch phase, run in this order in one engine")
ap.add_argument("--switch-max-tokens", type=int, default=256)
ap.add_argument("--switch-stagger", type=int, default=0,
                help="request j of a switch batch gets max_tokens = switch-max-tokens - j * stagger (at least 16), so "
                     "requests finish at different steps and the active rows shrink inside the batch; 0 = all equal")
ap.add_argument("--ep", action="store_true", help="enable_expert_parallel=True (EP over the TP ranks)")
ap.add_argument("--graphsel-seq", default="12,13,14,15,16,17,16,12,15,13,16,14,17,15,12,16",
                help="request counts of the graphsel phase, run in this order, twice, in one engine")
ap.add_argument("--graphsel-steps", type=int, default=32, help="decode steps per graphsel batch")
ap.add_argument("--prefill-mixed-reps", type=int, default=5,
                help="timed repeats per sub-workload in the prefill_mixed phase")
ap.add_argument("--probe-workers", action="store_true",
                help="collect worker_report (stock_graph, bf16_policy, kv_num_blocks, programs) after start and at end")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
phases = a.phases.split(",")

t0 = time.perf_counter()
_skip_env = (os.environ.get("TPU_MOE_SKIP_PADDED_TOKENS") or "").strip().lower()
probe_workers = (
    a.probe_workers
    or bool(os.environ.get("TPU_MOE_SKIP_STOCK_GRAPH"))
    or _skip_env not in ("", "0", "false", "off")
    or "graphsel" in phases
)
if probe_workers:
    # collective_rpc with a function defined here: the function is pickled to the workers.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
import pandas as pd  # noqa: E402
import vllm  # noqa: E402
import vllm_torchtpu  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm_torchtpu import envs as vtt_envs  # noqa: E402
from vllm_torchtpu.layers.adapter.quantization import unquantized  # noqa: E402

root = os.path.dirname(vllm_torchtpu.__file__)


def sha(rel):
    path = os.path.join(root, rel)
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16] if os.path.exists(path) else None


src = inspect.getsource(unquantized)
meta = dict(label=a.label, host=os.uname().nodename, node=os.environ.get("NODE_NAME"), group=os.environ.get("MSKIP_GROUP"),
            vllm=vllm.__version__, vllm_torchtpu_file=vllm_torchtpu.__file__,
            moe_skip_env=os.environ.get("TPU_MOE_SKIP_PADDED_TOKENS"),
            moe_skip_flag=bool(vtt_envs.TPU_MOE_SKIP_PADDED_TOKENS),
            unquantized_has_fix=("zero_routing_weights_for_padding" in src) or ("_padding_rows" in src),
            unquantized_mode=(
                "gcount" if "_padding_rows" in src
                else ("zero_weights" if "zero_routing_weights_for_padding" in src else "stock")
            ),
            sha256_16={r: sha(r) for r in ("envs.py", "layers/adapter/quantization/unquantized.py",
                                            "layers/adapter/token_padding.py", "layers/adapter/fused_moe.py",
                                            "layers/adapter/pipelined_fused_moe.py", "layers/core/fused_moe_gmm.py",
                                            "compilation/moe_stock_graph.py", "runner/tpu_runner.py")},
            moe_stock_graph_env=os.environ.get("TPU_MOE_SKIP_STOCK_GRAPH"),
            moe_skip_in_gmm_env=os.environ.get("TPU_MOE_SKIP_IN_GMM"),
            moe_skip_padding_rows_env=os.environ.get("TPU_MOE_SKIP_PADDING_ROWS"),
            router_sha256_16={r: sha(r) for r in ("kernels/router_topk.py", "layers/adapter/router_topk.py",
                                                   "layers/adapter/moe_routing.py",
                                                   "compilation/moe_stock_graph.py")},
            steady_ns=[int(x) for x in a.steady_ns.split(",")],
            steady_lens=[int(x) for x in a.steady_lens.split(",")], steady_reps=a.steady_reps,
            graphsel_seq=[int(x) for x in a.graphsel_seq.split(",")], graphsel_steps=a.graphsel_steps,
            prefill_mixed_reps=a.prefill_mixed_reps,
            torch_tpu_topology=os.environ.get("TORCH_TPU_TOPOLOGY"), phases=phases,
            v1_multiprocessing=os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
            sweep_ns=[int(x) for x in a.sweep_ns.split(",")], reps=a.reps, pod=os.environ.get("POD_NAME"),
            profile_ns=[int(x) for x in a.profile_ns.split(",")], profile_steps=a.profile_steps,
            switch_seq=[int(x) for x in a.switch_seq.split(",")], switch_max_tokens=a.switch_max_tokens,
            switch_stagger=a.switch_stagger, ep=a.ep,
            moe_collection_chunk_size=os.environ.get("TPU_MOE_COLLECTION_CHUNK_SIZE"))
print("META", json.dumps(meta), flush=True)

df = pd.read_parquet(a.data)
prof_dir = a.profile_dir or os.path.join(a.out, "prof", a.label)
extra_llm = {}
if "profile" in phases:
    # vLLM profiler API; on TPU each worker writes an XSpace (<dir>/<prefix>/plugins/profile/<ts>/*.xplane.pb).
    extra_llm["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": prof_dir}
if a.ep:
    extra_llm["enable_expert_parallel"] = True
llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_num_seqs=256, max_model_len=4352,
          max_num_batched_tokens=4096, gpu_memory_utilization=0.70, seed=0, max_logprobs=1, **extra_llm)
tok = llm.get_tokenizer()
prompts = [tok.apply_chat_template(list(m), tokenize=False, add_generation_prompt=True) for m in df["prompt"][:256]]
meta["engine_init_s"] = round(time.perf_counter() - t0, 1)
print("ENGINE_READY", meta["engine_init_s"], flush=True)
result = dict(meta=meta)


def save():
    json.dump(result, open(f"{a.out}/{a.label}.json", "w"))


def worker_report(worker):
    """Runs on every worker (collective_rpc): stock-graph state, policy, KV blocks, and compiled-program count."""
    import torch as _torch
    runner = worker.model_runner
    dispatch = getattr(runner, "_moe_stock_graph", None)
    policy = getattr(runner, "_bf16_moe_skip_policy", None)
    pad_state = getattr(runner, "_token_padding_state", None)
    kv_cfg = getattr(runner, "kv_cache_config", None)
    try:
        programs = len(_torch.tpu._get_cache_stats().per_entry_stats)
    except Exception as e:  # noqa: BLE001
        programs = f"unavailable: {e!r}"[:200]
    return dict(
        rank=getattr(worker, "rank", None),
        stock_graph=dispatch.stats() if dispatch is not None else None,
        bf16_policy=policy.as_dict() if policy is not None and hasattr(policy, "as_dict") else None,
        track_valid_count=(pad_state is not None and getattr(pad_state, "local_num_valid", None) is not None),
        kv_num_blocks=getattr(kv_cfg, "num_blocks", None),
        programs=programs,
    )


def probe():
    try:
        return llm.collective_rpc(worker_report)
    except Exception as e:  # noqa: BLE001  recorded; the checks treat a missing report as a failure
        return f"collective_rpc failed: {e!r}"[:500]


if probe_workers:
    result["workers"] = dict(after_start=probe())
    print("WORKERS", json.dumps(result["workers"]["after_start"])[:2000], flush=True)
    save()


def rows_of(outs):
    rows = []
    for o in outs:
        c = o.outputs[0]
        lps = None
        try:
            if c.logprobs:
                lps = [float(step[t].logprob) for step, t in zip(c.logprobs, c.token_ids)]
        except Exception as e:  # keep the token ids even if logprobs come in another container
            print("LOGPROB_EXTRACT_FAILED", repr(e)[:200], flush=True)
        rows.append(dict(prompt_sha=hashlib.sha256(o.prompt.encode()).hexdigest()[:16],
                         token_ids=list(c.token_ids), logprobs=lps, finish_reason=c.finish_reason))
    return rows


if "numerics" in phases:
    result["cases"] = {}
    for name, ps in {"n1": prompts[0:1], "n10": prompts[1:11], "n40": prompts[11:51]}.items():
        s = time.perf_counter()
        outs = llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=512, logprobs=1), use_tqdm=False)
        rows = rows_of(outs)
        result["cases"][name] = rows
        print("CASE", name, f"{time.perf_counter() - s:.1f}s", "tokens", sum(len(r["token_ids"]) for r in rows), flush=True)
        save()


def timed(ps, sp):
    s = time.perf_counter()
    llm.generate(ps, sp, use_tqdm=False)
    return time.perf_counter() - s


if "sweep" in phases:
    result["sweep"] = {}
    one = SamplingParams(temperature=0.0, max_tokens=1)
    long = SamplingParams(temperature=0.0, max_tokens=129, ignore_eos=True)
    for n in [int(x) for x in a.sweep_ns.split(",")]:
        ps = prompts[:n]
        timed(ps, SamplingParams(temperature=0.0, max_tokens=17, ignore_eos=True))  # warm-up / compile
        reps, t_prefills, t_fulls = [], [], []
        for _ in range(a.reps):
            t_prefill = timed(ps, one)
            t_full = timed(ps, long)
            t_prefills.append(round(t_prefill, 6))
            t_fulls.append(round(t_full, 6))
            reps.append((t_full - t_prefill) / 128 * 1e3)
        result["sweep"][n] = dict(
            ms_per_step=reps,
            t_prefill_s=t_prefills,
            t_full_s=t_fulls,
            best=min(reps),
            median=sorted(reps)[len(reps) // 2],
        )
        print("SWEEP", n, " ".join(f"{x:.3f}" for x in reps), flush=True)
        save()

if "steady" in phases:
    result["steady"] = {}
    l1, l2 = (int(x) for x in a.steady_lens.split(","))
    assert 1 <= l1 < l2, a.steady_lens
    sp_short = SamplingParams(temperature=0.0, max_tokens=l1, ignore_eos=True)
    sp_long = SamplingParams(temperature=0.0, max_tokens=l2, ignore_eos=True)
    for n in [int(x) for x in a.steady_ns.split(",")]:
        ps = prompts[:n]
        timed(ps, sp_short)  # warm-up / compile
        timed(ps, sp_long)
        per, t_short, t_long = [], [], []
        for r in range(a.steady_reps):
            if r % 2 == 0:
                ts = timed(ps, sp_short)
                tl = timed(ps, sp_long)
            else:
                tl = timed(ps, sp_long)
                ts = timed(ps, sp_short)
            t_short.append(round(ts, 6))
            t_long.append(round(tl, 6))
            per.append((tl - ts) / (l2 - l1) * 1e3)
        result["steady"][n] = dict(
            ms_per_step=per, t_short_s=t_short, t_long_s=t_long, lens=[l1, l2],
            best=min(per), median=sorted(per)[len(per) // 2],
        )
        print("STEADY", n, " ".join(f"{x:.4f}" for x in per), flush=True)
        save()

if "prefill_mixed" in phases:
    # Representative pure-prefill and mixed prefill/decode batches (P2 prefill/mixed regression check).
    result["prefill_mixed"] = {}
    long_prompts = [prompts[i] + "\n\n" + prompts[(i + 64) % 256] for i in range(64)]
    pm_cases = {
        "prefill_n16": (prompts[:16], SamplingParams(temperature=0.0, max_tokens=1)),
        "prefill_n64": (prompts[:64], SamplingParams(temperature=0.0, max_tokens=1)),
        "mixed_16_long": (
            prompts[:8] + long_prompts[8:16],
            [SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True) for _ in range(8)]
            + [SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True) for _ in range(8)],
        ),
        "mixed_64_stagger": (
            prompts[:32] + long_prompts[32:64],
            [SamplingParams(temperature=0.0, max_tokens=[8, 16, 24, 32][i % 4], ignore_eos=True) for i in range(64)],
        ),
    }
    for name, (ps, sp) in pm_cases.items():
        # Warm-up pass (also captures prompt & output token counts).
        warm_outs = llm.generate(ps, sp, use_tqdm=False)
        prompt_toks = sum(len(o.prompt_token_ids or []) for o in warm_outs)
        out_toks = sum(len(o.outputs[0].token_ids) for o in warm_outs)
        walls = []
        for _ in range(a.prefill_mixed_reps):
            walls.append(round(timed(ps, sp), 6))
        result["prefill_mixed"][name] = dict(
            wall_s=walls,
            best=min(walls),
            median=sorted(walls)[len(walls) // 2],
            requests=len(ps),
            prompt_tokens=prompt_toks,
            output_tokens=out_toks,
        )
        print("PREFILL_MIXED", name, " ".join(f"{x:.4f}" for x in walls), flush=True)
        save()

if "rollout" in phases:
    lengths = json.load(open(a.lengths))["lengths"]
    order = list(range(len(lengths)))
    random.Random(0).shuffle(order)  # fixed submission order, long requests not all last
    lens = [lengths[i] for i in order]
    sps = [SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True) for n in lens]
    walls = []
    for r in range(3):
        walls.append(timed(prompts[:len(lens)], sps))
        print("ROLLOUT", r, f"{walls[-1]:.2f}s", flush=True)
    result["rollout"] = dict(wall_s=walls, best=min(walls), requests=len(lens), tokens=sum(lens), max_len=max(lens))
    save()

if "profile" in phases:
    # Device traces of decode steps: per request count n, a warm-up, then one batch of n prompts with
    # --profile-steps decode steps (ignore_eos) between start_profile() and stop_profile().
    result["profile"] = {}
    for n in [int(x) for x in a.profile_ns.split(",")]:
        ps = prompts[:n]
        warm = SamplingParams(temperature=0.0, max_tokens=17, ignore_eos=True)
        timed(ps, warm)
        timed(ps, warm)
        sp = SamplingParams(temperature=0.0, max_tokens=a.profile_steps + 1, ignore_eos=True)
        try:
            llm.start_profile(profile_prefix=f"n{n}")
        except TypeError:  # older LLM.start_profile() without a prefix
            llm.start_profile()
        time.sleep(1.0)
        wall = timed(ps, sp)
        llm.stop_profile()
        result["profile"][n] = dict(wall_s=wall, steps=a.profile_steps, dir=prof_dir)
        print("PROFILE", n, f"{wall:.3f}s", flush=True)
        save()

if "switch" in phases:
    result["switch"] = []
    for i, n in enumerate(int(x) for x in a.switch_seq.split(",")):
        sps = [SamplingParams(temperature=0.0, max_tokens=max(16, a.switch_max_tokens - j * a.switch_stagger),
                              logprobs=1) for j in range(n)]
        s = time.perf_counter()
        rows = rows_of(llm.generate(prompts[:n], sps, use_tqdm=False))
        result["switch"].append(dict(i=i, n=n, rows=rows, seconds=round(time.perf_counter() - s, 2)))
        print("SWITCH", i, n, f"{time.perf_counter() - s:.1f}s", "tokens", sum(len(r["token_ids"]) for r in rows),
              flush=True)
        save()

if "graphsel" in phases:
    # Host-side cost of the per-step choice and of holding the stock graph for one forward (no device work).
    from vllm_torchtpu.compilation import moe_stock_graph as _msg  # noqa: E402
    _raw_sg = os.environ.get("TPU_MOE_SKIP_STOCK_GRAPH")
    spec = _msg.StockGraphSpec.parse(_raw_sg) if _raw_sg is not None else None
    if spec is None:
        after_start = (result.get("workers") or {}).get("after_start")
        if isinstance(after_start, list) and after_start and isinstance(after_start[0], dict):
            buckets = (after_start[0].get("stock_graph") or {}).get("buckets")
            if isinstance(buckets, dict) and buckets:
                spec = _msg.StockGraphSpec({int(k): int(v) for k, v in buckets.items()})
    if spec is None:
        spec = _msg.StockGraphSpec.parse("16:16,256:256")

    class _Wrapper:
        aot_compiled_fn = None

    disp = _msg.StockGraphDispatch(_Wrapper(), object(), spec)
    reps_n = 200_000
    timing = {}
    for name, (v, p) in {
        "moeskip_step_15": (15, 16),
        "stock_step_16": (16, 16),
        "moeskip_step_255": (255, 256),
        "stock_step_256": (256, 256),
    }.items():
        s = time.perf_counter_ns()
        for _ in range(reps_n):
            if disp.choose(v, p):
                with disp.use():
                    pass
        timing[name] = round((time.perf_counter_ns() - s) / reps_n, 1)
    result["graphsel"] = dict(seq=[int(x) for x in a.graphsel_seq.split(",")], steps=a.graphsel_steps,
                              select_ns_per_step=timing, passes=[])
    sp = SamplingParams(temperature=0.0, max_tokens=a.graphsel_steps + 1, ignore_eos=True)
    for p in range(2):
        batches = []
        for n in result["graphsel"]["seq"]:
            before = probe()
            s = time.perf_counter()
            llm.generate(prompts[:n], sp, use_tqdm=False)
            batches.append(dict(n=n, seconds=round(time.perf_counter() - s, 3), before=before, after=probe()))
            print("GRAPHSEL", p, n, f"{batches[-1]['seconds']:.2f}s", flush=True)
        result["graphsel"]["passes"].append(batches)
        save()

if probe_workers:
    result["workers"]["at_end"] = probe()
    save()

result["done"] = True  # every requested phase finished (check_arm.py checks the contents)
save()
print("DONE", a.label, flush=True)
os._exit(0)  # the engine can hang on shutdown and keep the TPU; results are written

# [XLA:TPU v7x] Small additive HLO change (+62 instructions, ~1%) re-packs instruction-memory overlays: overlay waits 57 → 320 per step and +290–310 µs/step on unchanged ops (Qwen3.5-35B-A3B decode, 16-row program; opposite effect in the 32-row program)

> **Public copy for [vllm-torchtpu #3350](https://github.com/vllm-project/vllm-torchtpu/pull/3350).** Exported from our
> internal notes on 2026-11-18. Host names are pseudonymized (`host-eNN`), internal storage paths, registry and cluster
> names are replaced by placeholders (`G` = our internal artifact store), and raw traces, HLO dumps and logs are not
> included. Numbers and text are otherwise unchanged. The report was drafted for the XLA TPU team and has not been filed.

> **DRAFT. Not filed.** Written only from existing evidence (no new TPU runs). Every number cites the file it
> comes from. Paths are relative to this directory; `G/...` paths point to our internal artifact store (not public).
> What could not be verified, and where the source doc and the files disagree, is in [NOTES.md](NOTES.md).

## 1. Summary

On one TPU v7x host (2x2x1, 4 chips, TP=4; PJRT/libtpu `TFRT TPU7x, Built on Oct 7 2026 00:47:01 (1791359221)`),
a vllm-torchtpu patch ("MoeSkip") makes the Qwen3.5-35B-A3B decode step 4–5% slower in the full 16-row bucket (wall clock:
+4.0% vs main with the flag set, +4.2–4.9% vs main; §6 E1). The patch
zeroes the routing weights of padded rows with one `torch.where` on a `[16, 8]` tensor per MoE layer. With 16 valid rows nothing
is padded and nothing is skipped. In the compiled 16-row forward program the change is purely additive:
- 62 new top-level instructions: one `pred[4096]` parameter, one `pred[16]` slice, and 30 `copy-start`/`copy-done` pairs that
  move the 16-element mask into memory space `S(1)`;
- one extra operand plus a `select` inside each of the 40 per-layer `compare_and_fusion` loop fusions.

The other ~5,700 top-level instructions keep their opcode, order, shape and memory-space annotation. The compile flags, compile
options and topology are byte-identical between the two variants. Even so, unchanged ops get 1.9–3.9 µs slower each, at
fixed instance positions: the TP all-reduce, `select_reduce_fusion`, and the GatedDeltaNet conv kernel. Together they add
**+309.3 / +288.5 µs per step** (two hosts), while the new ops cost +1.8 µs per step.

With `--xla_tpu_show_overlay_waits_in_profiler=true`, instruction-memory **overlay waits go from 57 to 320 per step**, with
identical counts on all three hosts where both variants were profiled with default overlay options. The extra wait time is
+227.1 to +227.7 µs per step, 71–79% of the extra op time, and it falls in exactly the ops that slowed down. The same source change
in the **32-row program moves the waits the other way (173 → 73)**, and that program gets 1.3–1.6% faster. Two libtpu flags were
tried:
- `--xla_tpu_merge_small_overlays_into_big_neighbors=false` has no effect.
- `--xla_tpu_overlay_enable_hlo_function_packs=false` removes the difference between the variants (forward −0.06% at n=16,
  −0.14% at n=32), but leaves every program much slower: for example the unpatched `base_on` program is 8–10% slower
  (cross-host comparison).

Rebasing onto a newer vllm-torchtpu main changes nothing; that run used the same libtpu build. Our working hypothesis is that
overlay packing is sensitive to small changes in code layout. We have not proven that inside the compiler (§11). We would like the
XLA TPU team's view on whether this is expected, and on how to keep the packing stable.

## 2. Impact

- **It blocks an upstream fix.** The MoeSkip bf16 patch has to show no regression before upstream will accept it, and the
  full-16-row case is a stable regression. Per-decode-step wall clock, final patch vs main on the same host:
  - +4.5% / +4.2% (`results/t3/g{2,3}/*_q*.json.gz`: 7.387 vs 7.067 ms and 7.377 vs 7.079 ms, 5 repeats each);
  - +4.90% / +4.54% (`results/t8/p2_t8.txt`).

  Against main with only the flag set, which isolates the zeroing change, it is +4.03% / +3.99% (`results/t14/p2_t14.txt`).
  The same patch speeds up padded steps by up to 24% (n=1: −24.13% / −23.51%, `results/t8/p2_t8.txt`).
- **The cost is not limited to n=16.** n=8 runs the same compiled 16-row program: the trace shows the same module id
  (`results/t4/align_r{1,2}_n8.txt`). The non-GMM ops there pay the same +306.2 / +285.3 µs per step
  (`results/t4/t4_summary.txt`), which the GMM savings hide. So every step with 1–16 active requests pays it. In the replayed
  R3 rollout length distribution ([`test/rollout_lengths.json`](test/rollout_lengths.json)), 1,434 of 1,999 decode steps (71.7%) have 1–16 active
  requests. This share is derived from the file and was not measured.
- **Concern for others (hypothesis):** if any small HLO change near this size can move overlay packing by ±100–300 µs per step,
  then A/B comparisons of unrelated model or kernel changes on large TPU decode programs can be confounded by about ±5%.

## 3. Environment

| Item | Value | Source |
|---|---|---|
| TPU | TPU v7x (`tpu7x`), one host, topology `2x2x1`, 4 chips (`google.com/tpu: 4`), GKE cluster `<cluster>` (shared `default` queue, spot) | `test/jobset.yaml.tmpl` (internal launcher, not included), `results/t4/arms_*.txt` (`TPU_TOPOLOGY=2x2x1`, `TPU_ACCELERATOR_TYPE=tpu7x`) |
| Process layout | vLLM in-process engine (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), TP=4, one worker process per chip (`TPU_MULTIHOST_BACKEND=mp`, `TORCH_TPU_TOPOLOGY=1,2,1,2`; log: `jax local_devices=1 global devices=4`) | [`test/run_pod.sh`](test/run_pod.sh), [`test/run_arm.sh`](test/run_arm.sh), `G/t4/g0/final_on_r1.log` |
| PJRT / libtpu build | `TFRT TPU7x` / `Built on Oct 7 2026 00:47:01 (1791359221)` / `C API version: 0.116`. TorchTPU tier-2 cache "binary fingerprint" `8ae752df13e66ac8` ("based on TorchTPU binary version 5 and PjRt version"). **Identical in every t4/t5/t6/t7/t10/t12 engine log**, including t10, which uses a newer image | `G/t*/g*/*.log`, `evidence/environment_excerpts.txt` |
| pip packages | libtpu 0.0.50, jax 0.11.2, jaxlib 0.11.2, torch 2.13.0+cpu, torch_tpu 0.2.0, vllm 0.33.0+empty (reports 0.33.0), vllm-torchtpu-core 0.0.0 | `G/t3/g0/pip.txt` and `G/t2/g0/pip.txt`. Recorded only in the t2/t3 images, which are built from the same base image as t4–t7 and t12 (NOTES §1) |
| Base image | `vllm/vllm-torchtpu@sha256:d88fcf418f32aaad79916c25498629ac4732cdfcc6faf8f6aab7e0ccc446dfac` (`nightly-20261114-f640d73-v0.33.0`, according to the Dockerfile comment) | [`test/Dockerfile`](test/Dockerfile) |
| Test images | t4/t5/t6: `<internal-image>@sha256:2b7d95b9fc83c854eebfcdd7fb3895e301eb94a71179fbc352019c9c456a2a91`; t7: `@sha256:06a244f0af28a012debc54dae458521f8ec7beefb1bd7d78710160085bdf2173` (adds `/opt/arm-vmul`); t12: `@sha256:d8e2e7225f0de13682734322a99f9df593e675564abbe7fea13b7dc4274f2894` (the t11 image); t10: `@sha256:f14bd68e35bf1bcbe3b93e90ae8d0e3a3beb6912af0d371d832a0924a9540723` (base `vllm/vllm-torchtpu@sha256:5e3c74f8f9905cf4a3717d115f9d7705845c12a24d0824247abc5876a0db44d9`, `nightly-20261115-b7c8b71-v0.33.0`) | `<workdir>/image_tag_t4`, `image_tag_t7`, `image_tag_t10`; launch-time JobSet specs `/tmp/mskip-t*-g*.yaml` (excerpts in `evidence/environment_excerpts.txt`); `<workdir>/build2/Dockerfile` |
| vllm-torchtpu | **base** `f640d734` (the image's own commit, `/opt/arm-base`); **candidate** `7e4d170a` = `f640d734` + `5f53fe77` (0001, zeroing) + `7e4d170a` (0002, the cached flag passed to kernel prebuild/selection) at `/opt/arm-final`. t10: base `b7c8b71e`, candidate = `b7c8b71e` + cherry-picks `507d06c8`, `e9cf6cf3` (same diff). Each run's JSON records source hashes (sha256, first 16 hex): `unquantized.py` base `5135462ae72a39a0` / candidate `36bf1126f10ecc22` (also in t10); `token_padding.py` `51fb4532d9e799e3`; `fused_moe.py` `53814294ce38b39e` | [`0001`](../patches/zero-weight-on-f640d734/), [`0002`](../patches/zero-weight-on-f640d734/), [`test/Dockerfile`](test/Dockerfile), `<workdir>/build2/Dockerfile`, `G/t*/g*/*.json` (`meta.sha256_16`) |
| Model | Qwen/Qwen3.5-35B-A3B (HF revision `59d61f3c…`, text-only checkpoint), bf16. 40 layers: `full_attention` every 4th layer (indices 3, 7, …, 39), the other 30 `linear_attention` (GatedDeltaNet); hidden 2048; 256 routed experts, top-8, `moe_intermediate_size` 512, plus a shared expert | `inputs_manifest.json` and `config.json` under `<inputs>` (config sha256 `4df6b696…` matches the manifest) |
| Engine args | `LLM(tensor_parallel_size=4, max_num_seqs=256, max_model_len=4352, max_num_batched_tokens=4096, gpu_memory_utilization=0.70, seed=0, max_logprobs=1)`; dtype bf16, prefix caching and chunked prefill on (engine config log line) | [`test/mskip_engine.py`](test/mskip_engine.py), `G/t5/g0/final_on_w1.log` |
| Compile configuration of the forward module | `num_partitions: 4`, `use_spmd_partitioning: true`, `use_shardy_partitioner: true`; one `env_option_overrides`: `xla_tpu_vmem_scavenging_mode = SAFE_IFF_KERNEL_REQUIRED`. Overlay defaults in the flagfile: `--xla_jf_always_overlay=true`, `--xla_tpu_overlay_enable_hlo_function_packs=true`, `--xla_tpu_merge_small_overlays_into_big_neighbors=true`, `--xla_tpu_single_overlay_mode=false`, `--xla_jf_internal_prefetch_overlays=true`, `--xla_tpu_last_overlay_prefetches_first=true` (full list in the excerpts file) | t4 HLO dump bundles; `evidence/environment_excerpts.txt` |

## 4. The trigger (source and HLO)

**Source change** ([`0001`](../patches/zero-weight-on-f640d734/0001-moe-skip-padded-tokens-unquantized.patch)). After routing, the bf16 MoE method calls
`token_padding.zero_routing_weights_for_padding`, which does
`topk_weights = torch.where(is_padding[:, None], torch.zeros_like(topk_weights), topk_weights)`. Here `is_padding` is a slice of
a persistent `bool[4096]` device mask (`TokenPaddingState.local_padding_mask`). Source: `<workdir>/build/arm-base/src/vllm_torchtpu/layers/adapter/token_padding.py`.
The GMM kernel variant that skips zero-weight routes (`skip_padded_tokens=True`) is used both with and without the patch,
whenever `TPU_MOE_SKIP_PADDED_TOKENS=1`.

**Variants compared** (all with fresh processes and fresh compile caches):
- **`base_on`**: main plus `TPU_MOE_SKIP_PADDED_TOKENS=1`. The skip kernel is selected and the runner updates the mask every
  step, but nothing is zeroed. This is the **reference for the compiler question**, because it differs from `final_on` only by
  the zeroing.
- **`final_on`**: the candidate plus the flag.
- **`base_off`**: main without the flag. It also differs in kernel variant and in KV-cache size: 2,450,944 vs 2,450,688 tokens
  (`G/t4/g0/*_r1.log`), which is 9,574 vs 9,573 cache blocks in the HLO parameter shapes. **Do not use it as the minimal pair.**

**Optimized HLO diff of the 16-row forward module** (`tt_jit_ir_ops_L89C0_fused_add_rms_norm_to_copy_copy_`, t4 g0 dumps,
`base_on_r1` `module_1082` vs `final_on_r1` `module_0340`; `results/t4/hlo_diff_b16.txt`, `results/t4/hlo_diff_b16_fusions.txt`,
order-independent check in `evidence/hlo_entry_opdiff.txt`):

| | `base_on` | `final_on` | Difference |
|---|---:|---:|---|
| ENTRY instructions | 5,730 | 5,792 | +62: 30 `copy-start (pred[16]@S1, pred[16], u32[]@S2)`, 30 `copy-done pred[16]@S1`, 1 `parameter pred[4096]`, 1 `slice pred[16]` (`as_strided`) |
| Modified instructions | | | 40 `compare_and_fusion` kLoop fusions with result `(pred[16,8]@S1, pred[16,8]@S1, bf16[16,8]@S1)`. Each gains one operand: the `copy-done` in 30 layers, the slice directly (without an `S(1)` copy) in 10 layers. Inside, each gains `parameter pred[16]`, `broadcast pred[16,8]` and `select(mask, 0, weights)` |
| Which 10 layers have no prefetch copy | | | The 4th, 8th, …, 40th `compare_and_fusion` in program order. These positions coincide with the 10 `full_attention` layers of `config.json` (why they get no copy is not known) |
| Ordering-only difference | | | `weights_ref_gdn_dt_bias = bf16[8] parameter()` is listed at a different position (one `-`/`+` pair in `hlo_diff_b16.txt`) |
| XLA ops per step in the trace | 4,747 | 4,808 | +61 = 30 + 30 + 1 (`results/t4/align_r1_n16.txt`) |
| `optimized_hlo_instruction_count` (all computations) | 27,033 | 27,215 | +182 (`*.basic_compiler_metadata.txt` in the dumps) |

**32-row module** (`module_1249` vs `module_0509`, `results/t4/hlo_diff_b32.txt`): 5,820 → 5,902 ENTRY instructions. The
change has the same form, with a prefetch copy in all 40 layers: 40 `copy-start` + 40 `copy-done` + 1 parameter + 1 slice, and
40 modified fusions.

**Before optimization** (`module_1079` vs `module_0337` for 16 rows, `module_1246` vs `module_0506` for 32 rows;
`evidence/hlo_entry_opdiff.txt`): +243 ENTRY instructions (13,686 → 13,929 and 13,676 → 13,919). This is 40 × {`slice pred[16]`,
`reshape pred[16]`, `reshape pred[16,1]`, `broadcast pred[16,1]`, `broadcast pred[16,8]`, `select bf16[16,8]`}, plus one
`broadcast bf16[16,8]`, one `constant bf16[]` and one `parameter pred[4096]`. Nothing else differs in the (opcode, shape) multiset.

**Identical compile environment.** Between `base_on` and `final_on`, for both buckets, these files are byte-identical (same
sha256; see `evidence/hlo_module_bundle_sha256.tsv`): `flagfile`, `tpu_comp_env.txt`, `execution_options.txt`,
`target_arguments.txt`, `compile_options.textproto` and `topology.textproto`. `debug_options` differs only in `xla_dump_to`, and
`hlo_module_config.txt` differs only in the extra parameter (`p601`, `pred[4096]`) and `xla_dump_to`.

**The compiled program is the same on both hosts.** The 16-row forward module carries the same trace module id on both t4 hosts:
`base_on` `(14512450523834468197)`, `final_on` `(14121214978197194209)` (`results/t4/align_r{1,2}_n16.txt`).

## 5. Reproduction (what we ran)

**Harness.** These steps use the experiment-repo scripts in [`test/`](test/). Commit `45dace7` holds the results and doc. The worktree
has uncommitted t15 additions, which do not touch the profile phase. The scripts that actually ran are the copies baked into each
image at `/opt/measure`. `launch.sh` applies the JobSets with `kubectl` to the `<cluster>` cluster context. Elsewhere,
use `jobset.yaml.tmpl` on any GKE node pool with `cloud.google.com/gke-tpu-accelerator: tpu7x` and
`cloud.google.com/gke-tpu-topology: 2x2x1`.

```bash
IMAGE=<internal-image>@sha256:2b7d95b9fc83c854eebfcdd7fb3895e301eb94a71179fbc352019c9c456a2a91
# launch.sh IMAGE RUN_ID PLAN: one single-host JobSet (v7x 2x2x1, 4 chips) per group; results go to G/<RUN_ID>/<group>/.
# Use a NEW RUN_ID: arms with a .done file under an existing prefix are skipped.
bash launch.sh "$IMAGE" <new_run_id> t5   # overlay-wait traces: base_on and final_on at n=16 and n=32, two hosts, opposite order
bash launch.sh "$IMAGE" <new_run_id> t4   # the same without markers, plus base_off; g0 also dumps HLO (MSKIP_XLA_DUMP=1)
bash launch.sh "$IMAGE" <new_run_id> t6   # the two libtpu overlay options, one per host
```

**What one arm runs inside the pod** (see `run_pod.sh` and `run_arm.sh`). Arms on a host run one after another. Each uses all 4 chips.

```bash
# run_pod.sh: unset TPU_WORKER_*, TPU_HOST_BOUNDS, TPU_CHIPS_PER_HOST_BOUNDS, TPU_TOPOLOGY*, TPU_PROCESS_*, MEGASCALE_*, then:
export TPU_ACCELERATOR_TYPE=tpu7x TPU_MULTIHOST_BACKEND=mp TORCH_TPU_TOPOLOGY=1,2,1,2
# inputs: <inputs> -> /data (fetch_inputs.py checks the sha256 of each file in inputs_manifest.json)
# run_arm.sh, per arm (fresh process, its own HOME / TMPDIR / XDG_CACHE_HOME, so its own compile caches):
unset TPU_MOE_SKIP_PADDED_TOKENS VLLM_TPU_FAST_TOKEN_SUBSTITUTION USE_PHASED_PROFILER TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT VLLM_ENABLE_V1_MULTIPROCESSING
export PYTHONPATH=/opt/arm-final/src:/workspace/vllm-torchtpu     # final_on; base_on uses /opt/arm-base/src
export TPU_MOE_SKIP_PADDED_TOKENS=1                                # base_off: leave unset
export VLLM_ENABLE_V1_MULTIPROCESSING=0                            # MSKIP_INPROC=1
export LIBTPU_INIT_ARGS="--xla_tpu_show_overlay_waits_in_profiler=true"   # t5/t7/t10/t12; t6 appends one overlay option; t4: unset
# t4 g0 only: export XLA_FLAGS="--xla_dump_to=/out/hlo/<label> --xla_dump_hlo_as_text"
python3 /opt/measure/mskip_engine.py --label <label> --out /out --phases profile --profile-ns 16,32 --profile-steps 32
```

**Profile phase** (`mskip_engine.py`). For each n: two warm-up `generate` calls on the first n GSM8K prompts
(`max_tokens=17`, `ignore_eos`). Then `llm.start_profile(profile_prefix="n<n>")`, sleep 1 s, one `generate` with `max_tokens=33`,
`ignore_eos`, greedy, and `llm.stop_profile()`. All n requests start and finish together, so every one of the 32 decode steps has
n valid rows. Traces go to `<label>/n<n>/plugins/profile/<ts>/rank{0..3}_*.xplane.pb`, and the torch traces to
`<label>/n<n>/rank_<k>/*.pt.trace.json.gz`.

**How to compare** (scripts in [`test/`](test/); they need protobuf):
- Use only rank 0, `/device:TPU:0`. The forward module is the module with the most device time among modules executed at least
  20 times (`tt_jit_ir_ops_L89C0_fused_add_rms_norm_to_copy_copy_`). One execution is one decode step. Skip the first 2 of 32
  steps and use the remaining 30.
- Overlay waits: `prof_overlay.py A.xplane.pb B.xplane.pb --match overlay-wait` counts the `overlay-wait` events on the
  `XLA TraceMe` line inside each step: the median count per step and the mean time per step. Each wait is attributed to the op
  running when it starts. `t5_summary.py RAW out.json --pairs base_on_w1:final_on_w1,...` produces the summary tables.
- Existing-op time: `prof_align.py` / `t4_summary.py` align the per-step op sequences by instruction name (numeric suffixes
  removed, plus shape) and sum the duration change of the aligned ops ("aligned-op Δ"). Module medians include idle gaps, which
  vary between runs (t4 n=16: −38.9 / +3.8 µs). The aligned-op Δ is the more robust metric.
- HLO: `hlo_layer_diff.py <base_on 16-row>.after_optimizations.txt <final_on 16-row>.after_optimizations.txt --layer -1 --context 0`,
  plus `evidence/entry_opdiff.py` for the order-independent multiset diff.
- Always compare `base_on` and `final_on` **on the same host**. All per-step numbers below are per execution of the forward module.

## 6. Evidence

### E1. Wall clock: only the full 16-row bucket regresses

Per decode step, `final_on` vs main `base_off`, same host (`results/t8/p2_t8.txt`, 5 timed repeats per point, A/A noise on g0
at n=16: 0.13%):

| n (bucket) | 1 (16) | 8 (16) | 12 (16) | 15 (16) | **16 (16)** | 17 (32) | 32 (32) | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| g0 / g1 | −24.13 / −23.51% | −10.19 / −10.28% | −4.50 / −4.46% | +1.57 / +1.38% | **+4.90 / +4.54%** | −14.30 / −14.23% | −0.53 / −0.41% | −1.15 / −1.24% | −0.80 / −0.78% | +0.56 / +0.56% |

The zeroing change alone, `final_on` vs `base_on` on the same host (`results/t14/p2_t14.txt`):
- n=16: +4.03% / +3.99%
- n=32: −1.28% / −1.44%
- n=128: −0.01% / −0.11%
- n=256: −0.02% / −0.00%
- A/A: −0.11% / −0.11% at n=16

### E2. Device time per step: all the extra time is in ops that did not change (t4, no markers)

Source: `results/t4/t4_summary.txt`. Two hosts in opposite order: r1 = `host-e01`, r2 = `host-e02`.
Values are `final_on` − `base_on` in µs per step over 30 steps.

| n | Program | Forward median (µs) | Aligned existing ops | of which GMM | Other existing ops | Idle gaps | New/removed ops |
|---:|---|---|---:|---:|---:|---:|---:|
| 16 | 16-row | r1 5,742.1 → 5,989.5 (+4.31%); r2 5,696.2 → 5,986.4 (+5.09%) | **+309.3 / +288.5** | +3.1 / +2.2 | +306.2 / +286.3 | −38.9 / +3.8 | +1.8 / +1.8 |
| 8 | 16-row (same module id) | r1 5,893.6 → 5,574.3; r2 5,918.9 → 5,173.3 | −768.8 / −791.3 | −1,075.0 / −1,076.6 | **+306.2 / +285.3** | +431.3 / +163.6 | +1.8 / +1.8 |
| 32 | 32-row | r1 8,072.9 → 7,962.0 (−1.37%); r2 8,088.5 → 7,959.8 (−1.59%) | −111.5 / −131.5 | −7.8 / −9.4 | **−103.7 / −122.1** | +26.6 / −10.8 | +2.0 / +2.0 |

Per op kind, n=16, r1 / r2 (`results/t4/t4_summary.txt`, `align_r{1,2}_n16.txt`). The extra time is concentrated in fixed
instances:

| Op | Instances per step | Slow instances `base_on` → `final_on` (slow = more than 1.15× the `base_on` median) | Fast / slow instance | Per-step sum r1; r2 |
|---|---:|---:|---|---|
| `distributed_all_reduce bf16[16,2048]` (TP all-reduce) | 81 | 6 → 31 | 10.53–10.73 / 14.44 µs | 867.2 → 982.8; 891.0 → 982.5 µs |
| `select_reduce_fusion` (per-expert group sizes) | 40 | 0 → 32 | 0.69 / 2.58–2.60 µs | 27.6 → 81.1; 27.6 → 81.3 µs |
| `fused_conv1d_gdn_batched` (GatedDeltaNet conv) | 30 | 5 → 24 | 12.59 / 15.03–15.06 µs | 390.7 → 438.3; 390.8 → 438.4 µs |

- The slow positions are **identical on both hosts**, for both variants (`results/t4/slow_positions_n16.txt`).
- An instance varies across the 30 steps by at most 0.45 / 0.65 µs (`select_reduce_fusion`) and 1.29 / 0.85 µs (conv).
- Other kinds that gained time: TP all-reduce (aligned) +113.2 / +92.1 µs, `fusion bf16[16,2048]` +32.1 / +32.7 µs,
  `as_strided bf16[16,4,192]` +21.8 / +21.9 µs; `router_topk` lost 12.5 µs. The modified `compare_and_fusion` itself costs
  +6.8 µs per step for its 40 instances (`align_r{1,2}_n16.txt`).
- The extra time is spread over the whole program: 35 of the 40 layer segments gain more than 3 µs per step (at most +18.2 µs), and
  two lose 5–7 µs (`results/t4/align_r1_n16.txt`).
- In the 32-row program, slow all-reduce instances go 10 → 5, and the other two kinds stay at 0 → 0.

### E3. Overlay waits explain most of the extra time (t5, t7 g0, t6 g0)

With `LIBTPU_INIT_ARGS=--xla_tpu_show_overlay_waits_in_profiler=true`, same image and source as t4; `base_on` and `final_on` run on
the same host. Our understanding of the flag is that the overlay wait routine emits a trace marker when it has to wait for an IMEM
overlay DMA. We have not verified whether it changes the generated code; see NOTES §1 item 5.

| Run / host | Overlay options | n=16 waits/step | n=16 wait time (µs/step) | n=16 aligned-op Δ | Wait share | Forward median n=16 (µs) | n=32 waits/step | n=32 wait time | n=32 aligned-op Δ | Forward median n=32 (µs) |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| t5 g0, `host-e02` | default | **57 → 320** | 67.8 → 295.5 (+227.7) | +315.5 | 72% | 5,799.9 → 6,007.3 | **173 → 73** | 152.0 → 59.5 (−92.5) | −109.4 | 8,082.6 → 7,969.3 (−1.40%) |
| t5 g1, `host-e03` | default | **57 → 320** | 68.0 → 295.6 (+227.6) | +287.3 | 79% | 5,775.0 → 6,111.8 | **173 → 73** | 152.0 → 59.6 (−92.4) | −137.3 | 8,096.8 → 7,966.3 (−1.61%) |
| t7 g0, `host-e04` | default | **57 → 320** | 68.1 → 295.2 (+227.1) | +321.5 | 71% | 5,857.1 → 5,997.6 | **173 → 73** | 151.9 → 59.5 (−92.4) | −107.9 | 8,066.9 → 7,961.0 (−1.31%) |
| t6 g0, `host-e01` | `merge_small_overlays_into_big_neighbors=false` | 57 → 320 | 68.0 → 295.3 (+227.3) | +323.6 | 70% | 5,760.2 → 5,997.4 | 173 → 73 | 151.7 → 59.4 (−92.3) | −101.7 | 8,065.0 → 7,961.6 (−1.28%) |

Sources: `results/t5/t5_summary.txt` and `.json`; `evidence/t7_g0_summary.txt` (copied from `<workdir>/results/t7_g0_summary.txt`;
the worktree `results/t7/` has only g1); `results/t6/t6_summary.txt`.

Where the extra waits fall, n=16, t5 g0 (`results/t5/overlay_w1_n16.txt`, `t5_summary.txt`; t5 g1 has the same wait counts, and
wait times within 0.2 µs):

| Op | Waits/step `base_on` → `final_on` | Wait time Δ | Duration Δ of the op |
|---|---:|---:|---:|
| `distributed_all_reduce bf16[16,2048]` | 7 → 32 | +94.4 µs | +120.3 µs (g1 +91.4) |
| `fused_conv1d_gdn_batched_b4_c1 bf16[16,8,128]` | 8 → 27 | +46.0 µs | +47.0 µs |
| `select_reduce_fusion.lhs_group_sizes_ref s32[256]` | 0 → 30 | +43.9 µs | +54.4 µs |
| `fusion bf16[16,2048]` | 0 → 26 | +24.8 µs | +32.4 µs |
| `as_strided bf16[16,4,192]` | 0 → 10 | +19.3 µs | +21.9 µs |
| `router_topk f32[16,8]` | 3 → 1 | −12.2 µs | −12.5 µs |

- **The counts are fixed by the program.** In the 30 analysed steps there are exactly 1,710 = 30 × 57 and 9,600 = 30 × 320
  `overlay-wait` events. At n=32 there are 5,190 = 30 × 173 and 2,190 = 30 × 73 (`overlay_w1_n16.txt`, `overlay_w1_n32.txt`).
- **The same counts appear on more hosts.**
  - `final_on` alone measured 320 on a fifth host: t7 g1, `host-e05` (`results/t7/t7_g1_summary.txt`).
  - The newer-main run (E7) measured 57 and 320 on two more hosts.
- **The markers do not change the effect.** With markers, existing ops gain +315.5 / +287.3 µs at n=16, against
  +309.3 / +288.5 µs without them (t4).
- **Part of the extra time is not covered by the markers.** The waits account for 71–79% (n=16) and 67–86% (n=32; t5 and t7 g0)
  of the aligned-op change. The rest is not covered by the markers; our hypothesis is overlay-switch overhead outside the DMA wait,
  which we have not measured.
- **In the 32-row program the waits go away in different places**, mainly: `fusion bf16[32]` 21 → 0 (−44.1 µs), all-reduce
  13 → 7 (−24.6 µs), conv 16 → 8 (−19.0 µs), `fusion bf16[32,2048]` 11 → 0 (−9.8 µs). One kind gains: `sort s32[256]` 0 → 7
  (+8.2 µs) (`overlay_w1_n32.txt`).

### E4. Other full buckets (t12, n = 64 / 128 / 256)

Source: `results/t12/t12_g1_summary.txt`. Same host `host-e06` (t12 g1), markers on; `final_on` vs `base_on`.

| n (bucket) | Waits/step | Wait time (µs/step) | Aligned-op Δ (µs) | Forward median (µs) |
|---:|---:|---:|---:|---|
| 16 (t5, for reference) | 57 → 320 | +227.7 / +227.6 | +315.5 / +287.3 | +3.58% / +5.83% |
| 32 (t5, for reference) | 173 → 73 | −92.5 / −92.4 | −109.4 / −137.3 | −1.40% / −1.61% |
| 64 | 216 → 186 | 152.9 → 100.1 (−52.8) | −107.7 | 10,893.1 → 10,784.0 (−1.00%) |
| 128 | 343 → 402 | 210.1 → 207.5 (−2.6) | −5.5 | 15,707.9 → 15,708.1 (+0.00%) |
| 256 | 484 → 501 | 372.2 → 426.1 (+53.9) | −19.6 | 23,575.5 → 23,605.8 (+0.13%) |

The same source change moves the wait count in every bucket (+263, −100, −30, +59, +17), in both directions. The count and the
cost do not move together: at n=128, 59 more waits cost 2.6 µs less. The t12 g0 pair (`host-e07`) completed but has
not been analysed (NOTES §3).

### E5. libtpu overlay options (t6)

Source: `results/t6/t6_summary.txt`. Same image; each host tests one option, with markers on.

| Option (one host each) | n=16 waits `base_on` → `final_on` | n=16 aligned-op Δ | n=16 forward | n=32 waits | n=32 forward |
|---|---:|---:|---|---:|---|
| `--xla_tpu_merge_small_overlays_into_big_neighbors=false` (`host-e01`) | 57 → 320 | +323.6 µs | 5,760.2 → 5,997.4 (+4.12%) | 173 → 73 | 8,065.0 → 7,961.6 (−1.28%) |
| `--xla_tpu_overlay_enable_hlo_function_packs=false` (`host-e08`) | **1,067 → 1,066** (377.4 → 380.5 µs) | **−5.9 µs** | **6,324.4 → 6,320.4 (−0.06%)** | 1,277 → 1,276 (502.5 → 504.5 µs) | 8,783.9 → 8,771.2 (−0.14%) |

With function packs disabled the two variants are equal, so the zeroing itself costs almost nothing. But every program is slower.
These comparisons are cross-host, because the packs=false host has no run with default options:
- `base_on` with packs=false is 6,324.4 µs at n=16, against 5,775.0–5,857.1 µs with default options on the t5 and t7 g0 hosts
  (5,760.2 on the t6 g0 host): +8.0% to +9.8%.
- `base_on` at n=32: 8,783.9 against 8,065.0–8,096.8 µs: +8.5% to +8.9%.
- `final_on` with packs=false is 3.4–5.4% slower at n=16 than `final_on` with default options (5,997.4–6,112.8 µs), and
  10.1–10.2% slower at n=32 (7,961.0–7,969.3 µs).

### E6. A semantically equivalent code change does not help (t7, multiply by a keep-mask instead of `torch.where`)

Sources: `evidence/t7_g0_summary.txt`, `results/t7/t7_g1_summary.txt`. `vmul_on` = `7e4d170a` plus the multiply variant
(`token_padding.py` sha256 `7598c550f25294f2`).

| | n=16 waits | n=16 vs `final_on` | n=32 waits | n=32 forward vs `final_on` |
|---|---:|---|---:|---|
| t7 g0 (`base_on` → `vmul_on`) | 57 → 320 | wait time +0.7 µs, aligned-op −12.8 µs | 173 → 187 (`final_on`: 73) | +1.13% |
| t7 g1 (`final_on` → `vmul_on`) | 320 → 320 | wait time −0.4 µs, aligned-op −16.8 µs | 73 → 187 | +0.98% |

### E7. Newer vllm-torchtpu main (t10): unchanged, but the compiler is the same build

Image `@sha256:f14bd68e…`: base `b7c8b71e` vs base plus the same two commits (`507d06c8`, `e9cf6cf3`). The libtpu build string is
the same ``. Only one arm per host completed, so this is a **cross-host** comparison:
`base_on_x1` on `host-e09` and `final_on_x2` on `host-e10` (`results/t10/overlay_x1_x2_crosshost.txt`,
`G/t10/g*/arms_*.txt`).
- n=16: waits 57 → 320, 68.2 → 294.7 µs (+226.5 µs mean); module mean +289.9 µs.
- n=32: waits 173 → 73, 152.6 → 59.5 µs (means; −93.1 µs).
- The per-op attribution matches t5: all-reduce 7 → 32, conv 8 → 27, `select_reduce_fusion` 0 → 30, …

## 7. Conditions that keep the trigger (as observed); not minimized

We have **not** minimized this. Every observation so far comes from the full vLLM engine on the full model. These are the conditions
under which the packing change has been seen, and a minimization must keep all of them until it is shown that one is not needed:

1. **The full 40-layer forward module** (`tt_jit_ir_ops_L89C0_fused_add_rms_norm_to_copy_copy_`, about 5,700–5,900 top-level
   instructions after optimization), compiled as one 4-partition SPMD program for a v7x 2x2x1 host. It includes the cross-chip TP
   all-reduces, the GatedDeltaNet conv/state kernels, attention, and the GMM custom calls. **We have never tried a smaller program
   (fewer layers, one layer, smaller model).** Overlay packing is a property of the whole program's code layout, so deleting
   layers may delete the trigger. (Hypothesis, based on the extra time being spread over 35 of the 40 layer segments, E2.)
2. **Both buckets.** The 16-row program, where waits go 57 → 320 and the program gets slower, is the trigger. The 32-row program,
   where waits go 173 → 73 and it gets faster, is the control showing that the same edit moves packing in the other direction.
   A reduced reproducer should show both, or it may be reproducing something else.
3. **The minimal pair is `base_on` vs `final_on`.** Both use the skip-kernel variant and the same KV-cache shapes. The only
   difference is the mask parameter, the slice, the copies, and the 40 modified fusions.
4. **The same compiler and options:** libtpu ``, the dumped `flagfile`, `compile_options.textproto` (including the
   `xla_tpu_vmem_scavenging_mode` override), `topology.textproto`, and default overlay settings. With
   `xla_tpu_overlay_enable_hlo_function_packs=false` the difference disappears.
5. **Measurement:** waits counted with `--xla_tpu_show_overlay_waits_in_profiler=true` on rank 0 / `/device:TPU:0`, steady-state
   steps (skip 2). The counts were the same on every host, with real weights and the same 16 or 32 GSM8K prompts. **Whether they
   depend on the input data has not been tested** (hypothesis: they do not).

## 8. Proposed minimization plan (keeps the full program)

Goal: get a reproducer that needs no vLLM or torch_tpu runtime, without deleting the trigger.

1. **Bundles (done; no TPU needed).** The t4 g0 dumps already contain everything needed to recompile each variant outside vLLM.
   For each of `base_on` / `final_on` / `base_off` × 16 / 32 rows (module ids in an internal artifact inventory (not public) §2):
   `module_<id>/module.mlir`, `module_<id>/compile_options.textproto`, `module_<id>/topology.textproto`,
   `*.before_optimizations.txt`, `*.flagfile`, `*.debug_options`, `*.hlo_module_config.txt`, and `*.tpu_comp_env.txt`. Reference
   outputs: `*.after_optimizations.txt`, `*.after_codegen.txt`, `*-memory-usage-report.txt`.
2. **Compile-only replay (XLA team tooling).** Compile the four main bundles (`base_on` / `final_on` × 16 / 32) standalone with the
   same libtpu build, flags, compile options and topology. Check that the optimized HLO matches the dumps: ENTRY 5,730 / 5,792
   and 5,820 / 5,902 instructions, diffs of 62 and 82 instructions. If an internal dump of overlay or HLO-function-pack
   assignment exists, compare it between `base_on` and `final_on` here, without running anything.
3. **Execution replay on one v7x 2x2x1 host** (4 chips are needed: the program has 4 partitions and TP all-reduces). Run each
   compiled program about 32 times with synthetic inputs of the dumped parameter shapes, with
   `--xla_tpu_show_overlay_waits_in_profiler=true`. **Acceptance:** waits per execution of 57 vs 320 (16-row) and 173 vs 73
   (32-row), or at least the same direction and size. If synthetic inputs change the counts, record that and fall back to real
   weights. We do not have a parameter dump yet; it would need a new capture.
4. **Bisection inside the full program (all 40 layers and both buckets kept).** Starting from the `base_on` before-optimization
   module (or its `module.mlir`), apply the `final_on` edit (mask parameter + slice + per-layer `select`) to only some layers.
   Try: layers 0–19 vs 20–39, then halves of whichever half moves the count; the 10 full-attention positions (no prefetch copy)
   vs the 30 others. Recompile and count waits each time. This tells whether particular positions trigger the repacking or
   whether the total size or layout is what matters.
5. **Neutral perturbation controls (all 40 layers kept).** Add a perturbation of similar size that cannot be optimized away, for
   example 30 extra tiny async copies or an extra live output, without the select. Check whether that also moves the counts. This
   separates "this particular edit" from "any small layout change".
6. **Only after steps 4 and 5 show which parts are needed**, try removing layers, while checking at every step that both the
   16-row and the 32-row deltas persist. Stop and keep the previous size if a delta disappears.

## 9. What we ask the XLA TPU team

1. **Is this expected?** A purely additive change (+62 top-level instructions, about 1% of about 5,700, plus one operand and a
   `select` in 40 fusions) moves the overlay waits per execution from 57 to 320. The same edit moves them from 173 to 73 in the
   32-row program, and by −30 / +59 / +17 in the 64 / 128 / 256-row programs. Is this level of packing sensitivity known and
   accepted?
2. **What changed in the packing?** The compile bundles of both variants are provided. Can you tell which overlays or HLO-function
   packs changed between `base_on` and `final_on` in the 16-row program, and why the hot per-layer ops (TP all-reduce,
   `select_reduce_fusion`, GDN conv, `fusion bf16[16,2048]`) now wait at the same instances every step?
3. **Is there a supported knob that keeps packing stable**, or that prefetches or pins these overlays, without the 8–10% loss of
   `--xla_tpu_overlay_enable_hlo_function_packs=false`? The flagfile also lists `xla_tpu_single_overlay_mode`,
   `xla_jf_internal_prefetch_overlays`, `xla_tpu_last_overlay_prefetches_first`, `xla_tpu_overlay_allocation_table_size` and
   `xla_tpu_prefer_conditional_same_overlay`. We have not tried them and do not know what they do. Which, if any, apply here? Can
   they be set for one module through `compile_options.env_option_overrides`?
4. **What is the remaining 21–29% of the extra time at n=16** (and 14–33% at n=32; t5 and t7 g0) that the wait markers do not
   cover? Would `--xla_tpu_show_overlay_overhead_in_profiler` or `--xla_tpu_show_overlay_details_in_profiler` show it?
5. **Does a newer libtpu than `` behave differently** on these two bundles?
6. **Any guidance for framework code?** Are some HLO patterns known to perturb overlay packing more than others, for example tiny
   `pred` operands that memory-space assignment prefetches into `S(1)` with async copies? Is there a way to write such per-layer
   masking so that it does not change the code layout?

## 10. What we already tried

| Attempt | Result | Source |
|---|---|---|
| Semantically equivalent rewrite: multiply by a keep-mask (`vmul`) instead of `torch.where` | n=16 unchanged (57 → 320); n=32 worse than `final_on` (73 → 187 waits, +0.98% / +1.13%) | E6 |
| `--xla_tpu_merge_small_overlays_into_big_neighbors=false` | No effect (57 → 320, +323.6 µs) | E5 |
| `--xla_tpu_overlay_enable_hlo_function_packs=false` | The variants become equal, but all programs are much slower (`base_on` 8–10% at n=16 and n=32) | E5 |
| Rebase onto vllm-torchtpu main `b7c8b71e` (newer nightly image, same libtpu build) | Unchanged (57 → 320, 173 → 73) | E7 |
| Checked the cost of the zeroing itself | New ops +1.8 µs per step; modified fusions +6.8 µs per step; with packs off the variants differ by −0.06% | E2, E5 |
| Checked the compile environment | Flags, options and topology byte-identical between the variants | §4 |

A runtime workaround is being prototyped (vllm-torchtpu side): keep a stock, non-zeroing graph for full or nearly full 16-row
batches (`launch.sh` plan `t15`). It has no results yet and is out of scope here.

## 11. Hypotheses (not established)

- **H1, layout-driven packing.** The 30 async `copy-start`/`copy-done` pairs, the slice and the larger fusions shift instruction
  sizes and addresses. The overlay or function-pack partitioner then cuts the program differently, and some hot per-layer ops
  start in an overlay whose prefetch has not finished. Supporting evidence:
  - the wait counts are deterministic across hosts;
  - the waits sit in the ops that slowed down;
  - with function packs off the difference vanishes;
  - the direction flips between buckets.

  Not shown: there is no compiler-internal overlay dump.
- **H2.** The 21–29% of the extra time not covered by wait markers is overlay-switch overhead outside the DMA wait.
- **H3.** The wait counts depend only on the compiled program, not on the input data. Identical counts across hosts with the same
  inputs support this; other inputs were not tested.
- **H4.** The 10 layers whose fusions read the mask without an `S(1)` prefetch copy are the full-attention layers. The positions
  coincide (§4), but we do not know why memory-space assignment treats them differently, or whether it matters.

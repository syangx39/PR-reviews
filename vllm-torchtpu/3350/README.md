# vllm-torchtpu #3350: evidence for BF16 MoE padded-token skipping

[Upstream pull request](https://github.com/vllm-project/vllm-torchtpu/pull/3350) · head `16ed9b0a7` (commits `08a78448e`, `fe5cbee79`, `056cee01c`, `9d9fcb849`, `16ed9b0a7`) on main `6cc233d90`. `9d9fcb849` adds tests only; `16ed9b0a7` addresses the second review's nits (logging, docstrings, an explicit default argument, tests) and was rechecked in t27.

Raw results (gzip JSON written by [`tools/mskip_engine.py`](tools/mskip_engine.py) and the direct check), the analysis scripts, and their outputs. Host names are pseudonymized (`host-NN`, stable across experiments); nothing else in the result files was changed.

## Source trees

| Tree (label prefix) | Source | Used for |
|---|---|---|
| `upbase` | vllm-torchtpu `6cc233d90` (main, the PR's parent) | baseline in every comparison |
| `up6` | `975685ab2` (the PR head: `16ed9b0a7` + `padded_row_filter` / `row_info` input checks) | t28: unit tests and type check |
| `up5mc`, `up5nb` | `16ed9b0a7` with the count copied but no padding-mask write (1 blocking copy per step), and with the count copied non-blocking (0 blocking copies); experiment only | t28: copy cost and async-scheduling overlap |
| `up5` | `16ed9b0a7` (`9d9fcb849` + the second review's nits) | t27: unit tests and type check, flag-on and flag-off numerics, EP fallback, MTP drafter, XLA dumps vs `up3` |
| `up4` | `9d9fcb849` (`056cee01c` + TPU tests of the row-count kernel and custom op; same `src/`) | t26: unit tests and type check |
| `up3` | `056cee01c` (`fe5cbee79` + the first review's fixes) | t25: unit tests, direct MoE check, flag-on and flag-off numerics, GSM8K quality, EP fallback, MTP drafter, decode time |
| `up2` | `fe5cbee79` (the head before the review fixes) | t22b / t22c numerics, t23 / t24 performance, t25 MTP failure reproduction and before/after decode time |
| `up` | `ce4a30948`: the PR before the count was renamed to `local_valid_tokens`; the only difference from `fe5cbee79` is that rename ([diff](patches/ce4a30948-to-fe5cbee-rename.diff)) | t22 performance, served path, first numerics |
| `upbar` | `fe5cbee79` + [per-layer mask variant](patches/0001-Compute-the-valid-row-mask-per-MoE-layer-optimizatio.patch) (not in the PR) | t23 variant |
| `base`, `gz` | `15be35dad`, and `15be35dad` + [zero-weight approach](patches/gate-zero-on-15be35d/) | gate-zero comparison quoted in the PR |
| early `base`, `final`, `epfb` (t3–t16) | `f640d734` (main); `final` = `7e4d170a` (`f640d734` + zero-weight [0001, 0002](patches/zero-weight-on-f640d734/)); `epfb` = `586e11bc` (`final` + [0003](patches/zero-weight-on-f640d734/0003-moe-skip-stock-path-under-ep.patch), stock path under EP) | [compiler report](compiler_report/REPORT.md) (t3–t14), EP switching (t11, t16) |

Arms are labelled `<tree>_<on|off>_<rep>`; `on` means `TPU_MOE_SKIP_PADDED_TOKENS=1`. A tree name ending in `sy` (t28) runs with async scheduling off. A tree name ending in `k<blocks>` (t24, t25, t27, t28) pins the KV cache to that many blocks (`num_gpu_blocks_override`). The arms quoted in the PR:

| Arm | Source | Flag | Where |
|---|---|---|---|
| Baseline | `6cc233d90` | off | every group |
| Candidate off | `056cee01c` | off | `results/t25/q1-correctness-quality` (`up3_off_f1`) |
| Candidate on | `056cee01c` | on | `results/t25/*` (`up3_on_*`, `up3k*`) |
| Head | `16ed9b0a7` | on and off | `results/t27/*` (`up5_*`, `up5k*`) |

## Environment

- Image: `vllm/vllm-torchtpu@sha256:f1685130e5ec6e13c37d982345d6c34ce6b7763c74e982a14226208b77bf9bd7` (nightly-20261116-15be35d-v0.33.0; vLLM 0.33.0, torch 2.13.0, torch_tpu 0.2.0, JAX 0.11.2, libtpu 0.0.50) with each tree's `src/` copied to `/opt/arm-<tree>`.
- An arm runs `tools/mskip_engine.py` in a fresh process with its own `HOME` / `TMPDIR` (own compile caches), `PYTHONPATH=/opt/arm-<tree>/src:/workspace/vllm-torchtpu`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, and `TPU_MOE_SKIP_PADDED_TOKENS=1` for flag-on arms. Arms of one group run one after another on one TPU v7x `2x2x1` host, TP=4.
- Early runs (t3–t16: compiler report, EP switching) used the `nightly-20261114-f640d73-v0.33.0` image with the early trees overlaid; details in [compiler_report/REPORT.md](compiler_report/REPORT.md) §3. Host names there are pseudonymized as `host-eNN`.
- Model: Qwen3.5-35B-A3B, BF16 text weights (no MTP weights; the MTP runs use dummy weights). Prompts: GSM8K test set with the chat template.
- KV cache: with default sizing, flag-on arms allocate the count before the KV cache and got 9,573 blocks; flag-off arms got 9,573 or 9,574. The workers' block counts are in each result (`workers.after_start[].kv_num_blocks`). t24 and parts of t25 pin the block count.

## PR claims and their sources

| Claim | Results | Output |
|---|---|---|
| Head `16ed9b0a7`: unit tests 1,174 passed, Pyrefly baseline; numerics + switching sequence (flag on and off), EP fallback and MTP drafter bit-identical to the parent; the fallback warning only in the EP and MTP runs; decode programs identical to `056cee01c`'s (XLA dumps, flag on and off) | `t27/*` (dumps not included) | [t27](summaries/t27_nits_head.txt), [HLO](summaries/hlo_t27_nits.txt) |
| Blocking padding-state copies: no measurable cost, and they do not break async-scheduling overlap (pre-registered; two hosts, KV pinned, async on / off) | `t28/a1-async-copies`, `t28/a2-async-copies` | [t28](summaries/t28_async_copies.txt) |
| Head `975685ab2`: unit tests 1,178 passed, Pyrefly baseline | `t28/u0-unit-head` | [t28](summaries/t28_async_copies.txt) |
| `9d9fcb849`: unit tests 1,174 passed (incl. the 9 added TPU cases), Pyrefly baseline | `t26/u0-unit-head` (status, pytest and Pyrefly logs) | [t26](summaries/t26_unit_tests_head.txt) |
| `056cee01c`: unit tests 1,165 / parent 1,094, Pyrefly baseline, direct MoE check 45/45 | `t25/q0-unit-n2` | [t25 validation](summaries/t25_validation.txt) |
| `056cee01c`: numerics + switching sequence vs parent, flag on and off; GSM8K quality (1,319 prompts, greedy and sampled) | `t25/q1-correctness-quality` | [t25 validation](summaries/t25_validation.txt) |
| `056cee01c`: EP fallback; MTP drafter with the flag (and the failure of `fe5cbee79`) | `t25/e1-ep-fallback`, `t25/m1-mtp-drafter` | [t25 validation](summaries/t25_validation.txt) |
| Review fixes: same decode programs (XLA dumps, `up3` vs `up2`); steady timing, KV pinned, ABBA, two hosts | (dumps not included), `t25/s1-fix-perf`, `t25/s2-fix-perf` | [HLO](summaries/hlo_t25_fix_vs_prev_head.txt), [fix perf](summaries/perf_t25_fix_no_change.txt) |
| KV cache size pinned for both arms (9,573 / 9,574 blocks) | `t24/k1-kv-pinned`, `t24/k2-kv-pinned` | [t24](summaries/perf_t24_kv_pinned.txt) |
| Earlier trees: unit tests 836 / 772, direct MoE check, numerics, EP fallback, served path | `t22/a0`, `t22/a4`, `t22/a5`, `t22b`, `t22c` | [correctness](summaries/correctness.txt) |
| Steady decode, 4 hosts | `t22/a1`, `a2`, `a6`, `a7` | [perf_t22_main](summaries/perf_t22_main.txt) |
| Other bucket points, full buckets per session | `t22/a8`, `a9` | [extra points](summaries/perf_t22_extra_points.txt), [per session](summaries/perf_t22_full_buckets_per_session.txt) |
| 256-request rollout | `t22/a3-rollout` | [rollout](summaries/perf_t22_rollout.txt) |
| `fe5cbee79` performance; per-layer mask variant (not adopted) | `t23/d1-perf`, `t23/d2-perf`, `t23/d4-correctness` | [t23](summaries/perf_t23_head_and_barrier.txt) |
| Zero-weight approach on the same nightly | `t21-gatezero/z3`, `z4` | [gate-zero](summaries/gatezero_t21.txt) |
| n=16 device-trace attribution | (traces not included) | [trace report](traces/n16_attribution_15be35d.md) |
| Zero-weight approach: why its 16-row program is slower (minimal pair `base_on` vs `final_on` on `f640d734`: the zeroing adds +1.8 µs of new ops and +6.8 µs in modified fusions, while instruction-memory overlay waits go 57 → 320 per step) | `compiler_report/results/*` (traces and HLO dumps not included) | [compiler report](compiler_report/REPORT.md) (drafted for the XLA TPU team, not filed) |
| Early zero-weight + EP switching: chosen-token log-probs of 2 rows differ (token ids identical); TP-only and the EP stock-path fallback are bit-identical to main | `results/t11-ep-zero-weight`, `results/t16-ep-fallback` | [EP summary](summaries/ep_zero_weight_t11_t16.txt) |

## Reproduce the outputs

From this folder (Python 3, standard library):

```bash
python3 tools/t28_verdict.py results/t28
python3 tools/analyze_mskip.py results/t27/n1-head-on --check --ref upbase_off_f1
python3 tools/analyze_mskip.py results/t27/n2-head-off --check --ref upbase_off_f1
python3 tools/analyze_mskip.py results/t27/e1-ep-fallback --check --ref upbase_off_e1
python3 tools/analyze_mskip.py results/t27/m1-mtp-drafter --check --ref upbase_off_m1
python3 tools/analyze_mskip.py results/t25/q1-correctness-quality --check --ref upbase_off_f1
python3 tools/n1_compare.py results/t25/q1-correctness-quality --ref upbase_off_f1 --arms up3_on_f1,up3_off_f1
python3 tools/quality_compare.py results/t25/q1-correctness-quality --ref upbase_off --arms up3_on,up3_off
python3 tools/analyze_mskip.py results/t25/e1-ep-fallback --check --ref upbase_off_e1
python3 tools/analyze_mskip.py results/t25/m1-mtp-drafter --check --ref upbase_off_m1
python3 tools/same_host_compare.py results/t25 --ref up2k9574_on --arms up3k9574_on,upbasek9574_off --points 1,16,64,256
python3 tools/same_host_compare.py results/t24 --ref upbasek9574_off --arms up2k9574_on --points 1,15,16,64,255,256
python3 tools/same_host_compare.py results/t24 --ref upbasek9573_off --arms up2k9573_on --points 1,15,16,64,255,256
python3 tools/served_compare.py results/t22/a5-served --ref upbase_off_d1 --arms up_on_d1
python3 tools/t22_compare.py results/t22                  # every t22 host: steady, prefill/mixed, rollout
python3 tools/same_host_compare.py results/t23/d1-perf --ref upbase_off --arms up2_on,upbar_on
python3 tools/t21_compare.py results/t21-gatezero --experiment gatezero
python3 tools/n1_compare.py results/t11-ep-zero-weight --ref base_off_e1 --arms final_on_e1,final_off_e1
python3 tools/ep_logprob_diff.py results/t11-ep-zero-weight --ref base_off_e1 --arm final_on_e1
python3 tools/n1_compare.py results/t16-ep-fallback --ref base_off_e2 --arms epfb_on_e2,epfb_off_e2
```

XLA dumps (`MSKIP_XLA_DUMP=1`; the dumps are not included):

```bash
python3 tools/hlo_dump_compare.py A_DUMP_DIR B_DUMP_DIR          # every compiled module, as a multiset
python3 tools/hlo_backbone_check.py up3=A_DUMP_DIR up5=B_DUMP_DIR  # the decode backbone programs (fused_add_rms_norm)
```

Engine runs (TPU, per arm):

```bash
# numerics + staggered batch-size switching + GSM8K quality in one engine
python3 tools/mskip_engine.py --label up3_on_f1 --out OUT --model MODEL --data GSM8K_PARQUET --phases numerics,switch,quality \
    --switch-stagger 13 --switch-seq 16,10,16,1,16,15,12,16,3,17,32,16,256,250,255,256 --probe-workers
# MTP drafter with dummy weights
python3 tools/mskip_engine.py --label up3_on_m1 --out OUT --model MODEL --data GSM8K_PARQUET --phases numerics \
    --speculative mtp:1 --load-format dummy --probe-workers
# steady decode: per step = (t(545 tokens) - t(33 tokens)) / 512; MSKIP_NUM_GPU_BLOCKS pins the KV cache
MSKIP_NUM_GPU_BLOCKS=9574 python3 tools/mskip_engine.py --label up3k9574_on_a1 --out OUT --model MODEL \
    --data GSM8K_PARQUET --phases steady --steady-ns 1,16,64,256 --steady-lens 33,545 --steady-reps 5 --probe-workers
# direct MoE check on one chip
python3 tools/moe_direct_check.py --out n2.json
```

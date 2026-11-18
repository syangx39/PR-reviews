# Notes: verification status, discrepancies, open questions

Companion to [REPORT.md](REPORT.md) and an internal artifact inventory (not public). "Brief" means the one-paragraph description of the anomaly in
the task request. "Doc" means our internal write-up (Chinese, not public). Where they disagree with a file, the file wins.

## 0. Claims from the brief, checked against the files

| Claim in the brief | What the files say | Status |
|---|---|---|
| 16-row bucket about 4.5% slower per step | Wall clock, `final_on` vs main: +4.5% / +4.2% (`results/t3/g{2,3}/*_q*.json.gz`), +4.90% / +4.54% (`results/t8/p2_t8.txt`). Zeroing only (`final_on` vs `base_on`): +4.03% / +3.99% (`results/t14/p2_t14.txt`). Device forward: +4.31% / +5.09% (`results/t4/t4_summary.txt`) | OK |
| A `torch.where` on `[16, 8]` per layer; the skip-kernel variant selected | `token_padding.zero_routing_weights_for_padding` (`<workdir>/build/arm-base/.../token_padding.py`). HLO: 40 × `select bf16[16,8]` before optimization (`evidence/hlo_entry_opdiff.txt`) | OK |
| "differs by only ~70 instructions (about 40 changed compare/fusion, 30 mask copy pairs, one param, one slice)" | +62 added top-level instructions (30 `copy-start` + 30 `copy-done` + 1 parameter + 1 slice) and 40 modified fusions: 102 instructions added or modified. "~70" counts each copy pair as one (40 + 30 + 1 + 1 = 72) | **Wording** (§2 item 1) |
| Unchanged ops 2–4 µs slower at fixed positions | Per slow instance: all-reduce +3.7–3.9 µs, `select_reduce_fusion` +1.9 µs, GDN conv +2.4–2.5 µs (`results/t4/t4_summary.txt`); positions identical on both hosts (`slow_positions_n16.txt`) | OK, range is 1.9–3.9 µs |
| +290–310 µs per step | Aligned existing ops: +309.3 / +288.5 µs (t4); +315.5 / +287.3 µs with markers (t5) | OK |
| Overlay waits 57 → 320 per step, identical on 3 hosts | t5 g0 `host-e02`, t5 g1 `host-e03`, t7 g0 `host-e04`: all 57 → 320. The third host's numbers exist only in the scratch file `<workdir>/results/t7_g0_summary.txt` (copied to `evidence/`) | OK (provenance caveat, §2 item 2) |
| +228 µs, 72–79% of the extra time | +227.7 / +227.6 µs at 72% / 79% (t5). The third host (t7 g0): +227.1 µs at 71% | OK; across 3 hosts 71–79% |
| 32-row program reverses (173 → 73, `final_on` faster) | 173 → 73 on all of t5 g0/g1, t7 g0, t6 g0; forward −1.28% to −1.61% | OK |
| `merge_small_overlays_into_big_neighbors=false` has no effect | 57 → 320, +323.6 µs (`results/t6/t6_summary.txt`) | OK |
| `overlay_enable_hlo_function_packs=false` equalizes (n16 −0.06%, n32 −0.14%) but both 8–10% slower | Equalizes: OK. The slowdown is cross-host and depends on which program you compare (§2 item 6) | **Partly** |
| Same on newer vllm-torchtpu main (t10) | 57 → 320, 173 → 73, but cross-host only, and the same libtpu build | OK with caveats (§2 item 7) |

## 1. Not verified, or verified only indirectly

1. **pip versions for the t4–t12 images.** `pip.txt` was uploaded only for t2/t3 (`G/t2/g0/pip.txt`, `G/t3/g0/pip.txt`; images
   `5b04ea68…`, `20189991…`). The t4–t7 and t12 images are built FROM the same base digest, according to [`test/Dockerfile`](test/Dockerfile), and
   every t4/t5/t6/t7/t10/t12 engine log prints the same PJRT build string (``, C API 0.116, TorchTPU fingerprint
   `8ae752df13e66ac8`). So libtpu 0.0.50 / jax 0.11.2 / torch_tpu 0.2.0 is very likely, but the pip list was not recorded for those
   images, or at all for the t10 base image.
2. **Base image tag ↔ digest** (`nightly-20261114-f640d73-v0.33.0` ↔ `d88fcf41…`, `nightly-20261115-b7c8b71-v0.33.0` ↔
   `5e3c74f8…`) comes only from Dockerfile comments. It was not checked against the registry.
3. **Image digests for t5, t6 and t12** come from the launch-time JobSet specs that are still in this workstation's `/tmp`
   (`/tmp/mskip-t*-g*.yaml`; their mtimes match the pod start times). There are no `image_tag_t5/t6/t12` files. t12 used the t11
   image `d8e2e722…`; the doc does not say so.
4. **Was `LIBTPU_INIT_ARGS` applied?** The engine logs and pod records do not print it (`run_pod.sh` records only `TPU_*` and
   `TORCH_TPU*`). Evidence that it was applied:
   - the `/tmp` specs;
   - `overlay-wait` TraceMe events are present in the traces of the runs that set the marker flag (`results/t5/overlay_w1_n16.txt`).
     That they are absent from the unflagged t4 traces was not checked: the protobuf-based trace parser could not be run in this
     environment;
   - the function-packs option visibly changes the counts.

   For `merge_small_overlays_into_big_neighbors=false` (t6 g0) nothing in the artifacts shows that the option took effect, because
   its result equals the default.
5. **Does the marker flag change the generated code?** The doc says `--xla_tpu_show_overlay_waits_in_profiler=true` only adds trace
   markers in the wait routine. The trace module ids do change between t4 (no flag) and t5 (flag): 16-row `base_on`
   `14512450523834468197` → `3071809948165506109`, `final_on` `14121214978197194209` → `2855544052435770862`
   (`results/t4/align_*`, `results/t5/align_w1_*`). So the compile key changes at least. No HLO was dumped with the flag on,
   so whether the code changes is not verified. The phenomenon persists either way (aligned-op Δ +315.5 / +287.3 with the flag,
   +309.3 / +288.5 without).
6. **How overlays work** (an IMEM overlay prefetched from HBM, with a stall when the next one is not ready) is our reading of the flag
   names and the doc. It was not checked against XLA internals. The report presents it as an understanding or hypothesis.
7. **Only rank 0 (`/device:TPU:0`) was analysed.** The archives contain rank 1–3 traces. Wait counts on the other chips are
   unknown.
8. **Whether the wait counts depend on the input data** has not been tested; all runs used the same GSM8K prompts and real weights.
9. **What the trace module ids mean.** The number in parentheses after the module name was treated as a program fingerprint
   (identical across hosts for the same program). Its exact meaning was not checked.
10. **Why t10 is incomplete.** Each t10 pod record stops after its first arm, with no `POD done` line
    (`G/t10/g*/arms_*.txt`). The doc attributes this to spot preemption; the artifacts do not record the cause.
11. **The `vmul` commit `5ae458f9`** comes from the Dockerfile comment in `<workdir>/build/Dockerfile`. The run JSONs confirm
    a different `token_padding.py` (sha256 `7598c550f25294f2`) and an unchanged `unquantized.py`.
12. **The standalone replay plan (REPORT §8) is untested.** In particular, we do not know whether synthetic inputs reproduce the
    counts, or whether the dumped `module.mlir` plus compile options recompile to the identical optimized HLO outside vLLM.
13. **Trace-derived numbers were not recomputed from the raw `xplane.pb` files.** The trace parser ([`test/prof_mskip.py`](test/prof_mskip.py)) needs
    protobuf, which this environment does not have. Every overlay-wait, aligned-op and module-time number is taken from the
    committed summary files (`results/t4`, `t5`, `t6`, `t7`, `t10`, `t12`, plus `evidence/t7_g0_summary.txt`) and checked against
    them string by string.

    These were recomputed independently:
    - the HLO facts, from the extracted dumps, with `evidence/entry_opdiff.py` and `evidence/scan_modules.py`; they agree with
      `results/t4/hlo_diff_b16.txt` and `hlo_diff_b32.txt`;
    - the t3 n=1 / n=16 wall-clock medians, from the raw `results/t3/g{2,3}/*.json.gz`;
    - the decode-step share, from `test/rollout_lengths.json`.

## 2. Discrepancies between the doc/brief and the files

1. **The "~70 instruction" count.** §5.4 of the doc ("程序只差约 70 条指令") and the brief both say this. The exact count, after
   optimization, for the 16-row ENTRY, is: +62 instructions (5,730 → 5,792) and 40 modified fusions, 102 in total. Before
   optimization it is +243 (13,686 → 13,929). In the 16-row program only 30 of the 40 layers get a prefetch copy pair: the 10
   fusions at positions 4, 8, …, 40 (the full-attention layers) read the slice directly. The 32-row program has 40 pairs.
   Sources: `results/t4/hlo_diff_b16.txt`, `hlo_diff_b32.txt`, `evidence/hlo_entry_opdiff.txt`. The normalized diff also
   shows one parameter declaration (`weights_ref_gdn_dt_bias`) at a different position, which is ordering only.
2. **The third host.** The doc's §5.4 table ("t7，3 台机器"; vmul +308.6 µs (g0); "t7 也在第三台机器上复现了 final 的结果
   （57 → 320 次，已有 op +321.5 µs）") rests on t7 g0. But `results/t7/` contains only the g1 summary and JSONs. The g0
   summary exists only at `<workdir>/results/t7_g0_summary.txt`; its numbers match the doc and it is copied to
   `evidence/t7_g0_summary.txt`.
3. **The wait share.** The doc and brief say 72–79%; that is t5's two hosts. With t7 g0 it is 71–79% (t6 g0, where only the
   merge option differs: 70%). The share is wait-time Δ divided by the aligned existing-op Δ, both measured with markers on, not
   divided by the module-time Δ.
4. **"2–4 µs".** The smallest per-instance increase is `select_reduce_fusion` at 0.69 → 2.58 µs, i.e. +1.9 µs.
5. **Module medians are noisier than the aligned-op metric.** For example, t7 g0 has forward +2.40% (5,857.1 → 5,997.6 µs) while
   aligned ops are +321.5 µs (≈ +5.5%), and t5 g1 has forward +5.83% while aligned ops are +287.3 µs. The doc mostly quotes
   aligned-op numbers, which is right. Readers should not compare module medians across hosts.
6. **"Both programs 8–10% slower" with function packs disabled.** Every one of these comparisons is cross-host: the packs=false
   run is on `host-e08`, which has no default-option run.
   - `base_on`: +8.0% to +9.8% at n=16 and +8.5% to +8.9% at n=32. Matches the claim.
   - `final_on` at n=32: +10.1–10.2%. Matches the claim.
   - `final_on` at n=16 is only 3.4–5.4% slower than `final_on` with default packing (6,320.4 vs 5,997.4–6,112.8 µs), because
     the default `final_on` already carries the regression.

   The doc's "默认 5,760–5,857 µs" includes 5,760.2 µs from t6 g0, which ran with `merge_small_overlays_into_big_neighbors=false`
   (no effect, but not strictly default). The default-only range is 5,775.0–5,857.1 µs.
7. **t10, "newer main".** It is cross-host only, because the same-host pair never finished. It also uses the **same PJRT/libtpu
   build** as every other run, so it is not evidence about a newer compiler. Only n=16 and n=32 were profiled, and there is no HLO
   dump.
8. **t12.** The doc's §5.6 quotes only device-time deltas (+0.13%, +0.00%, −1.00%). The files also give the wait counts:
   216 → 186, 343 → 402, 484 → 501. These come from g1 (`host-e06`); the doc does not name the host. The g0 pair
   (`host-e07`) finished at 05:06 UTC but has never been analysed, and `results/t12/` lacks `final_on_y1.json`.
9. **KV-cache size.** It is not mentioned in the doc. `base_off` gets one more KV-cache block than `base_on` / `final_on`:
   2,450,944 vs 2,450,688 tokens in the logs, and `bf16[9574,…]` vs `bf16[9573,…]` parameters in the HLO. Comparisons against
   `base_off` therefore contain this extra difference. `base_on` vs `final_on` do not.
10. **Doc §7** lists the raw results as `{t2,t3,…,t11}`, but `t12`, `t13`, `t13s` and `t14` also exist under `G/`.
11. **Paths.** The brief says the logs and JSON are under `G/{t4,t5,t6,t10,t12}/<group>/`. They are, and so is t7 (`G/t7/`), which
    the brief's listing omits.

## 3. Incomplete data (what we have not got)

- No minimization and no standalone replay. Every observation is from the full vLLM engine on the full model.
- HLO dumps exist only for t4 g0 (default options, no markers). There are none for t5–t12, so none for the packs=false programs
  or the t10 programs.
- t12 g0 is not analysed, and there is no sha256 for its two archives (no local copy).
- t10 has no same-host pair.
- Ranks 1–3 are not analysed.
- No compiler-internal view of the overlay or function-pack assignment.
- No test with a newer libtpu.
- No measurement of overlay overhead outside the wait markers (the 21–29% residual). Candidate flags such as
  `--xla_tpu_show_overlay_overhead_in_profiler` exist in the flagfile but were not tried.

## 4. Open questions and logistics before filing

1. **Access.** Can the XLA team read `G`? The project and ACL owner of the bucket were not checked. If they cannot,
   the minimal set (ARTIFACTS §1: the two t4 HLO archives, 0.84 GB, plus the two or four t5 trace archives, 0.80–1.60 GB; about
   1.6–2.4 GB in total) would need to be copied somewhere they can read.
2. **Weights.** The standalone replay needs either synthetic inputs, which is untested, or the 69 GB model inputs
   (`<inputs>`). Is sharing those acceptable?
3. **Where to file.** Which component or hotlist (XLA:TPU overlays / code layout), and who triages, has not been determined.
4. **Rerun first?** Should the trigger be re-checked with a newer libtpu before filing? All evidence is from ``.
5. **More captures?** Should we capture a parameter dump and per-module traces for an exact replay, and run the marker flag
   together with `--xla_dump_to`, so that the HLO of the marker-flag programs is available too?
6. **Dual-graph workaround.** Should the report mention the result of the runtime workaround (launch plan `t15`) once it is
   available?

## 5. Housekeeping

- Only new files were written, all under `compiler_report/`. Scratch work (hash computations, downloads of small logs and JSON
  files, generator scripts) is in
  `<workdir>/`.
- No TPU jobs were launched. Nothing was filed or posted. No git commit or push was made. No existing file was edited. No large
  archive was downloaded; the large archives already present in `<workdir>/results/t*raw/` from earlier work were only
  read, to hash them after checking each one's md5 against GCS.
- The `compiler_report/` directory is untracked in the experiment repo. The worktree also has unrelated uncommitted t15 edits
  under `test/`.

# vllm-torchtpu nightly-20261114-f640d73-v0.33.0 with vllm-torchtpu source trees for the MoeSkip test
# (src, tests, tools, pyproject.toml and the pyrefly baselines of each commit):
#   /opt/arm-base   f640d734, the image's own commit
#   /opt/arm-fix    f640d734 + the unquantized MoeSkip wiring (5f53fe77)
#   /opt/arm-final  5f53fe77 + the cached flag passed to the kernel prebuild and selection: 7e4d170a (the t3 image,
#                   sha256:20189991, was built with 5384ef82, which has the same src/ and the two new tests unfixed)
#   /opt/arm-vmul   7e4d170a + P1 screening variant: padded weights zeroed by multiplication (5ae458f9)
#   /opt/arm-dual   7e4d170a + the 16-row dual-graph prototype, TPU_MOE_SKIP_STOCK_GRAPH (b2a413d6)
#   /opt/arm-epfb   7e4d170a + the bf16 MoE keeps the stock path under expert parallelism (586e11bc)
#   /opt/arm-basemc f640d734 + the padding-mask write skipped when the state is unchanged (control, ff9d98ef)
#   /opt/arm-gskip  7e4d170a + the mask write skipped when unchanged (0b5f16aa) + the single-graph experiment:
#                   padded rows dropped in the GMM route filter, TPU_MOE_SKIP_IN_GMM=count|mask (beaf78fe)
#   /opt/arm-opta   f640d734 + Option A upstream candidate (82da5750a + bf98bcd4d; src tree d89c4fe27,
#                   identical across t19/t19u2/t19u3): BF16 row-count MoeSkip (padding_rows="count") +
#                   unified early capability gating + default full-bucket stock graph (16:16,256:256,
#                   configurable via TPU_MOE_SKIP_STOCK_GRAPH)
# A run picks one with PYTHONPATH=/opt/arm-<tree>/src. /opt/measure has the test scripts (run_pod.sh, run_arm.sh,
# run_chain.sh, run_unit.sh, run_n2.sh, mskip_engine.py, moe_direct_check.py, check_arm.py, graphsel_check.py, gcs_out.py,
# fetch_inputs.py, rollout_lengths.json).
FROM vllm/vllm-torchtpu@sha256:d88fcf418f32aaad79916c25498629ac4732cdfcc6faf8f6aab7e0ccc446dfac
COPY arm-base /opt/arm-base
COPY arm-fix /opt/arm-fix
COPY arm-final /opt/arm-final
COPY arm-vmul /opt/arm-vmul
COPY arm-dual /opt/arm-dual
COPY arm-epfb /opt/arm-epfb
COPY arm-basemc /opt/arm-basemc
COPY arm-gskip /opt/arm-gskip
COPY arm-opta /opt/arm-opta
COPY measure /opt/measure

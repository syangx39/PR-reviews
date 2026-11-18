#!/bin/bash
# One pod of the MoeSkip test (single TPU v7x host, 2x2x1). $MSKIP_ARMS lists the arms, run one after another:
#   <tree>_<on|off>_<rep>[@<phases>]  an engine arm (run_arm.sh); phases default to numerics,sweep,rollout
#   unittest                          run_unit.sh: the related CPU unit tests and the full-environment type check on
#                                     each tree in $MSKIP_UNIT_TREES (default "final base")
#   n2-<tree>                         run_n2.sh: the direct MoE real-row output check on one chip (label n2_<tree>)
#   rid-<tree>                        run_n2.sh with rid_direct_check.py: router-ID kernel + MoE real-row check and
#                                     the isolated stock-GMM HLO (label rid_<tree>)
#   srv-<tree>_<on|off>_<rep>         run_served.sh: one `vllm serve` arm (served_check.py); $EXTRA holds its
#                                     arguments (layout: --tp / --dp / --ep, --env K=V)
# Trees: /opt/arm-base (f640d734, the image's commit), /opt/arm-fix (+ 5f53fe77), /opt/arm-final (+ the final commit).
# MSKIP_INPROC=1 runs every engine arm with the in-process engine core; EXTRA adds mskip_engine.py arguments.
# A finished arm leaves <arm>.done on $OUT_PREFIX only after its results are stored there, and a restarted pod skips
# those arms, so a spot preemption costs only the arm in flight. Results also stay in /out; the pod then sleeps so
# they can be copied out.
set -u
ARMS="${MSKIP_ARMS:?}"
OUT=/out; mkdir -p "$OUT" /tmp/m
M=/opt/measure
export OUT
REC="$OUT/arms_${POD_NAME:-$(hostname)}.txt"
gcs() { [[ -z "${OUT_PREFIX:-}" ]] || python3 $M/gcs_out.py "$@" >> "$OUT/gcs.log" 2>&1; }
echo "POD group=${MSKIP_GROUP:-} pod=${POD_NAME:-$(hostname)} node=${NODE_NAME:-} arms=$ARMS start=$(date -u +%FT%TZ)" \
  | tee -a "$REC"
for v in $(env | cut -d= -f1 | grep -E '^(MEGASCALE_|TPU_WORKER_|TPU_HOST_BOUNDS$|TPU_CHIPS_PER_HOST_BOUNDS$|TPU_TOPOLOGY|TPU_PROCESS_)'); do
  echo "unset $v=${!v}" >> "$REC"; unset "$v"
done
export TPU_ACCELERATOR_TYPE=tpu7x TPU_MULTIHOST_BACKEND=mp TORCH_TPU_TOPOLOGY=1,2,1,2
env | grep -E '^(TORCH_TPU|TPU_|MEGASCALE)' | sort >> "$REC"
pip list 2>/dev/null | grep -iE '^(vllm|torch|torch.tpu|torch_tpu|jax|jaxlib|libtpu|vllm.torchtpu|pytest|pyrefly)' \
  > "$OUT/pip.txt"

if [[ " $ARMS " =~ [a-z]+_(on|off)_ ]]; then
  INPUT_PREFIX=<inputs> INPUT_DIR=/data \
    python3 $M/fetch_inputs.py > "$OUT/fetch.log" 2>&1
  echo "fetch rc=$? $(date -u +%FT%TZ)" | tee -a "$REC"
fi

fail=0
for SPEC in $ARMS; do
  if [[ "$SPEC" == srv-* ]]; then
    LABEL=${SPEC#srv-}; TREE=${LABEL%%_*}; FLAG=$(echo "$LABEL" | cut -d_ -f2)
    # shellcheck disable=SC2086
    bash $M/run_served.sh "$TREE" "$FLAG" "$LABEL" ${EXTRA:-} || fail=1
    continue
  fi
  if [[ "$SPEC" == n2-* ]]; then
    bash $M/run_n2.sh "${SPEC#n2-}" "n2_${SPEC#n2-}" || fail=1
    continue
  fi
  if [[ "$SPEC" == rid-* ]]; then
    N2_SCRIPT=rid_direct_check.py bash $M/run_n2.sh "${SPEC#rid-}" "rid_${SPEC#rid-}" || fail=1
    continue
  fi
  if [[ "$SPEC" != unittest ]]; then
    LABEL=${SPEC%%@*}; PHASES=numerics,sweep,rollout; [[ "$SPEC" == *@* ]] && PHASES=${SPEC#*@}
    TREE=${LABEL%%_*}; FLAG=$(echo "$LABEL" | cut -d_ -f2)
    # shellcheck disable=SC2086
    bash $M/run_arm.sh "$TREE" "$FLAG" "$LABEL" "$PHASES" ${EXTRA:-} || fail=1
    continue
  fi
  # shellcheck disable=SC2086
  bash $M/run_unit.sh ${MSKIP_UNIT_TREES:-final base} || fail=1
done
echo "POD done fail=$fail $(date -u +%FT%TZ)" | tee -a "$REC"
gcs put "$REC" "$OUT/gcs.log"
sleep infinity

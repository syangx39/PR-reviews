#!/bin/bash
# Run one engine arm of the MoeSkip test and publish its results.
#   bash run_arm.sh <tree> <on|off> <label> [phases] [extra mskip_engine.py args...]
#     tree     /opt/arm-<tree>: base (f640d734), fix (+ 5f53fe77), final (+ the final commit: 7e4d170a; the t3
#              image has 5384ef82, same src/), dual (7e4d170a + the 16-row dual-graph prototype). dual<N> also runs
#              /opt/arm-dual, with TPU_MOE_SKIP_STOCK_GRAPH=16:<N> when on (dual alone: 16:15; MSKIP_STOCK_GRAPH
#              overrides); dualfull runs /opt/arm-dual with 16:15,256:256 when on. gcount / gmask run
#              /opt/arm-gskip with TPU_MOE_SKIP_IN_GMM=count / mask when on (single-graph experiment). optans runs
#              /opt/arm-opta with TPU_MOE_SKIP_STOCK_GRAPH=0 when on (Option A row count, single graph). rid runs
#              /opt/arm-rid with TPU_MOE_SKIP_PADDING_ROWS=router when on (router-ID single graph). t21 (image on
#              nightly-20261116-15be35d): base = 15be35d, gz = + the gate-zero commits 0001-0003, opta = + Option A,
#              optmc / optnb = Option A with the padding-mask write skipped when the valid count is tracked / and the
#              count copied non-blocking; optans, optmc and optnb run with TPU_MOE_SKIP_STOCK_GRAPH=0 when on (single
#              graph). Every other tree runs with TPU_MOE_SKIP_STOCK_GRAPH, TPU_MOE_SKIP_IN_GMM and
#              TPU_MOE_SKIP_PADDING_ROWS unset.
#     on/off   TPU_MOE_SKIP_PADDED_TOKENS=1 / unset
#     phases   comma-separated mskip_engine.py phases (default numerics,sweep,rollout)
# Env: MSKIP_INPROC=1 runs the engine core in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0), so every request of a batch
# is scheduled before the first step; MSKIP_XLA_DUMP=1 dumps the compiled HLO; OUT (default /out); OUT_PREFIX
# (<artifact-store> optional); POD_NAME.
#
# Arms on one host must run one after another (run_chain.sh): each uses all four chips, and the cleanup below kills
# every VLLM:: process on the host.
#
# 1. Skip when $OUT_PREFIX/<label>.done exists (a finished arm, e.g. before a spot preemption).
# 2. Run mskip_engine.py in a fresh process with its own HOME / TMPDIR (own compile caches).
# 3. The arm succeeded only if the engine exited 0 and check_arm.py finds every requested phase complete; otherwise
#    the JSON is renamed <label>.partial.json, which the analysis never reads as a run.
# 4. Upload the log and the JSON; only after the arm succeeded and every upload is verified, write and upload
#    <label>.done. Exit 0 on success, otherwise the engine's exit status (or 1).
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
TREE=$1; FLAG=$2; LABEL=$3; PHASES=${4:-numerics,sweep,rollout}
shift $(( $# < 4 ? $# : 4 ))
OUT=${OUT:-/out}; mkdir -p "$OUT"
REC="$OUT/arms_${POD_NAME:-$(hostname)}.txt"
note() { echo "$*" | tee -a "$REC"; }
gcs() { [[ -z "${OUT_PREFIX:-}" ]] || python3 "$DIR/gcs_out.py" "$@" >> "$OUT/gcs.log" 2>&1; }

if [[ -n "${OUT_PREFIX:-}" ]] && python3 "$DIR/gcs_out.py" has "$LABEL.done"; then
  note "ARM $LABEL already done on GCS, skipped $(date -u +%FT%TZ)"; exit 0
fi
note "ARM $LABEL start tree=$TREE flag=$FLAG phases=$PHASES inproc=${MSKIP_INPROC:-0} $(date -u +%FT%TZ)"
TREE_DIR=$TREE; STOCK_GRAPH=""; SKIP_IN_GMM=""; PADDING_ROWS=""
if [[ $TREE =~ ^dual([0-9]*)$ ]]; then
  TREE_DIR=dual
  [[ $FLAG == on ]] && STOCK_GRAPH=${MSKIP_STOCK_GRAPH:-16:${BASH_REMATCH[1]:-15}}
elif [[ $TREE == dualfull ]]; then
  TREE_DIR=dual
  [[ $FLAG == on ]] && STOCK_GRAPH=16:15,256:256
elif [[ $TREE == gcount || $TREE == gmask ]]; then
  TREE_DIR=gskip
  [[ $FLAG == on ]] && SKIP_IN_GMM=${TREE#g}
elif [[ $TREE == optans ]]; then
  TREE_DIR=opta
  [[ $FLAG == on ]] && STOCK_GRAPH=0
elif [[ $TREE == optmc || $TREE == optnb ]]; then
  [[ $FLAG == on ]] && STOCK_GRAPH=0
elif [[ $TREE == rid ]]; then
  [[ $FLAG == on ]] && PADDING_ROWS=router
fi
F=/tmp/m/$LABEL; rm -rf "$F"; mkdir -p "$F/home" "$F/tmp"
rm -f "$OUT/$LABEL.json" "$OUT/$LABEL.partial.json" "$OUT/$LABEL.done"
(
  export PYTHONPATH="/opt/arm-$TREE_DIR/src:/workspace/vllm-torchtpu"
  export HOME="$F/home" TMPDIR="$F/tmp" XDG_CACHE_HOME="$F/home/.cache"
  unset TPU_MOE_SKIP_PADDED_TOKENS VLLM_TPU_FAST_TOKEN_SUBSTITUTION USE_PHASED_PROFILER \
    TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT VLLM_ENABLE_V1_MULTIPROCESSING TPU_MOE_SKIP_STOCK_GRAPH \
    TPU_MOE_SKIP_IN_GMM TPU_MOE_SKIP_PADDING_ROWS
  [[ "$FLAG" == on ]] && export TPU_MOE_SKIP_PADDED_TOKENS=1
  [[ -n "$STOCK_GRAPH" ]] && export TPU_MOE_SKIP_STOCK_GRAPH=$STOCK_GRAPH
  [[ -n "$SKIP_IN_GMM" ]] && export TPU_MOE_SKIP_IN_GMM=$SKIP_IN_GMM
  [[ -n "$PADDING_ROWS" ]] && export TPU_MOE_SKIP_PADDING_ROWS=$PADDING_ROWS
  [[ "${MSKIP_INPROC:-0}" == 1 ]] && export VLLM_ENABLE_V1_MULTIPROCESSING=0
  # MSKIP_XLA_DUMP=1: dump the HLO of every module this engine compiles (text, before / after optimization).
  [[ "${MSKIP_XLA_DUMP:-0}" == 1 ]] && export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=$OUT/hlo/$LABEL --xla_dump_hlo_as_text"
  exec timeout 3600 python3 "$DIR/mskip_engine.py" --label "$LABEL" --out "$OUT" --phases "$PHASES" "$@"
) > "$OUT/$LABEL.log" 2>&1
rc=$?
pkill -9 -f "^VLLM::" 2>/dev/null; sleep 3

status=$rc
if [[ $status == 0 ]] && ! python3 "$DIR/check_arm.py" "$OUT/$LABEL.json" --phases "$PHASES" >> "$OUT/$LABEL.log" 2>&1; then
  status=1
fi
if [[ $status != 0 && -f "$OUT/$LABEL.json" ]]; then
  mv "$OUT/$LABEL.json" "$OUT/$LABEL.partial.json"
fi
note "ARM $LABEL engine_rc=$rc status=$status end $(date -u +%FT%TZ)"

files=("$OUT/$LABEL.log" "$REC")
for f in "$OUT/$LABEL.json" "$OUT/$LABEL.partial.json"; do [[ -f "$f" ]] && files+=("$f"); done
if [[ -d "$OUT/prof/$LABEL" ]]; then  # profile phase: the traces of this arm as one archive
  tar czf "$OUT/$LABEL.prof.tgz" -C "$OUT/prof" "$LABEL" && files+=("$OUT/$LABEL.prof.tgz")
fi
if [[ -d "$OUT/hlo/$LABEL" ]]; then  # MSKIP_XLA_DUMP: the HLO dumps as one archive
  tar czf "$OUT/$LABEL.hlo.tgz" -C "$OUT/hlo" "$LABEL" && files+=("$OUT/$LABEL.hlo.tgz")
fi
if ! gcs put "${files[@]}" && [[ $status == 0 ]]; then
  status=1; note "ARM $LABEL upload failed, no .done"
fi
if [[ $status == 0 ]]; then
  date -u +%FT%TZ > "$OUT/$LABEL.done"
  if ! gcs put "$OUT/$LABEL.done"; then status=1; note "ARM $LABEL .done upload failed"; fi
fi
exit "$status"

#!/usr/bin/env bash
# Run the MERFISH 5-slice 10-method x 7-combo matrix on four GPUs.
#
# Prerequisites: $MERFISH5_ROOT/{canonical,observed_50um.h5ad} and
# $MERFISH5_UPSTREAM are ready; source work/env.sh first.
#
# Usage:
#   source work/env.sh
#   MERFISH5_GPUS="0 1 4 5" bash scripts/run_merfish5_matrix_4gpu.sh
#
# Skip predictions that already exist (default 1):
#   SKIP_EXISTING=0 ...   # rerun everything
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${MERFISH5_ROOT:?source work/env.sh first}"
: "${MERFISH5_UPSTREAM:?source work/env.sh first}"
WORK="$MERFISH5_ROOT"
LOGS="$WORK/logs/matrix"
SCENARIOS=(M1a M1b M1c M2a M2b M2c M3a)
GPUS_STR="${MERFISH5_GPUS:-0 1 4 5}"
GPU_LIST=(${GPUS_STR//,/ })
SKIP_EXISTING="${SKIP_EXISTING:-1}"
mkdir -p "$LOGS"

ASSIGN=(
  "UniST"
  "DeepSpatial isoST"
  "NTF Spateo SINTER3D"
  "PASTE-OT-Linear Ours-EncMTP ReHo SpatialZ"
)

run_worker() {
  local gpu="$1"; shift
  export CUDA_VISIBLE_DEVICES="$gpu"
  local m s out rc
  for m in "$@"; do
    for s in "${SCENARIOS[@]}"; do
      out="$WORK/predictions/$s/$m.npz"
      if [ "$SKIP_EXISTING" = "1" ] && [ -s "$out" ]; then
        echo "[skip] GPU$gpu $m $s" | tee -a "$LOGS/run.log"
        continue
      fi
      mkdir -p "$WORK/predictions/$s"
      echo "[run ] GPU$gpu $m $s" | tee -a "$LOGS/run.log"
      ( cd "$ROOT" && SCENARIO="$s" bash scripts/run_merfish5_small.sh "$m" ) \
        > "$LOGS/${m}_${s}.log" 2>&1
      rc=$?
      echo "[done] GPU$gpu $m $s rc=$rc" | tee -a "$LOGS/run.log"
    done
  done
}

for i in "${!ASSIGN[@]}"; do
  run_worker "${GPU_LIST[$i]}" ${ASSIGN[$i]} &
done
wait

echo "[validate] all scenarios" | tee -a "$LOGS/run.log"
fail=0
for s in "${SCENARIOS[@]}"; do
  "$PY_COMMON" scripts/validate_predictions.py \
    --canonical "$WORK/canonical" --predictions "$WORK/predictions/$s" \
    --scenario "$s" > "$LOGS/validate_$s.log" 2>&1 || fail=1
  tail -1 "$LOGS/validate_$s.log" | tee -a "$LOGS/run.log"
done
[ "$fail" = 0 ] && echo "[matrix] all prediction contracts pass" || echo "[matrix] failures; see $LOGS"
exit "$fail"

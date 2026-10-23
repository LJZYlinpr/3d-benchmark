#!/usr/bin/env bash
# Run one held-out scenario (default M1a) for the MERFISH 5-slice methods.
#
# Prerequisites:
#   $MERFISH5_ROOT/canonical/{bins_all.h5ad,scenarios.json,target_gt.npz,READY}
#   $MERFISH5_UPSTREAM/{SINTER3D/SINTER3D-master,isoST,SpatialZ,DeepSpatial}
#   a Python environment with torch/anndata/ot/scipy (+ method extras)
#
# Usage:
#   export MERFISH5_ROOT=/path/to/all_methods
#   export MERFISH5_UPSTREAM=/path/to/upstream/methods
#   export PY_COMMON=/path/to/env/bin/python
#   ./scripts/run_merfish5_small.sh                 # the six runnable methods
#   ./scripts/run_merfish5_small.sh ReHo SpatialZ   # subset
#
# NTF/Spateo/UniST need additional upstream code and are not in this wrapper.
# isoST is selectable but needs CUDA-enabled torch_cluster/torch_sparse.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${MERFISH5_ROOT:?set MERFISH5_ROOT to the experiment root containing canonical/}"
: "${MERFISH5_UPSTREAM:?set MERFISH5_UPSTREAM to the upstream methods directory}"
PY_COMMON="${PY_COMMON:-python}"
PY_DEEPSPATIAL="${PY_DEEPSPATIAL:-$PY_COMMON}"
SCENARIO="${SCENARIO:-M1a}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

if [ ! -f "$MERFISH5_ROOT/canonical/READY" ]; then
  echo "missing $MERFISH5_ROOT/canonical/READY" >&2
  echo "build canonical first; see docs/REPRO_SERVER1.md" >&2
  exit 1
fi
mkdir -p "$MERFISH5_ROOT/predictions" "$MERFISH5_ROOT/methods"

METHODS=("$@")
if [ ${#METHODS[@]} -eq 0 ]; then
  METHODS=(ReHo Ours-EncMTP PASTE-OT-Linear SpatialZ SINTER3D DeepSpatial)
fi

run_one() {
  local method="$1" script="$2"; shift 2
  local py="$PY_COMMON"
  case "$method" in
    DeepSpatial) py="$PY_DEEPSPATIAL" ;;
  esac
  echo "=== $method ($SCENARIO) ==="
  ( cd "$ROOT" && "$py" "$script" "$@" )
  local rc=$?
  echo "=== $method rc=$rc ==="
  return $rc
}

fail=0
for method in "${METHODS[@]}"; do
  case "$method" in
    ReHo)
      run_one ReHo src/methods/merfish/ReHo/run_merfish5.py --scenarios "$SCENARIO" || fail=1 ;;
    Ours-EncMTP)
      run_one Ours-EncMTP src/methods/merfish/Ours-EncMTP/run_merfish5.py --scenarios "$SCENARIO" --steps 3000 || fail=1 ;;
    PASTE-OT-Linear)
      run_one PASTE-OT-Linear src/methods/merfish/PASTE-OT-Linear/run_merfish5.py --scenarios "$SCENARIO" || fail=1 ;;
    SpatialZ)
      run_one SpatialZ src/methods/merfish/SpatialZ/run_merfish5.py --scenario "$SCENARIO" || fail=1 ;;
    SINTER3D)
      run_one SINTER3D src/methods/merfish/SINTER3D/run_merfish5.py --scenario "$SCENARIO" --steps 3000 || fail=1 ;;
    DeepSpatial)
      run_one DeepSpatial src/methods/merfish/DeepSpatial/run_merfish5.py --scenario "$SCENARIO" --epochs 20 || fail=1 ;;
    isoST)
      run_one isoST src/methods/merfish/isoST/run_merfish5.py --scenario "$SCENARIO" --epochs 20,20,20 || fail=1 ;;
    *)
      echo "unknown method: $method" >&2; fail=1 ;;
  esac
done
exit $fail

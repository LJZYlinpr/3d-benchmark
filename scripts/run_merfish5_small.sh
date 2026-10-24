#!/usr/bin/env bash
# Run one held-out scenario (default M1a) for the MERFISH 5-slice methods.
#
# Prerequisites:
#   $MERFISH5_ROOT/canonical/{bins_all.h5ad,scenarios.json,target_gt.npz,READY}
#   $MERFISH5_UPSTREAM/{SINTER3D,isoST,SpatialZ,DeepSpatial,...}
#   a Python environment with torch/anndata/ot/scipy (+ method extras)
#
# Usage:
#   export MERFISH5_ROOT=/path/to/work
#   ./scripts/run_merfish5_small.sh                 # all ten methods
#   ./scripts/run_merfish5_small.sh ReHo SpatialZ   # subset
#
# Upstream code is vendored under vendor/methods by default; no fetch step.
# isoST uses vendored/shims/torch_cluster when CUDA PyG is unavailable.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${MERFISH5_ROOT:?set MERFISH5_ROOT to the experiment root containing canonical/}"
MERFISH5_UPSTREAM="${MERFISH5_UPSTREAM:-$ROOT/vendor/methods}"
export MERFISH5_UPSTREAM
export PYTHONPATH="${PYTHONPATH:-}:$ROOT/vendor/shims"
PY_COMMON="${PY_COMMON:-python}"
PY_DEEPSPATIAL="${PY_DEEPSPATIAL:-$PY_COMMON}"
PY_UNIST="${PY_UNIST:-$PY_COMMON}"
PY_ISOST="${PY_ISOST:-$PY_COMMON}"
SCENARIO="${SCENARIO:-M1a}"
PILOT="${PILOT:-0}"
if [ "$PILOT" = "1" ]; then
  OURS_STEPS=20; SINTER_STEPS=10; DEEP_EPOCHS=1; ISO_EPOCHS=1,1,1
  NTF_ITERS=10; SPATEO_ITERS=20; UNIST_GAE=1; UNIST_INR=1
else
  OURS_STEPS=3000; SINTER_STEPS=3000; DEEP_EPOCHS=20; ISO_EPOCHS=20,20,20
  NTF_ITERS=12000; SPATEO_ITERS=60000; UNIST_GAE=50; UNIST_INR=100
fi
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
  METHODS=(ReHo Ours-EncMTP PASTE-OT-Linear SpatialZ SINTER3D DeepSpatial isoST NTF Spateo UniST)
fi

prepare_visible() {
  local method="$1"
  local out="$MERFISH5_ROOT/$method/visible_${SCENARIO}.h5ad"
  "$PY_COMMON" scripts/make_visible_observed.py \
    --observed "$MERFISH5_ROOT/observed_50um.h5ad" \
    --scenarios "$MERFISH5_ROOT/canonical/scenarios.json" \
    --combo "$SCENARIO" --out "$out"
}

holdout_iz() {
  "$PY_COMMON" - "$MERFISH5_ROOT" "$SCENARIO" <<'PYEOF'
import json, sys, numpy as np
root, combo = sys.argv[1], sys.argv[2]
sec = json.load(open(root + "/canonical/sections.json"))
sc = json.load(open(root + "/canonical/scenarios.json"))
spec = next(x for x in sc["scenarios"] if x["name"] == combo)
target = spec["target"][0]
z = sec[target]["z_um"] / 1000.0
g = np.load(root + "/grid_50um.npz")
print(int(np.argmin(np.abs(g["z_centers"] - z))))
PYEOF
}

run_one() {
  local method="$1" script="$2"; shift 2
  local py="$PY_COMMON"
  case "$method" in
    DeepSpatial) py="$PY_DEEPSPATIAL" ;;
    UniST) py="$PY_UNIST" ;;
    isoST) py="$PY_ISOST" ;;
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
      run_one Ours-EncMTP src/methods/merfish/Ours-EncMTP/run_merfish5.py --scenarios "$SCENARIO" --steps "$OURS_STEPS" || fail=1 ;;
    PASTE-OT-Linear)
      run_one PASTE-OT-Linear src/methods/merfish/PASTE-OT-Linear/run_merfish5.py --scenarios "$SCENARIO" || fail=1 ;;
    SpatialZ)
      run_one SpatialZ src/methods/merfish/SpatialZ/run_merfish5.py --scenario "$SCENARIO" || fail=1 ;;
    SINTER3D)
      run_one SINTER3D src/methods/merfish/SINTER3D/run_merfish5.py --scenario "$SCENARIO" --steps "$SINTER_STEPS" || fail=1 ;;
    DeepSpatial)
      run_one DeepSpatial src/methods/merfish/DeepSpatial/run_merfish5.py --scenario "$SCENARIO" --epochs "$DEEP_EPOCHS" || fail=1 ;;
    NTF)
      prepare_visible NTF || fail=1
      MERFISH5_OBSERVED="$MERFISH5_ROOT/NTF/visible_${SCENARIO}.h5ad" \
      MERFISH5_NTF_OUT="$MERFISH5_ROOT/NTF" \
      MERFISH5_GRID="$MERFISH5_ROOT/grid_50um.npz" \
        run_one NTF src/methods/merfish/NTF/run_merfish5.py --n_iter "$NTF_ITERS" --batch_size 8192 || fail=1
      "$PY_COMMON" scripts/volume_to_target.py \
        --volume "$MERFISH5_ROOT/NTF/NTF_50um_uniform.h5ad" \
        --target-gt "$MERFISH5_ROOT/canonical/target_gt.npz" \
        --scenario "$SCENARIO" --method NTF \
        --out "$MERFISH5_ROOT/predictions/$SCENARIO/NTF.npz" || fail=1
      ;;
    Spateo)
      eval "$("$ROOT/scripts/make_spateo_shim.sh" "$ROOT/spateo_shim")"
      HO=$(holdout_iz)
      run_one Spateo src/methods/merfish/Spateo/run_merfish5.py --iters "$SPATEO_ITERS" --holdout-iz "$HO" || fail=1
      "$PY_COMMON" scripts/points_to_target.py \
        --points-npz "$MERFISH5_ROOT/Spateo/holdout_pred_m1a.npz" \
        --target-gt "$MERFISH5_ROOT/canonical/target_gt.npz" \
        --scenario "$SCENARIO" --method Spateo \
        --out "$MERFISH5_ROOT/predictions/$SCENARIO/Spateo.npz" || fail=1
      ;;
    isoST)
      "$PY_ISOST" "$ROOT/scripts/patch_pyg_torch_cluster.py" || true
      run_one isoST src/methods/merfish/isoST/run_merfish5.py --scenario "$SCENARIO" --epochs "$ISO_EPOCHS" || fail=1 ;;
    UniST)
      run_one UniST src/methods/merfish/UniST/run_merfish5.py --scenario "$SCENARIO" --gae-epochs "$UNIST_GAE" --inr-epochs "$UNIST_INR" || fail=1 ;;
    *)
      echo "unknown method: $method" >&2; fail=1 ;;
  esac
done
exit $fail

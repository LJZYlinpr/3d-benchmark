#!/usr/bin/env bash
# One-command MERFISH 5-slice M1a reproduction from a fresh clone.
#
#   bash scripts/reproduce_m1a.sh
#   bash scripts/reproduce_m1a.sh ReHo SpatialZ
#   MERFISH5_PROFILE=ada bash scripts/reproduce_m1a.sh
#   SKIP_ENV=1 MERFISH5_ROOT=/path/to/work bash scripts/reproduce_m1a.sh
#
# Data shards and method code are vendored in this repository.  The only
# network access is the Python environment setup (pip/PyTorch wheels).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export MERFISH5_ROOT="${MERFISH5_ROOT:-$REPO/work}"
export MERFISH5_UPSTREAM="${MERFISH5_UPSTREAM:-$REPO/vendor/methods}"
export MERFISH5_SCENARIO="${MERFISH5_SCENARIO:-M1a}"
export SCENARIO="$MERFISH5_SCENARIO"
mkdir -p "$MERFISH5_ROOT"

if [ ! -f "$MERFISH5_ROOT/env.sh" ] && [ "${SKIP_ENV:-0}" != "1" ]; then
  if [ "${MERFISH5_PROFILE:-}" = "ada" ]; then
    bash "$REPO/scripts/setup_merfish5_env_ada.sh"
  else
    bash "$REPO/scripts/setup_merfish5_env.sh"
  fi
fi
if [ -f "$MERFISH5_ROOT/env.sh" ]; then
  # shellcheck disable=SC1090
  source "$MERFISH5_ROOT/env.sh"
fi
PY_BUILD="${PY_BUILD:-${PY_COMMON:-python3}}"
PY_METRIC="${PY_METRIC:-${PY_COMMON:-python3}}"
PY_ISOST="${PY_ISOST:-${PY_COMMON:-python3}}"
export PYTHONPATH="${PYTHONPATH:-}:$REPO/vendor/shims"
"$PY_ISOST" "$REPO/scripts/patch_pyg_torch_cluster.py" || true

echo "== build observed =="
"$PY_BUILD" "$REPO/scripts/build_observed_50um.py" \
  --shards-dir "$REPO/data/merfish_5slice" \
  --out "$MERFISH5_ROOT/observed_50um.h5ad"

echo "== build targets =="
MERFISH5_ROOT="$MERFISH5_ROOT" "$PY_BUILD" "$REPO/src/canonical/make_targets.py"

echo "== build canonical =="
MERFISH5_ROOT="$MERFISH5_ROOT" "$PY_BUILD" "$REPO/src/canonical/merfish_5slice.py"
cp -f "$REPO/data/merfish_5slice/grid_50um.npz" "$MERFISH5_ROOT/grid_50um.npz"

METHODS=("$@")
if [ ${#METHODS[@]} -eq 0 ]; then
  METHODS=(ReHo Ours-EncMTP PASTE-OT-Linear SpatialZ SINTER3D DeepSpatial isoST NTF Spateo UniST)
fi

echo "== run ${#METHODS[@]} methods on $MERFISH5_SCENARIO =="
bash "$REPO/scripts/run_merfish5_small.sh" "${METHODS[@]}"

echo "== validate predictions =="
"$PY_METRIC" "$REPO/scripts/validate_predictions.py" \
  --canonical "$MERFISH5_ROOT/canonical" \
  --predictions "$MERFISH5_ROOT/predictions/$MERFISH5_SCENARIO"

echo "== expression metrics =="
"$PY_METRIC" "$REPO/scripts/evaluate_merfish5_predictions.py" \
  --canonical "$MERFISH5_ROOT/canonical" \
  --predictions-root "$MERFISH5_ROOT/predictions" \
  --modules "$REPO/manifests/merfish_5slice_modules.json" \
  --scenarios "$MERFISH5_SCENARIO" \
  --out "$MERFISH5_ROOT/metrics_${MERFISH5_SCENARIO}.csv"

if [ "${PILOT:-0}" = "1" ]; then
  echo "[pilot] skipping strict metric validation"
  echo "[done] predictions: $MERFISH5_ROOT/predictions/$MERFISH5_SCENARIO"
  exit 0
fi
if [ ${#METHODS[@]} -eq 10 ]; then
  SUBSET=""
else
  SUBSET="--allow-subset"
fi
"$PY_METRIC" "$REPO/scripts/validate_reproduction.py" \
  --repo "$REPO" \
  --canonical "$MERFISH5_ROOT/canonical" \
  --pred-dir "$MERFISH5_ROOT/predictions/$MERFISH5_SCENARIO" \
  $SUBSET

echo "[done] predictions: $MERFISH5_ROOT/predictions/$MERFISH5_SCENARIO"

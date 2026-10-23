#!/usr/bin/env bash
# Build a minimal `spateo` package shim that bypasses the upstream
# segmentation/alignment imports (pyvista, cv2, csbdeep, gpytorch, ...).
# The DeepInterpolation module itself does not need them.
#
# Usage:
#   export MERFISH5_UPSTREAM=/path/to/upstream/methods
#   eval "$(./scripts/make_spateo_shim.sh)"   # prints PYTHONPATH export
set -euo pipefail
: "${MERFISH5_UPSTREAM:?set MERFISH5_UPSTREAM}"
OUT="${1:-$PWD/spateo_shim}"
REAL="$MERFISH5_UPSTREAM/Spateo/spateo-release/spateo"
mkdir -p "$OUT/spateo/tdr/interpolations"
printf '__path__ = ["%s", "%s"]\n' "$OUT/spateo" "$REAL" > "$OUT/spateo/__init__.py"
printf '__path__ = ["%s", "%s"]\n' "$OUT/spateo/tdr" "$REAL/tdr" > "$OUT/spateo/tdr/__init__.py"
printf '__path__ = ["%s", "%s"]\n' "$OUT/spateo/tdr/interpolations" "$REAL/tdr/interpolations" > "$OUT/spateo/tdr/interpolations/__init__.py"
echo "export PYTHONPATH=\"$OUT:$MERFISH5_UPSTREAM/Spateo/spateo-release:\${PYTHONPATH:-}\""

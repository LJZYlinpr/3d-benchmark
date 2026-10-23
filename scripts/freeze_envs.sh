#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/envs/full"
for e in spateo ntf SINTER3D_env isoST spatialz deepspatial unist benchmark; do
  if conda env list | awk '{print $1}' | grep -qx "$e"; then
    conda env export -n "$e" > "$ROOT/envs/full/${e}.yml"
    conda run -n "$e" pip freeze > "$ROOT/envs/full/${e}.pip.txt" 2>/dev/null || true
  fi
done

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CELLS="${1:?usage: $0 cells.npz [out_dir]}"
OUT="${2:-$ROOT/grids}"
mkdir -p "$OUT"
python "$ROOT/src/benchmark/grid.py" \
  --cells "$CELLS" \
  --out-dir "$OUT" \
  --resolutions 10 20 40 100

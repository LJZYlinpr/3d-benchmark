#!/usr/bin/env bash
# Download the MERFISH 5-slice data package (separate from the Git code).
#
# The five lossless per-slice shards + grid are a GitHub release asset:
#   data-v1 / merfish_5slice_data_v1.tar.gz
# They are not tracked in this repository.
#
# Usage:
#   bash scripts/fetch_data_merfish5.sh [DEST_DIR]
#   MERFISH5_DATA_TARBALL=/path/to/merfish_5slice_data_v1.tar.gz \
#     bash scripts/fetch_data_merfish5.sh
#
# The default destination is data/merfish_5slice/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/data/merfish_5slice}"
URL="${MERFISH5_DATA_URL:-https://github.com/LJZYlinpr/3d-benchmark/releases/download/data-v1/merfish_5slice_data_v1.tar.gz}"
SHA="187af1cc750634512b58c8333a86eb94dec5b619bd24b2074319f94fcc67ab08"

need=(observed_50um_S1.h5ad observed_50um_S2.h5ad observed_50um_S3.h5ad \
      observed_50um_S4.h5ad observed_50um_S5.h5ad grid_50um.npz)
if [ -d "$DEST" ]; then
  ok=1
  for f in "${need[@]}"; do [ -s "$DEST/$f" ] || ok=0; done
  if [ "$ok" = 1 ]; then echo "[data] already present: $DEST"; exit 0; fi
fi

mkdir -p "$DEST"
tmp="$DEST/.merfish_5slice_data_v1.tar.gz.part"
if [ -n "${MERFISH5_DATA_TARBALL:-}" ]; then
  cp -f "$MERFISH5_DATA_TARBALL" "$tmp"
else
  echo "[data] download $URL"
  curl -L --retry 5 --retry-delay 5 -C - -o "$tmp" "$URL"
fi
echo "$SHA  $tmp" | sha256sum -c -
tar -xzf "$tmp" --strip-components=1 -C "$DEST"
rm -f "$tmp"
for f in "${need[@]}"; do [ -s "$DEST/$f" ] || { echo "missing $DEST/$f" >&2; exit 1; }; done
echo "[data] ready: $DEST"

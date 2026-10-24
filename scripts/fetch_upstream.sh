#!/usr/bin/env bash
# Fetch the upstream method trees at the commits recorded in
# docs/upstream/method_commits.tsv.
#
# The upstream trees are intentionally not committed to this repository.
# GitHub tarballs (codeload.github.com) are used instead of git clone because
# some cluster nodes have a broken git TLS stack but working HTTPS downloads.
#
# Usage:
#   export MERFISH5_UPSTREAM=/path/to/upstream/methods
#   ./scripts/fetch_upstream.sh
#
# Re-running is safe: completed trees are skipped, partial tarballs resume.
# SINTER3D has no public commit anchor; supply SINTER3D/SINTER3D-master
# separately (release asset or project snapshot).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="${MERFISH5_UPSTREAM:-$ROOT/upstream/methods}"
CACHE="$UPSTREAM/.cache"
mkdir -p "$UPSTREAM" "$CACHE"

fetch_tar() {
  local owner="$1" repo="$2" commit="$3" dir="$4"
  if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "[skip] $dir"
    return
  fi
  local tar="$CACHE/${repo}-${commit}.tar.gz"
  local url="https://codeload.github.com/$owner/$repo/tar.gz/$commit"
  if [ -s "$tar" ] && ! tar -tzf "$tar" >/dev/null 2>&1; then
    echo "[discard] incomplete cached tarball: $tar" >&2
    rm -f "$tar"
  fi
  if [ ! -s "$tar" ]; then
    echo "[download] $url"
    curl -L --retry 5 --retry-delay 5 -C - -o "$tar" "$url"
  fi
  if ! tar -tzf "$tar" >/dev/null 2>&1; then
    echo "[error] invalid tarball after download: $tar" >&2
    rm -f "$tar"
    exit 1
  fi
  echo "[extract] $dir"
  mkdir -p "$dir"
  tar -xzf "$tar" --strip-components=1 -C "$dir"
}

apply_patch() {
  local dir="$1" patch="$2"
  if patch -p1 -R --dry-run -d "$dir" < "$patch" >/dev/null 2>&1; then
    echo "[skip] patch already applied: $patch"
  else
    echo "[patch] $patch"
    patch -p1 -d "$dir" < "$patch"
  fi
}

copy_new_files() {
  local src="$1" dst="$2"
  [ -d "$src" ] || return
  echo "[new-files] $src -> $dst"
  ( cd "$src" && find . -type f -print0 | while IFS= read -r -d '' f; do
      mkdir -p "$dst/$(dirname "$f")"
      cp "$f" "$dst/$f"
    done )
}

fetch_tar GYQ-form NTF f0b4941a07330d9749567ef52df8bc9f5cef976a "$UPSTREAM/NTF"
apply_patch "$UPSTREAM/NTF" "$ROOT/docs/upstream/patches/NTF_torch_fallback.patch"

fetch_tar deng-ai-lab isoST 805981c3fad66e3c9f5409816f95c17332906600 "$UPSTREAM/isoST"
apply_patch "$UPSTREAM/isoST" "$ROOT/docs/upstream/patches/isoST.patch"
copy_new_files "$ROOT/docs/upstream/new_files/isoST" "$UPSTREAM/isoST"

fetch_tar lanshui98 UniST a2fd35a70a6080caab5fdd60652242f85059b09a "$UPSTREAM/UniST"
apply_patch "$UPSTREAM/UniST" "$ROOT/docs/upstream/patches/UniST.patch"
copy_new_files "$ROOT/docs/upstream/new_files/UniST" "$UPSTREAM/UniST"

fetch_tar aristoteleo spateo-release cbea5b322316dd1d3cbcd06bd0bc61368fe0bff3 "$UPSTREAM/Spateo/spateo-release"

fetch_tar senlin-lin SpatialZ e1b5ed01e933729149e08399febca08db16ced14 "$UPSTREAM/SpatialZ"
fetch_tar yyh030806 DeepSpatial fdb98657fca95679cc45adf3510a161de05bca08 "$UPSTREAM/DeepSpatial"

if [ -d "$UPSTREAM/SINTER3D/SINTER3D-master" ]; then
  echo "[info] using supplied SINTER3D tree under $UPSTREAM/SINTER3D"
else
  echo "[info] SINTER3D is vendored under vendor/methods/SINTER3D/SINTER3D-master; no fetch needed"
fi

echo "[done] upstream root: $UPSTREAM"

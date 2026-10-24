#!/usr/bin/env bash
# Read-only preflight for the MERFISH 5-slice reproduction.
# Prints a profile recommendation and actionable warnings; installs nothing.
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
FAIL=0
WARN=0
say()  { printf '%s\n' "$*"; }
ok()   { printf '[ok] %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*"; WARN=$((WARN+1)); }
bad()  { printf '[FAIL] %s\n' "$*"; FAIL=$((FAIL+1)); }

say "== MERFISH5 preflight =="
say "repo: $REPO"

# OS / arch
case "$(uname -s)" in Linux) ok "Linux $(uname -m)";; *) bad "expected Linux, got $(uname -s)";; esac

# Python
if command -v "$PY" >/dev/null 2>&1; then
  PYVER="$("$PY" -c 'import sys; print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo unknown)"
  case "$PYVER" in
    3.9|3.10) ok "Python $PYVER";;
    3.11|3.12|3.13) bad "Python $PYVER: install 3.10 (common/Ada pin anndata 0.9.1, numba, PyG)";;
    *) warn "Python $PYVER: untested; use 3.9/3.10";;
  esac
else
  bad "Python '$PY' not found; set PYTHON=/path/to/python3.10"
fi

# GPU
CAP=""
GPU_MEM=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LINE="$(nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader 2>/dev/null | head -1)"
  if [ -z "$GPU_LINE" ]; then
    bad "nvidia-smi found but no GPU visible"
  else
    ok "GPU: $GPU_LINE"
    GPU_MEM="$(printf '%s' "$GPU_LINE" | awk -F, '{gsub(/[^0-9.]/,"",$2); print $2+0}')"
    CAP="$(printf '%s' "$GPU_LINE" | awk -F, '{gsub(/ /,"",$3); print $3}')"
    [ "${GPU_MEM:-0}" -lt 10000 ] && warn "GPU has <10GB; isoST/DeepSpatial may OOM (run methods individually)"
  fi
else
  bad "nvidia-smi not found; a CUDA GPU is required"
fi

# Profile recommendation
PROFILE="common"
if [ -n "$CAP" ]; then
  case "$CAP" in
    8.9|9.0) PROFILE="ada";;
  esac
  ok "recommended profile: $PROFILE (compute capability $CAP)"
else
  warn "cannot determine compute capability; default profile: common"
fi

# CUDA / nvcc
ADA_CUDA_HOME="${ADA_CUDA_HOME:-/usr/local/cuda-12.1}"
if [ "$PROFILE" = "ada" ]; then
  if [ -x "$ADA_CUDA_HOME/bin/nvcc" ]; then
    ok "Ada CUDA toolkit: $ADA_CUDA_HOME"
  else
    warn "no nvcc at $ADA_CUDA_HOME; set ADA_CUDA_HOME or install CUDA 12.1 (isoST/KeOps will fail without it)"
  fi
else
  if command -v nvcc >/dev/null 2>&1; then
    ok "nvcc: $(command -v nvcc)"
  else
    warn "no system nvcc; common setup creates a local CUDA 11.7 toolkit (isoST/KeOps)"
  fi
fi

# Disk
FREE_GB="$(df -Pk "$REPO" | awk 'NR==2 {print int($4/1024/1024)}')"
if [ "${FREE_GB:-0}" -ge 15 ]; then
  ok "free disk: ${FREE_GB} GB"
else
  bad "free disk: ${FREE_GB} GB; need >=15 GB for envs, canonical data and predictions"
fi
say "      work dir: ${MERFISH5_ROOT:-$REPO/work}"

# Data presence + sizes
DDIR="$REPO/data/merfish_5slice"
if [ -f "$DDIR/SHA256SUMS" ] && [ -f "$DDIR/observed_50um_S1.h5ad" ]; then
  ok "data shards + SHA256SUMS present"
else
  bad "missing data/merfish_5slice files; re-clone the repository"
fi

# Network probes (best effort)
probe() {
  local url="$1" name="$2"
  if command -v curl >/dev/null 2>&1; then
    if curl -LIsS --max-time 8 "$url" >/dev/null 2>&1; then ok "$name reachable"; else warn "$name unreachable (setup may need a mirror/manual wheels)"; fi
  else
    warn "curl not found; skip $name probe"
  fi
}
probe https://pypi.org/simple/ "PyPI"
probe https://pypi.tuna.tsinghua.edu.cn/simple/ "Tsinghua PyPI mirror"
probe https://data.pyg.org/whl/ "PyG wheels"

# Summary
say ""
if [ "$FAIL" -gt 0 ]; then
  say "== BLOCKED: $FAIL hard issue(s), $WARN warning(s) =="
  exit 1
fi
say "== READY ($WARN warning(s)) =="
say "Next:"
if [ "$PROFILE" = "ada" ]; then
  say "  bash scripts/setup_merfish5_env_ada.sh"
else
  say "  bash scripts/setup_merfish5_env.sh"
fi
say "  bash scripts/reproduce_m1a.sh"
say "If setup fails, see docs/AI_REPRODUCTION_RUNBOOK.md."

#!/usr/bin/env bash
# Bootstrap the MERFISH 5-slice reproduction environments.
#
# This script is deliberately explicit: the methods need incompatible
# torch/CUDA/PyG stacks, and several packages have platform-specific wheels.
# It creates everything under $MERFISH5_ROOT (default: ./work) and never
# modifies the system/conda base environment.
#
# Usage:
#   bash scripts/setup_merfish5_env.sh            # create/update envs
#   bash scripts/setup_merfish5_env.sh --check    # report what is missing
#
# Requirements: Linux x86_64, Python 3.9/3.10, NVIDIA driver, internet.
# Optional: conda (needed for the local CUDA 11.7 toolkit used by isoST/KeOps).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${MERFISH5_ROOT:-$REPO/work}"
UP="${MERFISH5_UPSTREAM:-$REPO/vendor/methods}"
PY="${PYTHON:-python3}"
if [ -z "${PYTHON:-}" ]; then
  for cand in python3.10 python3.9; do
    if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
  done
fi
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1
[ "${1:-}" = "--help" ] && { sed -n '2,30p' "$0"; exit 0; }

mkdir -p "$ROOT" "$UP"
COMMON="$ROOT/venv"
OMNI="$ROOT/venv_omni"

say() { printf '\n== %s ==\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

say "prerequisites"
echo "repo=$REPO"
echo "experiment root=$ROOT"
echo "upstream=$UP"
echo "python=$($PY -V 2>&1)"
if have nvidia-smi; then nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; else
  echo "WARNING: nvidia-smi not found; GPU methods will not run" >&2
fi

if [ "$CHECK" != "1" ]; then
  "$PY" - <<'PYVER'
import sys
if sys.version_info[:2] not in ((3, 9), (3, 10)):
    raise SystemExit(
        f"Python {sys.version.split()[0]} is not supported by torch 1.13/PyG 2.3; "
        "use Python 3.9 or 3.10 (set PYTHON=python3.10).")
PYVER
fi

if [ "$CHECK" = "1" ]; then
  echo "common env: $([ -x "$COMMON/bin/python" ] && echo present || echo MISSING)"
  echo "omni env:   $([ -x "$OMNI/bin/python" ] && echo present || echo MISSING)"
  echo "cuda nvcc:  $([ -x "$ROOT/cuda117/bin/nvcc" ] && echo present || echo MISSING)"
  echo "vendor SINTER3D: $([ -d "$REPO/vendor/methods/SINTER3D/SINTER3D-master" ] && echo present || echo MISSING)"
  echo "upstream methods: $(ls "$UP" 2>/dev/null | tr '\n' ' ')"
  exit 0
fi

# ---------------------------------------------------------------------------
# 1. Common env: torch 1.13.1 + CUDA PyG + KeOps (isoST/NTF/Spateo/core)
# ---------------------------------------------------------------------------
say "common venv"
if [ ! -x "$COMMON/bin/python" ]; then
  "$PY" -m venv --system-site-packages "$COMMON"
fi
PIP="$COMMON/bin/pip"
"$PIP" install -q --upgrade pip wheel
"$PIP" install "torch==1.13.1+cu117" \
  --index-url https://download.pytorch.org/whl/cu117
"$PIP" install torchdiffeq geomloss "torch_geometric==2.3.1"
"$PIP" install "nvidia-cuda-runtime-cu11==11.7.99" \
  "nvidia-cuda-nvrtc-cu11==11.8.89" "pykeops==2.2.3"
"$PIP" install "numpy==1.22.2" "pandas==2.0.0" "scipy==1.10.1" \
  "anndata==0.9.1" "scikit-learn==1.2.2" "scanpy==1.9.3" \
  "statsmodels==0.14.1" "h5py==3.11.0" "scikit-image==0.20.0" \
  "pot==0.9.3" "squidpy==1.2.3" shapely "numba==0.55.2" "llvmlite==0.38.1" \
  pyyaml tqdm matplotlib seaborn
"$PIP" install --no-deps "SOMENDER==1.1"
"$PIP" install "setuptools==80.9.0"

say "CUDA PyG extensions"
if ! "$PIP" install torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-1.13.1+cu117.html; then
  cat >&2 <<'EOF'
ERROR: could not install the CUDA PyG wheels (data.pyg.org may be blocked).
Download these on a machine with access, copy them to this host, then run:
  $COMMON/bin/pip install --force-reinstall --no-deps \
    torch_cluster-1.6.1+pt113cu117-cp39-cp39-linux_x86_64.whl \
    torch_sparse-0.6.17+pt113cu117-cp39-cp39-linux_x86_64.whl \
    torch_scatter-2.1.1+pt113cu117-cp39-cp39-linux_x86_64.whl
EOF
  exit 1
fi

say "patch PyG optional torch_cluster ops"
"$COMMON/bin/python" "$REPO/scripts/patch_pyg_torch_cluster.py"

say "Spateo runtime deps"
"$PIP" install --no-deps pyvista vtk opencv-python-headless colorcet \
  "kornia==0.6.12" scooby
"$PIP" install --force-reinstall --no-deps "numpy==1.22.2"

# ---------------------------------------------------------------------------
# 2. Local CUDA 11.7 toolkit + KeOps compat tree
# ---------------------------------------------------------------------------
say "CUDA toolkit for KeOps"
CUDA_ENV="$ROOT/cuda117"
CUDA_COMPAT="$ROOT/cuda117_compat"
if [ ! -x "$CUDA_ENV/bin/nvcc" ]; then
  if have conda; then
    conda create -y -p "$CUDA_ENV" -c nvidia -c conda-forge \
      cuda-nvcc=11.7 cuda-cudart-dev=11.7 cuda-nvrtc-dev=11.7
  else
    echo "WARNING: conda not found; install a CUDA 11.7 toolkit or isoST will not run" >&2
  fi
fi
NVRTC_DIR=""
if [ -x "$CUDA_ENV/bin/nvcc" ]; then
  mkdir -p "$CUDA_COMPAT/bin" "$CUDA_COMPAT/include" "$CUDA_COMPAT/lib64"
  for f in "$CUDA_ENV"/bin/*; do ln -sfn "$f" "$CUDA_COMPAT/bin/"; done
  for f in "$CUDA_ENV"/include/*; do ln -sfn "$f" "$CUDA_COMPAT/include/"; done
  RUNTIME_LIB="$(find "$COMMON" -path '*nvidia/cuda_runtime/lib/libcudart.so.11.0' | head -1)"
  NVRTC_LIB="$(find "$COMMON" -path '*nvidia/cuda_nvrtc/lib/libnvrtc.so.11.2' | head -1)"
  if [ -n "$NVRTC_LIB" ]; then NVRTC_DIR="$(dirname "$NVRTC_LIB")"; fi
  [ -n "$RUNTIME_LIB" ] && ln -sfn "$RUNTIME_LIB" "$CUDA_COMPAT/lib64/libcudart.so"
  [ -n "$NVRTC_LIB" ] && ln -sfn "$NVRTC_LIB" "$CUDA_COMPAT/lib64/libnvrtc.so"
  BUILTINS="$(find "$NVRTC_DIR" -name 'libnvrtc-builtins.so.11.8' | head -1)"
  if [ -n "$BUILTINS" ]; then
    ln -sfn "$BUILTINS" "$NVRTC_DIR/libnvrtc-builtins.so.11.2"
    ln -sfn "$BUILTINS" "$CUDA_COMPAT/lib64/libnvrtc-builtins.so.11.2"
  fi
fi

# ---------------------------------------------------------------------------
# 3. Omni env: torch 2.6+ for DeepSpatial and UniST
# ---------------------------------------------------------------------------
if [ -n "$NVRTC_DIR" ]; then
  say "verify CUDA PyG extensions"
  if ! LD_LIBRARY_PATH="$NVRTC_DIR:$CUDA_COMPAT/lib64:${LD_LIBRARY_PATH:-}" \
      "$COMMON/bin/python" - <<'PYCHECK'
import torch
from torch_cluster import knn_graph
x = torch.randn(64, 3, device="cuda")
knn_graph(x, 6)
print("CUDA torch_cluster OK")
PYCHECK
  then
    echo "WARNING: torch_cluster is CPU-only or CUDA libraries are missing." >&2
    echo "Reinstall the CUDA PyG wheels (see the error block above); isoST needs them." >&2
  fi
fi

say "omni venv (torch 2.x)"
if [ ! -x "$OMNI/bin/python" ]; then
  if [ -n "${OMNI_BASE_PY:-}" ]; then
    "$OMNI_BASE_PY" -m venv --system-site-packages "$OMNI"
  else
    "$PY" -m venv "$OMNI"
  fi
fi
PIP2="$OMNI/bin/pip"
"$PIP2" install -q --upgrade pip wheel
if ! "$OMNI/bin/python" -c 'import torch; assert int(torch.__version__.split(".")[0]) >= 2' 2>/dev/null; then
  "$PIP2" install "torch==2.6.0" "torchvision==0.21.0" \
    --index-url https://download.pytorch.org/whl/cu124
fi
"$PIP2" install lightning tensorboardX omegaconf rich einops \
  "scikit-learn>=1.4" timm torchdiffeq pot pytorch-lightning torch_geometric \
  scanpy anndata
"$OMNI/bin/python" "$REPO/scripts/patch_pyg_torch_cluster.py"
if ! "$OMNI/bin/python" -c 'import torchvision' 2>/dev/null; then
  "$PIP2" install "torchvision==0.21.0" \
    --index-url https://download.pytorch.org/whl/cu124
fi

# ---------------------------------------------------------------------------
# 4. Environment file used by the wrapper
# ---------------------------------------------------------------------------
say "write $ROOT/env.sh"
NVRTC_EXPORT=""
if [ -n "${NVRTC_DIR:-}" ]; then
  NVRTC_EXPORT="export LD_LIBRARY_PATH=\"$NVRTC_DIR:\${LD_LIBRARY_PATH:-}\""
fi
cat > "$ROOT/env.sh" <<EOF
# Generated by scripts/setup_merfish5_env.sh; source this before running.
export MERFISH5_ROOT="$ROOT"
export MERFISH5_UPSTREAM="$UP"
export PYTHONPATH="${PYTHONPATH:-}:$REPO/vendor/shims"
export PY_COMMON="$COMMON/bin/python"
export PY_DEEPSPATIAL="$OMNI/bin/python"
export PY_UNIST="$OMNI/bin/python"
export UNIST_PY="$OMNI/bin/python"
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export CUDA_HOME="$CUDA_COMPAT"
export CUDA_PATH="$CUDA_COMPAT"
export PATH="$CUDA_ENV/bin:\$PATH"
$NVRTC_EXPORT
export LIBRARY_PATH="$CUDA_COMPAT/lib64:\${LIBRARY_PATH:-}"
EOF
chmod +x "$ROOT/env.sh"

say "done"
echo "source $ROOT/env.sh"
echo "next: bash scripts/reproduce_m1a.sh  (upstream vendored)"

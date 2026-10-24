#!/usr/bin/env bash
# Ada / RTX 4090 (sm_89) profile for the MERFISH 5-slice reproduction.
#
# The legacy setup_merfish5_env.sh targets torch 1.13.1+cu117 (Turing/Ampere).
# RTX 4090D needs a CUDA 12 build; this profile uses:
#   common: torch 2.3.1+cu121 + CUDA PyG 2.3 + KeOps + CUDA 12.1 nvcc
#   omni:   torch 2.6.0+cu124 + lightning/timm (DeepSpatial/UniST)
#
# It works without conda and without ensurepip (virtualenv fallback).
#
# Usage:
#   MERFISH5_PROFILE=ada bash scripts/setup_merfish5_env_ada.sh
#   MERFISH5_PROFILE=ada bash scripts/setup_merfish5_env_ada.sh --check
#
# Optional overrides:
#   TORCH_WHEEL=/path/torch-2.3.1+cu121-cp310-...whl
#   TORCH_OMNI_WHEEL=/path/torch-2.6.0+cu124-cp310-...whl
#   TORCHVISION_OMNI_WHEEL=/path/torchvision-0.21.0+cu124-cp310-...whl
#   ADA_CUDA_HOME=/usr/local/cuda-12.1
#   PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${MERFISH5_ROOT:-$REPO/work}"
UP="${MERFISH5_UPSTREAM:-$REPO/upstream/methods}"
PY="${PYTHON:-python3.10}"
ADA_CUDA_HOME="${ADA_CUDA_HOME:-/usr/local/cuda-12.1}"
PIP_INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1
[ "${1:-}" = "--help" ] && { sed -n '2,35p' "$0"; exit 0; }

mkdir -p "$ROOT" "$UP" "$ROOT/wheels"
COMMON="$ROOT/venv_ada"
OMNI="$ROOT/venv_omni"
say() { printf '\n== %s ==\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

say "prerequisites"
echo "repo=$REPO"; echo "experiment root=$ROOT"; echo "upstream=$UP"
echo "python=$($PY -V 2>&1)"
if have nvidia-smi; then nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader; else
  echo "WARNING: nvidia-smi not found" >&2
fi
if [ -x "$ADA_CUDA_HOME/bin/nvcc" ]; then
  echo "nvcc=$("$ADA_CUDA_HOME/bin/nvcc" --version | tail -2 | head -1)"
else
  echo "WARNING: no nvcc at $ADA_CUDA_HOME; isoST/KeOps will not run" >&2
fi

if [ "$CHECK" = "1" ]; then
  echo "common env: $([ -x "$COMMON/bin/python" ] && echo present || echo MISSING)"
  echo "omni env:   $([ -x "$OMNI/bin/python" ] && echo present || echo MISSING)"
  echo "cuda:       $([ -x "$ADA_CUDA_HOME/bin/nvcc" ] && echo present || echo MISSING)"
  echo "vendor SINTER3D: $([ -d "$REPO/vendor/SINTER3D/SINTER3D-master" ] && echo present || echo MISSING)"
  exit 0
fi

make_venv() {
  local path="$1"
  if "$PY" -m venv "$path" 2>/dev/null; then return 0; fi
  if "$PY" -m virtualenv "$path"; then return 0; fi
  echo "[venv] installing virtualenv" >&2
  "$PY" -m pip install --user --index-url "$PIP_INDEX" virtualenv
  "$PY" -m virtualenv "$path"
}

download() {
  local url="$1" out="$2"
  [ -s "$out" ] || curl -L --retry 5 --retry-delay 5 -C - -o "$out" "$url"
}

# ---------------------------------------------------------------------------
# 1. Common Ada env
# ---------------------------------------------------------------------------
say "common venv (torch 2.3.1+cu121)"
make_venv "$COMMON"
PIP="$COMMON/bin/pip"
"$PIP" install -q --no-cache-dir --upgrade pip wheel
TORCH_WHEEL="${TORCH_WHEEL:-$ROOT/wheels/torch-2.3.1+cu121-cp310-cp310-linux_x86_64.whl}"
download "https://mirrors.aliyun.com/pytorch-wheels/cu121/torch-2.3.1%2Bcu121-cp310-cp310-linux_x86_64.whl" "$TORCH_WHEEL"
"$PIP" install --no-cache-dir --index-url "$PIP_INDEX" "$TORCH_WHEEL"
"$PIP" install -q --no-cache-dir --index-url "$PIP_INDEX" \
  "numpy==1.26.4" "pandas==2.0.0" "scipy==1.11.4" "anndata==0.9.1" \
  "scanpy==1.9.3" "scikit-learn==1.2.2" "h5py==3.11.0" \
  "scikit-image==0.22.0" "statsmodels==0.14.1" "pot==0.9.3" \
  "squidpy==1.2.3" shapely "numba==0.59.1" "llvmlite==0.42.0" \
  pyyaml tqdm matplotlib seaborn pooch pillow platformdirs \
  torchdiffeq geomloss "torch_geometric==2.3.1"
"$PIP" install -q --no-cache-dir --no-deps --index-url "$PIP_INDEX" "SOMENDER==1.1"
"$PIP" install -q --no-cache-dir --index-url "$PIP_INDEX" "setuptools==80.9.0"
"$PIP" install -q --no-cache-dir torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
"$PIP" install -q --no-cache-dir --index-url "$PIP_INDEX" pykeops==2.2.3
"$PIP" install -q --no-cache-dir --no-deps --index-url "$PIP_INDEX" \
  pyvista vtk "opencv-python-headless==4.10.0.84" colorcet "kornia==0.6.12" scooby
"$PIP" install -q --no-cache-dir --force-reinstall --no-deps --index-url "$PIP_INDEX" "numpy==1.26.4"

say "verify common GPU stack"
CUDA_HOME="$ADA_CUDA_HOME" LD_LIBRARY_PATH="$ADA_CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" \
  "$COMMON/bin/python" - <<'PY'
import torch, torch_cluster, torch_sparse, pykeops
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
x = torch.randn(64, 3, device="cuda")
print("knn", torch_cluster.knn_graph(x, 6).shape)
print("sparse", torch_sparse.SparseTensor(row=torch.tensor([0], device="cuda"),
      col=torch.tensor([0], device="cuda"), value=torch.ones(1, device="cuda"),
      sparse_sizes=(1, 1)).nnz())
PY

# ---------------------------------------------------------------------------
# 2. Omni env: torch 2.6+cu124 for DeepSpatial/UniST
# ---------------------------------------------------------------------------
say "omni venv (torch 2.6.0+cu124)"
make_venv "$OMNI"
PIP2="$OMNI/bin/pip"
"$PIP2" install -q --no-cache-dir --upgrade pip wheel
TORCH_OMNI_WHEEL="${TORCH_OMNI_WHEEL:-$ROOT/wheels/torch-2.6.0+cu124-cp310-cp310-linux_x86_64.whl}"
TORCHVISION_OMNI_WHEEL="${TORCHVISION_OMNI_WHEEL:-$ROOT/wheels/torchvision-0.21.0+cu124-cp310-cp310-linux_x86_64.whl}"
download "https://mirrors.aliyun.com/pytorch-wheels/cu124/torch-2.6.0%2Bcu124-cp310-cp310-linux_x86_64.whl" "$TORCH_OMNI_WHEEL"
download "https://mirrors.aliyun.com/pytorch-wheels/cu124/torchvision-0.21.0%2Bcu124-cp310-cp310-linux_x86_64.whl" "$TORCHVISION_OMNI_WHEEL"
"$PIP2" install --no-cache-dir "$TORCH_OMNI_WHEEL" "$TORCHVISION_OMNI_WHEEL"
"$PIP2" install -q --no-cache-dir --index-url "$PIP_INDEX" \
  lightning tensorboardX omegaconf rich einops "scikit-learn>=1.4" \
  timm torchdiffeq pot pytorch-lightning torch_geometric scanpy anndata

# ---------------------------------------------------------------------------
# 3. env.sh
# ---------------------------------------------------------------------------
say "write $ROOT/env.sh"
cat > "$ROOT/env.sh" <<EOF
# Generated by scripts/setup_merfish5_env_ada.sh (RTX 4090 / sm_89 profile).
export MERFISH5_ROOT="$ROOT"
export MERFISH5_UPSTREAM="$UP"
export MERFISH5_PROFILE="ada"
export PY_COMMON="$COMMON/bin/python"
export PY_DEEPSPATIAL="$OMNI/bin/python"
export PY_UNIST="$OMNI/bin/python"
export UNIST_PY="$OMNI/bin/python"
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export CUDA_HOME="$ADA_CUDA_HOME"
export CUDA_PATH="$ADA_CUDA_HOME"
export PATH="$ADA_CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="$ADA_CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
EOF
chmod +x "$ROOT/env.sh"

say "done"
echo "source $ROOT/env.sh"
echo "next: bash scripts/fetch_upstream.sh"

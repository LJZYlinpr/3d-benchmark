# MERFISH 5-slice reproduction environment

This note covers the fixed-coordinate MERFISH 5-slice task (sections
`.006-.010`, 50 µm bins, scenarios M1a-M3a). It is the environment used for the
server 1 reproduction in `docs/REPRO_SERVER1.md`.

## Required layout

```text
$MERFISH5_ROOT/
  observed_50um.h5ad              # rebuilt from data/merfish_5slice/*.h5ad
  targets/<combo>/target.npz      # 7 combos, fixed-width strings
  canonical/
    bins_all.h5ad
    scenarios.json
    target_gt.npz
    READY
  predictions/<combo>/<method>.npz
  methods/<method>/               # outputs, status.json, run.log
$MERFISH5_UPSTREAM/
  SINTER3D/SINTER3D-master/
  isoST/
  SpatialZ/
  DeepSpatial/
```

The canonical builder is `src/canonical/merfish_5slice.py`; it expects
`$MERFISH5_ROOT/observed_50um.h5ad` and `$MERFISH5_ROOT/targets/`.

## Build the input and canonical data

```bash
# 1. committed shards -> single observed matrix
python scripts/build_observed_50um.py \
  --shards-dir data/merfish_5slice \
  --out "$MERFISH5_ROOT/observed_50um.h5ad"

# 2. observed -> targets/<combo>/target.npz (7 combos)
export MERFISH5_ROOT=/path/to/all_methods
python src/canonical/make_targets.py

# 3. observed + targets -> canonical/
python src/canonical/merfish_5slice.py
```

`make_targets.py` writes `obs_names` and `slice` as fixed-width Unicode
(`<U...`); it does not create pickled object arrays. This is required for
NumPy 1.22 environments.

## Python environments (server 1 example)

The original HPC runs used separate conda envs (`spateo`, `ntf`,
`SINTER3D_env`, `isoST`, `spatialz`, `deepspatial`, `unist`). A new machine
does not have them; these are the minimal tested combinations:

### Common env: ReHo, Ours-EncMTP, PASTE-OT-Linear, SpatialZ, SINTER3D

```text
base: VITA_spatialz_opt
  python 3.9, torch 1.13.0+cu117, anndata 0.9.1, ot 0.9.3, sklearn 1.2.2
add:
  torchdiffeq, geomloss, torch_geometric==2.3.1,
  torch-scatter, torch-cluster (CPU wheels are sufficient for SINTER3D)
```

PASTE on torch 1.13 needs `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`;
torch >=2.0 can use `expandable_segments:True`. The runner now selects this
automatically.

### DeepSpatial

DeepSpatial requires torch >=2.0, pytorch-lightning >=2.1 and `timm`; it cannot
run in the Python 3.9 common env. Tested combination:

```text
base: omicverse
  python 3.10, torch 2.14.0+cu130, pytorch-lightning 2.6.6,
  torch_geometric, scanpy, anndata
add:
  timm, torchdiffeq, pot
```

### isoST

isoST needs CUDA-enabled `torch_cluster` and `torch_sparse`. The CPU wheels fail
with `Not compiled with CUDA support` / `'SparseTensor' requires 'torch-sparse'`.
Use a dedicated env whose PyG extensions match the torch/CUDA build.

### NTF / Spateo / UniST

- NTF: needs `tinycudann`.
- Spateo: needs the `spateo` package and the upstream tree.
- UniST: needs the 1.1 GB upstream `UniST` + `SUICA_pro`; its per-scenario
  `_run_suica.py` files embed absolute paths and must be regenerated.

## Run the small comparison

```bash
export MERFISH5_ROOT=/path/to/all_methods
export MERFISH5_UPSTREAM=/path/to/upstream/methods
export PY_COMMON=/path/to/common/env/bin/python
export PY_DEEPSPATIAL=/path/to/deepspatial/env/bin/python

./scripts/run_merfish5_small.sh \
  ReHo Ours-EncMTP PASTE-OT-Linear SpatialZ SINTER3D DeepSpatial
```

This writes `predictions/<scenario>/<method>.npz` only; it does not run the
official metric suite. Use `metrics_v1_all` for paper metrics. The frozen
budgets used in the reported table are SINTER3D 3,000 steps, isoST 20/20/20
epochs, DeepSpatial 20 epochs, Ours-EncMTP 3,000 steps.

## Known caveats

- `_rb_common.py`, `common.py` and `_heart_common.py` require
  `MERFISH5_ROOT`; the upstream runners require `MERFISH5_UPSTREAM`.
- SINTER3D's prediction npz omits `method`, `scenario` and `genes`; validate
  the contract before feeding it to a downstream consumer.
- Status files are written into `$MERFISH5_ROOT/methods/<method>/` and must be
  cleared between runs.

## Fetch upstream method code

The upstream trees are not in Git. Fetch the pinned commits with:

```bash
export MERFISH5_UPSTREAM=/path/to/upstream/methods
./scripts/fetch_upstream.sh
```

This uses `codeload.github.com` tarballs, applies the recorded isoST/UniST/NTF
patches and copies the untracked new files. SINTER3D has no public commit
anchor and must be supplied separately as
`$MERFISH5_UPSTREAM/SINTER3D/SINTER3D-master`.

## Server 1 exact additions (tested 2026-10-23)

These are the extra steps needed on a machine that only has generic conda
base environments. They are recorded because the upstream methods are not
self-contained.

### Common torch 1.13 / PyG / KeOps env (isoST, NTF, Spateo, core methods)

```bash
python -m venv --system-site-packages "$ROOT/venv"
"$ROOT/venv/bin/pip" install "torch==1.13.1+cu117" \
  --index-url https://download.pytorch.org/whl/cu117
"$ROOT/venv/bin/pip" install torchdiffeq geomloss "torch_geometric==2.3.1"
"$ROOT/venv/bin/pip" install nvidia-cuda-runtime-cu11==11.7.99 \
  nvidia-cuda-nvrtc-cu11==11.8.89 pykeops==2.2.3
# CUDA PyG wheels (cp39, torch 1.13) from data.pyg.org:
#   torch_scatter-2.1.1+pt113cu117-cp39-cp39-linux_x86_64.whl
#   torch_sparse-0.6.17+pt113cu117-cp39-cp39-linux_x86_64.whl
#   torch_cluster-1.6.1+pt113cu117-cp39-cp39-linux_x86_64.whl
# On clusters where data.pyg.org is blocked, download on a machine with access
# and copy the wheels over.
```

For KeOps `backend="online"` (isoST), a CUDA toolkit is required:

```bash
conda create -y -p "$ROOT/cuda117" -c nvidia -c conda-forge \
  cuda-nvcc=11.7 cuda-cudart-dev=11.7 cuda-nvrtc-dev=11.7
# Build a compat tree with libcudart.so.11.0, libnvrtc.so and
# libnvrtc-builtins.so.11.2 (the pip nvrtc wheel ships 11.8 builtins; link it).
export CUDA_HOME="$ROOT/cuda117_compat"
export CUDA_PATH="$CUDA_HOME"
export LD_LIBRARY_PATH="$NVRTCDIR:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_HOME/lib64:$LIBRARY_PATH"
export PATH="$ROOT/cuda117/bin:$PATH"
```

Keep `numpy==1.22.2` in this env. Installing `opencv-python-headless` can pull
NumPy 2.x and break pandas/anndata; reinstall `numpy==1.22.2` afterwards.

### NTF (pure PyTorch fallback)

`tinycudann` is optional: `NTF/NTF/models.py` falls back to a pure PyTorch
hash grid. The fallback needs `NTF_torch_fallback.patch` (applied by
`fetch_upstream.sh`) for integer casts and the missing `n_output_dims` /
`box_offsets`. No CUDA extension is needed.

### Spateo

The upstream `spateo` package imports segmentation/alignment modules
(`pyvista`, `cv2`, `csbdeep`, `gpytorch`, ...) that DeepInterpolation does not
need. Install the minimal deps and use the shim:

```bash
"$ROOT/venv/bin/pip" install --no-deps pyvista vtk opencv-python-headless \
  colorcet kornia scooby
eval "$(./scripts/make_spateo_shim.sh)"
```

The shim exposes `spateo` and `spateo.tdr.interpolations` paths without
executing their heavy `__init__.py` files. For M1a the Spateo full-volume
script uses `--holdout-iz 13` (S2 z layer); the wrapper computes this index from
`canonical/sections.json` and `grid_50um.npz`.

### UniST

Use a torch >=2.6 environment for the current SUICA code (`torch.serialization.
add_safe_globals`), for example a venv over `omicverse`:

```bash
"$ROOT/venv_unist/bin/pip" install lightning tensorboardX omegaconf rich \
  einops "scikit-learn>=1.4"
export UNIST_PY="$ROOT/venv_unist/bin/python"
```

The UniST source tarball is small; the 1.1 GB local tree includes logs and
checkpoint artifacts that the from-scratch SUICA pipeline does not require.

### DeepSpatial

Already covered above: use an `omicverse` venv with `timm`, `torchdiffeq`,
`pot`; set `PY_DEEPSPATIAL`.

## Resource notes

- RTX 2080 Ti 11 GB: isoST peak 10.8 GB, DeepSpatial peak 9.9 GB. Run those
  alone.
- NTF/Spateo/UniST can run concurrently in the server 1 run (~5.3 GB total),
  but they contend for the single GPU; the timings above are concurrent.
- SINTER3D's prediction contract omits `method`, `scenario`, `genes`; validate
  before use.

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

# 2. observed + targets -> canonical/
export MERFISH5_ROOT=/path/to/all_methods
python src/canonical/merfish_5slice.py
```

`targets/*.npz` must store `obs_names` and `slice` as fixed-width Unicode
(`<U...`), not pickled object arrays. The server 1 run converted them because
NumPy 1.22 cannot unpickle NumPy-2 object arrays.

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

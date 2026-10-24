# MERFISH 5-slice quick start (fresh clone)

This is the tested path from a fresh clone to the ten-method M1a predictions.
It assumes a Linux x86_64 GPU host. All upstream code is either fetched from
GitHub at pinned commits or vendored under `vendor/`; no HPC paths are needed.

## 1. Clone and layout

```bash
git clone --depth 1 https://github.com/LJZYlinpr/3d-benchmark.git
cd 3d-benchmark
# or download the code-only tarball:
# https://github.com/LJZYlinpr/3d-benchmark/releases/download/v0.3.0/3d_benchmark_code_v1.tar.gz
export MERFISH5_ROOT="$PWD/work"          # experiment root
export MERFISH5_UPSTREAM="$PWD/upstream/methods"
mkdir -p "$MERFISH5_ROOT" "$MERFISH5_UPSTREAM"
```

## 2. Environments

```bash
bash scripts/setup_merfish5_env.sh
source "$MERFISH5_ROOT/env.sh"
```

**RTX 4090 / Ada (sm_89) or a host without conda/ensurepip:** use the Ada
profile instead. It uses torch 2.3.1+cu121, CUDA 12.1 PyG wheels and the system
`/usr/local/cuda-12.1` toolkit, and creates `venv_ada` + `venv_omni`:

```bash
MERFISH5_PROFILE=ada bash scripts/setup_merfish5_env_ada.sh
source "$MERFISH5_ROOT/env.sh"
```

This creates:

```text
$MERFISH5_ROOT/venv        # torch 1.13.1 + CUDA PyG + KeOps (isoST/NTF/Spateo/core)
$MERFISH5_ROOT/venv_omni   # torch 2.6 + lightning/timm (DeepSpatial/UniST)
$MERFISH5_ROOT/cuda117     # local nvcc 11.7 for KeOps
```

Notes:

- Python 3.9/3.10 is required for the common env; the script auto-selects
  `python3.10`/`python3.9` when available.
- The CUDA PyG wheels are platform-specific. If `data.pyg.org` is blocked, the
  script prints the exact three wheel names; download them elsewhere and
  install with `pip install --no-deps`.
- `conda` is needed only for the local CUDA 11.7 toolkit used by isoST. The
  other nine methods do not need it.

## 3. Fetch upstream methods

```bash
bash scripts/fetch_upstream.sh
```

Fetches NTF, isoST, UniST, Spateo, SpatialZ and DeepSpatial at the pinned
commits in `docs/upstream/method_commits.tsv`, applies the recorded patches and
copies the untracked new source. SINTER3D is already vendored under
`vendor/SINTER3D/SINTER3D-master`; no fetch is needed for it.

## 4. Fetch the data package, then build input and canonical data

The five shards + grid are a separate release asset (`data-v1`), not Git blobs:

```bash
bash scripts/fetch_data_merfish5.sh
```

```bash
python scripts/build_observed_50um.py \
  --shards-dir data/merfish_5slice \
  --out "$MERFISH5_ROOT/observed_50um.h5ad"

python src/canonical/make_targets.py
python src/canonical/merfish_5slice.py

cp data/merfish_5slice/grid_50um.npz "$MERFISH5_ROOT/grid_50um.npz"
```

Expected: `canonical/{bins_all.h5ad,scenarios.json,target_gt.npz,READY}`.
The released shard rebuild is value-identical to the HPC `observed_50um.h5ad`;
the canonical hashes are recorded in `docs/REPRO_SERVER1.md`.

## 5. Run the ten methods on M1a

```bash
bash scripts/run_merfish5_small.sh \
  ReHo Ours-EncMTP PASTE-OT-Linear SpatialZ SINTER3D DeepSpatial \
  isoST NTF Spateo UniST
```

The wrapper reads `MERFISH5_ROOT`, `MERFISH5_UPSTREAM`, `PY_COMMON`,
`PY_DEEPSPATIAL` and `PY_UNIST` from `env.sh` and writes:

```text
$MERFISH5_ROOT/predictions/M1a/<method>.npz
```

Use `SCENARIO=M1b` etc. for another held-out combo. The wrapper computes the
Spateo holdout z index from `sections.json`/`grid_50um.npz`.

Expected wall time on one RTX 2080 Ti: about 2 hours for all ten (UniST ~52
min, DeepSpatial ~20 min, Spateo ~12 min, isoST ~9 min, SINTER3D ~9 min).

## 6. Check the outputs

```bash
python scripts/validate_predictions.py \
  --canonical "$MERFISH5_ROOT/canonical" \
  --predictions "$MERFISH5_ROOT/predictions/M1a"
```

This checks `pred`/`row_index` shape, finiteness and M1a row alignment.

Core expression metrics (module-mean gene PCC, pooled PCC, RMSE, MAE):

```bash
python scripts/evaluate_merfish5_predictions.py \
  --canonical "$MERFISH5_ROOT/canonical" \
  --predictions-root "$MERFISH5_ROOT/predictions" \
  --modules manifests/merfish_5slice_modules.json \
  --scenarios M1a \
  --out "$MERFISH5_ROOT/metrics_m1a.csv"
```

The full paper metric suite is separate; this script covers the expression
metrics used in the reproduction reports.

## Known caveats

- NTF uses the pure-PyTorch hash-grid fallback and the wrapper runs 3,000
  iterations. This is a smoke test, not the full HPC budget.
- UniST is a from-scratch 300/400 SUICA run; the HPC table value came from a
  different run/configuration.
- Spateo must use `Spateo/holdout_pred_m1a.npz`; the full-volume h5ad contains
  measured values at observed rows (including the held-out layer) and will
  leak if used directly. The wrapper already does the right thing.
- isoST peaks at ~10.8 GB; run it alone on an 11 GB card.
- SINTER3D's prediction npz omits `method/scenario/genes`; `validate_predictions.py`
  only requires `pred`/`row_index`.

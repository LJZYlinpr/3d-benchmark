# 3D Spatial Transcriptomics Reconstruction Benchmark

**Version: **v0.1.1-draft**  
**Repository:** https://github.com/LJZYlinpr/3d-benchmark  
**Status:** internal first-version package; scientific results still need the baseline fixes listed in `docs/BASELINE_AUDIT.md`.

## What this project is

A benchmark for reconstructing spatial transcriptomics expression on held-out
sections under the **fixed-coordinate** protocol:

- visible sections are used for fitting;
- target coordinates of every held-out cell/bin are available;
- the target expression matrix is **never** used for fitting, calibration,
  thresholding, module definition, or model selection;
- predictions are written only for target rows; no dense 3D volume is required.

This is intentionally different from:

- native coordinate generation;
- arbitrary-direction virtual-section generation;
- dense 3D volume reconstruction.

Those tasks are separate and are not mixed into this benchmark's fixed-coordinate
scores.

## Scope and resolution policy

The v1 grid policy uses:

```text
10, 20, 40, 100 micrometres
```

The older 50 µm grids are retained only for reproducing the historical
5-slice runs and are **not** the v1 resolution policy.

This package does **not** use 50-PC features. The new metric suite is computed
on the full gene panel of each dataset. Any historical 50-PC analysis is outside
this package.

## Methods and baselines

Current method count depends on the definition:

| Category | Count | Methods |
|---|---:|---|
| Paper common-task methods | **8** | SINTER3D, UniST, isoST, SpatialZ, NTF, DeepSpatial, PASTE-OT-Linear, ReHo |
| Self-built baselines | **3** | Ours-EncMTP, PASTE-OT-Linear, ReHo |
| Supplemental interpolation reference | **1** | Spateo |
| Current runnable implementations | **10** | 8 common + Ours-EncMTP + Spateo |
| Blocked | **2** | Spa3D, ASIGN |

PASTE-OT-Linear and ReHo are both common-task methods and self-built baselines.
Ours-EncMTP is an additional self-built baseline. Spateo is supplemental and is
not part of the 8-method common table.

## Datasets and current experiments

| Dataset | Scope | Current status |
|---|---|---|
| MERFISH ABCA-3 | 5 slices `.006–.010`; 50 µm legacy; 7 missing combinations | 10 methods × 7 combos predictions; new metrics complete |
| Heart | 10 slices, sections 21–30; 238 genes | 10 methods × 26 scenarios predictions complete; metrics not yet unified |
| So3D | 46 datasets ≥5 slices; 901,284 spots; 2k/3k/4k genes | 8 methods × 46 predictions complete; new metrics complete |

The paper draft's 54-slice legacy formulation is **not** the same as the
current 5-slice workspace. A 54-slice manifest must be frozen before claiming a
54-slice v1 main result.

## Repository layout

```text
3d_benchmark/
  README.md
  configs/                  protocol, resolutions, dataset configs
  data/merfish_5slice/      data manifest + download target (data is release data-v1)
  docs/                     data, methods, metrics, environment, reproduction
  docs/methods/             per-method/per-baseline reproduction notes
  envs/                     conda env exports (from-history) + merfish5 pip freezes
  vendor/SINTER3D/          vendored SINTER3D source (no public commit anchor)
  manifests/                dataset/scenario/module manifests (no raw data)
  results/                  small summary tables, no predictions/volumes
  src/
    benchmark/grid.py       uniform 10/20/40/100 grid builder
    canonical/              canonical dataset builders
    metrics/merfish/        new 10-metric suite (full genes, no 50-PC)
    metrics/so3d/           external-dataset metric adaptation
    methods/merfish/        method adapters for the 5-slice common task
    methods/heart/          method adapters for heart 10-slice
    methods/so3d/           method adapters for So3D 46
  scripts/                  thin reproduction wrappers
  tests/                    smoke tests
```

Raw `.h5ad`, predictions, and volumes are intentionally not committed. The
MERFISH 5-slice input is a **separate release asset** (`data-v1`:
`merfish_5slice_data_v1.tar.gz`, five lossless per-slice 50 µm shards + grid).
Download it with `scripts/fetch_data_merfish5.sh`; the Git tree contains only
the manifest and hashes. Use `git clone --depth 1` for a small code-only
checkout. Rebuild the single input with `scripts/build_observed_50um.py`.

## Downloads

- Code-only package: `v0.3.0` / `3d_benchmark_code_v1.tar.gz`
  <https://github.com/LJZYlinpr/3d-benchmark/releases/download/v0.3.0/3d_benchmark_code_v1.tar.gz>
- Data package: `data-v1` / `merfish_5slice_data_v1.tar.gz`
  <https://github.com/LJZYlinpr/3d-benchmark/releases/download/data-v1/merfish_5slice_data_v1.tar.gz>

The Git tree is code-only; use a shallow clone or the code tarball.  The data
package is downloaded and SHA-256 verified by
`scripts/fetch_data_merfish5.sh`.

## Quick start

For the tested fresh-clone path see
[`docs/QUICKSTART_MERFISH5.md`](docs/QUICKSTART_MERFISH5.md); the environment
lessons are in [`docs/REPRODUCTION_LESSONS.md`](docs/REPRODUCTION_LESSONS.md).

### 1. Create an environment

Choose the environment for the method family; see `docs/ENVIRONMENT.md`.

```bash
conda env create -f envs/spateo.yml     # example; from-history export
conda activate spateo
```

The `envs/*.yml` files are lightweight `--from-history` exports, not complete
locks. For exact reproduction, freeze the environment on the machine that
produced the result.

### 2. Build uniform grids

Input is an `.npz` containing `xyz` (or `x`, `y`, `z`) coordinates in
millimetres:

```bash
python src/benchmark/grid.py \
  --cells cells.npz \
  --out-dir grids \
  --resolutions 10 20 40 100
```

Outputs match the historical runner schema:

```text
grid_{10,20,40,100}um.npz
  x_centers, y_centers, z_centers
  obs_mask, query_mask, inside, query_near5
  x0, y0, z0, bs, nx, ny, nz
```

`obs_mask` is observed voxels; `query_mask = inside & ~obs_mask`;
`query_near5` is the subset within 5 xy bins of an observed voxel.

### 3. Follow a reproduction track

- `docs/REPRODUCE.md` — MERFISH 5-slice, heart and So3D commands.
- `docs/methods/index.md` — per-method reproduction documents.
- `docs/METHOD_CONSTRUCTION.md` — how adapters were constructed from xpy, including provenance/patch caveats.
- `docs/METRICS.md` — new metric definitions.
- `docs/METHODS.md` — method code and known deviations.
- `docs/DATA.md` — input formats and data locations.
- `docs/BASELINE_AUDIT.md` — P0/P1 issues found in the three self-built baselines.

## New metric suite

The new suite is gene- and module-based, not 50-PC based:

- median/mean per-gene PCC, RMSE, MAE, cosine, Spearman, variance ratio;
- cell-type marker module PCC, variance ratio, module RMSE;
- masked 2D MS-SSIM;
- |Δ Moran's I| and |Δ Geary's C|;
- hotspot precision/recall/AP;
- boundary precision/recall/F1;
- support/N/A diagnostics and coverage.

No single composite score is defined. Gene-level PCC on very sparse So3D data
is not a reliable main ranking; use module and spatial metrics with explicit
support/N/A columns.

## Current headline results

### MERFISH 5-slice

10 methods, 7 held-out combinations. See `results/merfish_5slice_final_all_methods.md`.
The strongest module-level result is Ours-EncMTP; ReHo remains a strong simple
lower bound. This is not the final 54-slice paper result.

### So3D

46 datasets × 8 methods. See `results/so3d_method_summary.md` and
`manifests/SO3D_BY_DATASET.md`.

Important caveats:

- So3D target expression is extremely sparse (median zero fraction ≈0.978);
- gene PCC is near zero for every method and should not be the main criterion;
- two datasets have <100 target spots, so all spatial metrics are N/A;
- boundary/hotspot support differs by method; N/A is not zero;
- only 7/46 datasets have clearly physical z spacing.

## Scientific cautions before a public v1

1. **Fix the P0 baseline issues** in `docs/BASELINE_AUDIT.md`:
   - PASTE M2b bracket bug;
   - PASTE is PPA / 6k-anchor, not faithful PASTE;
   - Ours provenance and relative-offset unit bug;
   - ReHo is nearest-visible, not upstream preceding-slice.
2. Freeze 54-slice versus 5-slice scope. Do not claim both.
3. Run the 10-metric heart summary.
4. Add exact environment locks, per-scenario seeds, checkpoints and hashes.
5. Do not use a single overall score; report support/N/A and per-dataset tables.

## Third-party methods

Method implementations were adapted from the corresponding upstream projects
and the local benchmark adapters. See `docs/THIRD_PARTY.md` for attribution and
known deviations.

## License

No license has been selected for this draft. Do not redistribute until the
license and third-party terms are finalized.

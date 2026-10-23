# Server 1 reproduction, 2026-10-23

Isolated run on external server `1` (`aicb1`, 1 x RTX 2080 Ti 11 GB, 251 GB RAM).
Purpose: rebuild the MERFISH 5-slice input from the committed shards, run a
small multi-method comparison on one held-out scenario (M1a), and record every
reproducibility problem found.

Run root (temporary):

```text
/mnt/data1/linpeiran/tmp/3d_benchmark_repro_20261023/
```

## Fix status (2026-10-23)

Fixed in the same change set:

- `scripts/build_observed_50um.py` materialises `obs['bin_id']` from the
  shard index.
- `src/canonical/common.py` + `src/canonical/make_targets.py` now regenerate
  the seven targets from the rebuilt observed matrix with fixed-width strings;
  the generated arrays match the converted HPC targets exactly.
- Target-only 5-slice runners for ReHo, Ours-EncMTP, PASTE-OT-Linear,
  SpatialZ, SINTER3D, isoST and DeepSpatial are now under
  `src/methods/merfish/<method>/run_merfish5.py`, with `MERFISH5_ROOT` and
  `MERFISH5_UPSTREAM` instead of HPC paths.
- `.gitignore` no longer excludes the `src/canonical/` package; the canonical
  builders are now committed (the previous GitHub tree had none).
- PASTE selects `expandable_segments` only on torch >=2.0; torch 1.13 gets
  `max_split_size_mb:512`.
- A wrapper (`scripts/run_merfish5_small.sh`) and environment note
  (`docs/ENVIRONMENT_MERFISH5.md`) were added.

Still open:

- NTF (`tinycudann`), Spateo (`spateo` package + upstream), UniST (1.1 GB
  upstream + SUICA) and isoST (CUDA PyG extensions) need dedicated
  environments/upstream code.
- Exact per-method environment locks and upstream commits are not yet frozen.
- SINTER3D's prediction keys and the stale status metadata are not fixed.

## 1. Data rebuild

The repository's compact shards were copied to the server and rebuilt:

```bash
python scripts/build_observed_50um.py \
  --shards-dir data/merfish_5slice \
  --out $REPRO/observed_50um.h5ad
```

- 5 shards -> 122,669 bins x 1,122 genes, 124.3 MB output.
- The first attempt failed because the committed shards do not carry an
  explicit `obs['bin_id']` column. The merge script now materialises
  `bin_id` from the shard `obs` index; see `scripts/build_observed_50um.py`.
- Canonical build (`all_methods/code/01_build_canonical.py`) then produced
  `bins_all.h5ad` (177 MB) and `target_gt.npz` (322 MB, 298,876 rows).

Cross-check against the original HPC canonical:

| artifact | HPC SHA-256 | server 1 SHA-256 | match |
|---|---|---|---|
| `bins_all.h5ad` X (float32) | `db8a8271...8041f507` | `db8a8271...8041f507` | yes |
| `target_gt.npz` all arrays | `b006a566...d0d1ed28` | `b006a566...d0d1ed28` | yes |

The canonical matrix and target rows are value-identical to the HPC copy.
The full package pipeline was also rerun end-to-end from the committed shards
(`build_observed_50um.py` -> `make_targets.py` -> `merfish_5slice.py`) and
produced the same two hashes.

### Target `.npz` compatibility

The existing `targets/<combo>/target.npz` files were written by NumPy 2.x and
contain pickled object arrays. NumPy 1.22 fails with
`ModuleNotFoundError: numpy._core`. For this run the seven target files were
converted to fixed-width Unicode arrays:

```text
obs_names  object -> <U13
slice      object -> <U13
```

No numeric values changed. The converter should become part of
`make_targets.py` so targets are portable without a NumPy shim.

## 2. Methods run on M1a

One held-out scenario: hide section 2 (`C57BL6J-3.007`), visible 1/3/4/5.
Target: 25,281 bins x 1,122 genes.

| method | status | wall | GPU peak | module-mean gene PCC (server 1) | HPC table | diff | module corr |
|---|---|---:|---:|---:|---:|---:|---:|
| ReHo | DONE | 8.2 s | CPU | 0.414783 | 0.414783 | 0.000000 | 1.0000 |
| Ours-EncMTP | DONE | 110.1 s | 1,718 MB | 0.528220 | 0.532664 | -0.004444 | 0.9970 |
| PASTE-OT-Linear | DONE | 114.9 s | 2,504 MB | 0.442714 | 0.442688 | +0.000026 | 1.0000 |
| SpatialZ | DONE | 120.4 s | 1,712 MB | 0.497145 | 0.497124 | +0.000021 | 1.0000 |
| SINTER3D | DONE | 554.0 s | not recorded | 0.024909 | 0.012095 | +0.012813 | 0.8750 |
| DeepSpatial | DONE | 1,171.5 s | 9,908 MB | 0.347453 | 0.346893 | +0.000560 | 0.9993 |

`module-mean gene PCC` is the mean over the 18 cell-type marker modules of the
per-gene PCC median, i.e. the same definition as
`metrics_v1_all/metrics_by_method_combo.csv`. Predictions were checked for
shape, finite values, and `row_index` equality with the M1a target rows.

Deterministic methods (ReHo, PASTE, SpatialZ) reproduce exactly. DeepSpatial is
within 0.0006; Ours-EncMTP differs by 0.0044; SINTER3D differs by 0.0128 on a
near-zero signal. The ordering of the six methods matches the HPC table.

## 3. Methods not run

| method | blocker |
|---|---|
| NTF | requires `tinycudann`; not available in the generic env |
| Spateo | requires the `spateo` package and a 650 MB upstream tree; not installed |
| UniST | 1.1 GB upstream + `SUICA_pro`; per-scenario scripts embed absolute paths |
| isoST | requires CUDA-enabled `torch_cluster`; CPU wheel fails with `Not compiled with CUDA support`. After a CPU kNN patch it then requires `torch_sparse` (`'SparseTensor' requires 'torch-sparse'`) |

All four are main/common-task methods. The current repository cannot reproduce
them without upstream code and dedicated environments.

## 4. Issues found

### P0 - blocks reproduction on a new machine

1. **Hard-coded HPC paths in shared helpers.** `common.py`, `_heart_common.py`
   and `_rb_common.py` set `ROOT` to
   `/storage/.../missing_combos_50um/all_methods`; every runner fails elsewhere.
   Fixed for the new runners via `MERFISH5_ROOT`; the legacy NTF/Spateo
   scripts still use HPC paths.
2. **Hard-coded upstream paths in method runners.** `SINTER3D_DIR`, `ISOST_DIR`,
   `SPATIALZ_DIR`, `METHOD_DIR` point into
   `/storage/.../xpy/lab/baseline/methods/...`. The upstream trees are not in
   Git and are not pinned/downloadable by a script.
3. **The 5-slice runners for the eight newer methods are not in the package.**
   Fixed for ReHo, Ours-EncMTP, PASTE-OT-Linear, SpatialZ, SINTER3D, isoST and
   DeepSpatial (`src/methods/merfish/*/run_merfish5.py`). NTF/Spateo/UniST
   target-only adapters are still missing.
4. **Committed shards have no `obs['bin_id']` column.** `bin_id` is the shard
   index; the canonical builder expects a column. Fixed in the merge script,
   but the shards themselves should either add the column or document the index
   contract explicitly.
5. **Target `.npz` files are NumPy-2 pickles.** They cannot be loaded by NumPy
   1.22. Fixed by `src/canonical/common.py` + `make_targets.py`: regenerated
   targets use fixed-width Unicode and match the converted HPC arrays exactly.
   Legacy target files still need regeneration on older NumPy.
6. **The package's `.gitignore` excluded `src/canonical/`.** The pattern
   `canonical/` matched the source directory, so the GitHub tree had no
   canonical builders. Fixed by removing the pattern and committing
   `src/canonical/`.
7. **No per-method environment definition.** Server 1 had no matching conda
   envs; this run used:
   - `VITA_spatialz_opt` (torch 1.13.0+cu117, anndata 0.9.1, ot 0.9.3) plus a
     venv with `torchdiffeq`, `geomloss`, `torch_geometric 2.3.1`,
     `torch_scatter`, `torch_cluster`, for ReHo/Ours/PASTE/SpatialZ/SINTER3D.
   - `omicverse` (torch 2.14.0+cu130, pytorch-lightning 2.6.6) plus a venv with
     `timm`, `torchdiffeq`, `pot`, for DeepSpatial.
   The repo only has `--from-history` environment exports.

### P1 - method/contract issues

8. **PASTE sets an allocator option that torch 1.13 does not understand.**
   `run_paste_heart.py:160` sets
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; torch 1.13 raises
   `RuntimeError: Unrecognized CachingAllocator option`. Fixed in
   `src/methods/merfish/PASTE-OT-Linear/run_merfish5.py` by selecting the
   option from the torch major version; the legacy `run_paste_heart.py` keeps
   the old behaviour.
9. **isoST needs CUDA PyG extensions.** `torch_cluster` was CPU-only; after
   moving kNN to CPU it failed on `torch_sparse` in `GraphUNet`. A dedicated
   env with CUDA `torch-scatter/sparse/cluster` matching the torch build is
   required.
10. **SINTER3D prediction contract is incomplete.** Its npz contains
   `pred,row_index,bin_id,section,x,y` but omits `method,scenario,genes` that
   the other runners write. It also left `gpu_peak_mb` unset and kept a stale
   `slurm_job: 9383994` from the HPC run.
11. **Status files are not reset between runs.** `SINTER3D` inherited HPC
    metadata; `Ours-EncMTP` status elapsed (71.5 s) disagrees with the actual
    run (110.1 s). Status must be reinitialised on every new run.
12. **Default budgets differ from the reported configuration.** SINTER3D
    defaults to 6,000 steps (reported 3,000); isoST defaults to 60/60/60
    (reported 20/20/20); DeepSpatial defaults to 120 epochs (reported 20).
    The reproduction script must pass the frozen budgets explicitly.
13. **Metric definitions differ between quick checks and the official table.**
    Raw RMSE on the server is not comparable to the table's standardized RMSE.
    Use `metrics_v1_all` for paper numbers; the module-mean gene PCC above is
    the comparable metric.

## 5. Recommended next steps

1. Done for seven runners. Remaining: NTF/Spateo/UniST target-only runners.
2. Done: `scripts/run_merfish5_small.sh` and `docs/ENVIRONMENT_MERFISH5.md`.
3. Done: `src/canonical/common.py` + `make_targets.py` use fixed-width strings;
   `bin_id` is materialised by `build_observed_50um.py`.
4. Package or fetch upstream method trees with pinned commits; do not rely on
   `/storage/.../xpy` paths.
5. Add a prediction-contract validator (keys, shapes, row order, finiteness)
   and run it for every method.
6. Done: the PASTE runner selects the allocator option by torch major version.
7. Record per-run GPU peak, elapsed, config, seed, and library versions in a
   uniform `status.json` that is reset at start.

## 6. Reproduce this run

The exact commands and logs are on server 1 under the run root above:
`logs/`, `missing_combos_50um/all_methods/methods/*/run.log`,
`predictions/M1a/`, and `predictions/M1a/server1_vs_hpc_M1a.csv`.

The run root is temporary; the data pipeline is reproducible from the
committed shards: `build_observed_50um.py` -> `make_targets.py` ->
`merfish_5slice.py`. The regenerated targets match the converted HPC targets
exactly for all seven combos.

# Reproduction lessons (server 1, 2026-10-23)

These are the problems found while reproducing the MERFISH 5-slice M1a
comparison from a Git-only checkout on external server `1`. Every item either
points to a fix already in this repository or a remaining limitation.

## Code / packaging

| symptom | root cause | status |
|---|---|---|
| GitHub package had no canonical builders | `.gitignore` pattern `canonical/` also matched `src/canonical/` | fixed: pattern removed; `src/canonical/` committed |
| canonical builder failed on `obs['bin_id']` | compact shards stored the bin id only as the obs index | fixed: `build_observed_50um.py` materialises `bin_id` |
| NumPy 1.22 could not load targets | target npz stored pickled object strings | fixed: `src/canonical/make_targets.py` uses fixed-width Unicode; `make_targets.py` accepts CSR observed X |
| every runner failed on a new host | `common.py`/`_heart_common.py`/`_rb_common.py` hard-coded `/storage/...` | fixed: `MERFISH5_ROOT` |
| upstream imports failed | `SINTER3D_DIR`/`ISOST_DIR`/`SPATIALZ_DIR`/`METHOD_DIR` hard-coded `/storage/.../xpy` | fixed: `MERFISH5_UPSTREAM`; SINTER3D falls back to `vendor/` |
| 8 newer methods had no 5-slice runners | only heart adapters were in the package | fixed: `src/methods/merfish/*/run_merfish5.py` |
| upstream code not reproducible | no fetch script, patches/new files missing | fixed: minimal code vendored under `vendor/methods/`; `fetch_upstream.sh` optional |
| SINTER3D had no public commit | `no_vcs` in `method_commits.tsv` | fixed: 30 KB source vendored under `vendor/methods/SINTER3D` (no LICENSE in upstream tree; see NOTICE) |

## Runtime / environment

| symptom | root cause | status |
|---|---|---|
| PASTE crashed at startup | script set `expandable_segments:True`, unsupported by torch 1.13 | fixed: version-aware allocator choice in `run_merfish5.py` |
| isoST `Not compiled with CUDA support` | CPU-only `torch_cluster`; then `torch_sparse` missing | fixed: CUDA PyG wheels for the documented profiles; `vendor/shims/torch_cluster` CPU fallback |
| isoST `generic_logsumexp is not defined` | KeOps `backend="online"` needs pykeops/CUDA toolkit | fixed: local `cuda-nvcc 11.7` + pykeops 2.2.3 + compat `CUDA_HOME`; peak 10.8 GB |
| NTF crashed in fallback | `HashEmbedder` got float args, no `n_output_dims`, `box_offsets` registered after `return` | fixed: `NTF_torch_fallback.patch` |
| NTF held-out assertion failed | full-volume script assumed input observed = all grid-observed voxels | fixed: use input-observed voxels as observed; target voxels are sampled |
| Spateo produced 0.999 (leakage) | full-volume output fills observed rows with measured GT, including held-out layer | fixed: runner saves `holdout_pred_m1a.npz`; `points_to_target.py` consumes it |
| Spateo imports pulled segmentation stacks | `spateo/__init__` imports pyvista/cv2/csbdeep/gpytorch | workaround: `scripts/make_spateo_shim.sh`; no need to import segmentation |
| UniST `add_safe_globals` missing | current SUICA code expects torch >=2.6 | fixed: `venv_omni` on torch 2.6; `setup_merfish5_env.sh` |
| UniST `tensorboardX` missing | Lightning TensorBoardLogger dependency | fixed: setup installs `tensorboardX` |
| Spateo target coords off by 1000 | target rows in µm, volumes in mm | handled: `volume_to_target.py`/`points_to_target.py` auto-scale and print the median NN distance |
| isoST `ModuleNotFoundError: No module named 'utils'` | the upstream repo lacks the project-local `utils/` training directory | fixed: added `utils/{inference,ploting,postprocess,train_ode}.py` under `new_files/isoST`; `fetch_upstream.sh` copies them |
| upstream tarball extraction failed with `Unexpected EOF` | a partial cached `*.tar.gz` was treated as complete | fixed: `fetch_upstream.sh` validates with `tar -tzf` and re-downloads |
| Ada resolver conflict | `squidpy==1.3.0` requires `scikit-image<=0.20`; setup pinned 0.22 | fixed: both profiles pin `scikit-image==0.20.0` |
| SpatialZ fresh env failed | squidpy 1.3 pulls spatialdata/dask; squidpy 1.2.3 works | fixed: Ada setup installs 1.2.3 with `--no-deps` after the resolver batch |
| SpatialZ `No module named 'MENDER'` | `SOMENDER` (MENDER) and its `squidpy/shapely/numba` deps were not installed | fixed: setup installs `SOMENDER==1.1`, `shapely`, numba; Ada uses `squidpy==1.3.0` (1.2.3 requires numba<0.56) |
| SINTER3D `CUDA out of memory` on GPU3 | a co-tenant already used ~19 GB on the 24 GB card | operational: run SINTER3D on a free GPU; it fits alone (GPU0 run: 4.7 min) |
| fresh env.sh failed under `set -u` | `LD_LIBRARY_PATH`/`LIBRARY_PATH` referenced before assignment | fixed: use `${LD_LIBRARY_PATH:-}` / `${LIBRARY_PATH:-}` |
| PyG import failed on `torch_cluster.random_walk` | PyG node2vec/dropout reads an op missing from some torch_cluster builds | fixed: `scripts/patch_pyg_torch_cluster.py`, called by setup and the isoST runner |
| PASTE OT step 425 s/combo on Ada | `ot` Sinkhorn path did not use the GPU efficiently with the current stack | known performance issue; budget ~7 min/combo in the 4-GPU matrix |

## Remaining limitations

- **NTF** was run for 3,000 iterations with the pure-PyTorch fallback; the HPC
  table used a different budget. Freeze the budget/seed for a like-for-like
  number.
- **UniST** was run from scratch (300/400) in a torch 2.6 environment; the HPC
  0.339 reference came from a different run/config. Freeze that run ID/config.
- **SINTER3D** source is vendored but has no explicit upstream license; confirm
  redistribution terms before publishing outside the project.
- **isoST** needs a CUDA 11.7 toolkit and an 11 GB GPU; a fresh host without
  conda can run the other nine methods but not isoST.
- The official metric suite (`metrics_v1_all`) is not run by
  `run_merfish5_small.sh`; the quick-start only validates the prediction
  contract. Use the metric suite for paper numbers.

## Rule for future additions

Before calling a method reproducible, add:

1. a pinned upstream source (commit or vendored code),
2. a target-only runner with `MERFISH5_ROOT`/`MERFISH5_UPSTREAM`,
3. an exact environment recipe or lock file,
4. a `validate_predictions.py`-clean output,
5. a one-line entry in `docs/upstream/method_commits.tsv`.

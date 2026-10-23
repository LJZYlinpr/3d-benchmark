# SINTER3D — So3D 46-dataset held-out-middle-slice baseline (runner B)

Runner: So3D runner B, lane job `9383992` / node `gvna03`
(`srun --jobid=9383992 --overlap --nodelist=gvna03 --ntasks=1 --cpus-per-task=4 --gres=gpu:1`).
One method process at a time; no `scancel`, no preemption of the lane owner.

Code: `run_sinter3d.py` (adapted from `Workspace/merfish_test/SINTER3D/run_sinter3d_50um.py`
and `Workspace/merfish_test/heart_10slice/methods/SINTER3D/run_sinter3d_heart.py`).
Shared helpers: `../_so3d_common.py`.

## Task

For each of the 46 canonical So3D datasets (>=5 serial slices), one held-out middle
slice is the target. Train on the visible slices only; predict the target spots at
their fixed coordinates. Target expression (`target.npz['X']`) is evaluation-only and
is never opened by this method (the loader only reads coords/obs_names/slice/z).

## Inputs / output

- Input: `canonical/<dataset>/visible.h5ad` — all genes of that dataset's panel
  (2,000 / 3,000 / 4,000), canonical X used **as-is** (already producer-normalised
  log1p-like values; no renormalisation, no filtering).
- Output: `predictions/<dataset>/SINTER3D.npz` — `pred` (n_target, n_genes) float32
  in canonical target order, plus echoed `coords` / `obs_names` / `slice` /
  `row_index` / `genes`; `predictions/<dataset>/SINTER3D.status.json`.
  No grid/volume is materialised.
- Status/metrics: `methods/SINTER3D/{status.json,run.log,compute_metrics.json}`,
  aggregate `methods/RUNNER_B_STATUS.json`.

## Official model and adaptations

Model, loss, optimiser and hyper-parameters are the official SINTER3D
`SINTER3D.model_spatial.Model` / `DeconvNet` (hidden [512,128], slice emb 16,
mid_channel 200, Adamax lr 1e-3, alpha_poisson 5, lambda_feature 4,
gamma_feature 3, 6 Fourier freqs, coord emb 64, `normalize=100`, 6000 steps,
seed 2024). Documented adaptations:

1. **No scRNA reference / no target labels.** The official `utils.preprocess`
   needs an scRNA reference for the deconvolution basis. Canonical visible spots
   carry no class labels usable here, and the basis must not come from target
   expression, so the basis is built from **visible bins only**: KMeans with
   k = min(8, max(2, n_visible//500)) fitted on at most 30k visible rows, basis
   rows = unit-normalised cluster-mean profiles. The predicted quantity is the
   decoder head (`X_pred`, `deconv=False`), which is coordinate-only.
2. **Coordinate frame.** Canonical coordinates are per-dataset source units that
   differ by orders of magnitude (x spans 20 to 20,000; z spans 0.3 to 100). For
   each dataset, visible-only scales map xy so that the median within-slice
   nearest-neighbour distance is 100 and z so that the median inter-slice gap is
   100 — i.e. the official `c2c_dist = normalize = 100` µm frame. Target
   coordinates use the same visible-derived transform (no target statistics).
   The native spacings, scales and bboxes are recorded per dataset in
   `compute_metrics.json`.
3. **Mini-batch SGD.** The official trainer is full-batch; with 2k-4k gene panels
   and up to 60k visible spots one full-batch step measured ~0.42 s, i.e. > 9 h
   for 46 datasets x 6000 steps. We use plain mini-batch SGD (batch <= 16384,
   epoch-wise reshuffled permutation through all visible rows, identical loss /
   optimiser / lr / 6000-step budget). Measured 35 ms/step on 4,000-gene panels
   and 29 ms/step on 3,000-gene panels; datasets with fewer than 16,384 visible
   spots are trained full-batch.
4. **Output space.** The decoder is fitted directly in canonical-X units
   (`obsm['count'] = X`, `library_size = row sums`), so `pred` is already in the
   evaluation space; no expm1 back-scale, only clipping at 0.

## Caveats

- In-sample RMSE (on up to 8,000 visible rows, visible data only) is recorded per
  dataset as a fit diagnostic; it is **not** a benchmark metric and no PCC/R2 is
  computed here (metrics are computed centrally from the predictions).
- SINTER3D's deconvolution branch is trained against a KMeans basis rather than a
  matched scRNA reference; this can only affect the latent shape, not the
  decoder's coordinate-only prediction path.
- Very small targets are legitimate: `Heart__Mouse__3Ddataset_1` has 52 target
  spots (of 74 slices), `Embryo__Drosophila__3Ddataset_3` 99, `Embryo_4` 199.
- Datasets with fewer than 100 visible spots / 10 genes would be reported
  `UNSUPPORTED` (none occurred in the 46).

## Measured run (final)

- Coverage: **46/46 datasets DONE**, 0 FAILED, 0 UNSUPPORTED.
- Wall time 117.1 min on the lane; per dataset min/median/max = 68.3 / 173.3 / 224.7 s.
- GPU (device-wide, shared card): peak 7,197 MiB, util up to 99 %; torch alloc peak
  per dataset 288 / 2,696 / 4,821 MiB (min/median/max). Peak RSS 5.1 GiB, 1.14 cores mean.
- Output: 46 `SINTER3D.npz` + 46 `.status.json`, 0.69 GB total.
- In-sample visible RMSE (diagnostic only) min/median/max = 0.195 / 0.377 / 0.844.
- Prediction mean vs visible mean: median prediction mean 0.084; tracks the visible
  mean per dataset within roughly +/-30 %.
- Constant-gene audit (`../RUNNER_B_DIAGNOSTICS.json`): every constant prediction is
  all-zero, and those genes have median visible nonzero fraction 0.1-1 % (i.e. no
  usable signal in the visible slice either). SINTER3D's output is otherwise a dense
  smooth field: exact-zero fraction of predictions is 0.10-0.47 while the visible X is
  79-99 % zeros, so low-expression genes get small positive values instead of zeros
  (feature-loss/decoder behaviour, not a target-derived adjustment).

# SpatialZ — So3D 46-dataset held-out-middle-slice baseline (runner B)

Runner: So3D runner B, lane job `9383992` / node `gvna03`
(`srun --jobid=9383992 --overlap --nodelist=gvna03 --ntasks=1 --cpus-per-task=4 --gres=gpu:1`).

Code: `run_spatialz.py` (adapted from `Workspace/merfish_test/SpatialZ/run_spatialz_50um.py`
and `Workspace/merfish_test/heart_10slice/methods/SpatialZ/run_heart.py`).
Method code: official `SpatialZ.Generate_spatialz`, unmodified.
Helpers: `../_so3d_common.py`.

## Task

One held-out middle slice per dataset. Everything is derived from the two nearest
**visible** slices; target expression is never read (only target geometry:
coordinates, obs_names, slice, z).

## Procedure per dataset

1. **Bracket.** Visible slice ids are ordered by their median z; the nearest
   visible slice below and above the target z are the pair `(lo, hi)`.
   `alpha = (z_hi - z_target) / (z_hi - z_lo)` using native z (ratio only, so
   units cancel). A target that is not bracketed on both sides would be reported
   `UNSUPPORTED` (did not occur: prep chose interior slices).
2. **Synthesis.** `Generate_spatialz(visible_lo, visible_hi, alpha, ...)` with
   `syn_mode='fast'`, `lr=40`, `nb_iter_max=1000`, `num_projections=200`,
   `k_sam=3`, `n_mag=1`, seed 42 — the config validated on the 50 µm MERFISH and
   heart 10-slice runs. XY is fed in a common **pseudo-mm frame**: visible-only
   scale such that the median within-slice nearest-neighbour distance is 0.05 mm
   (50 µm spot spacing); this keeps the 50 µm-tuned `lr` and the method's
   internal `epsilon=0.1` meaningful for every dataset (native units range from
   ±20 to ±20,000). z is used only for `alpha`.
3. **Cell class.** Canonical spots do not carry class labels for the virtual
   slice (and target composition must not be used), so both visible frames use a
   single class `"all"`. Cell-type transfer is therefore a no-op; expression
   synthesis samples same-class nearest neighbours (`fast` mode).
4. **Mapping to the target slice.** Each virtual point is assigned to its nearest
   target coordinate in xy; target points receive the mean over assigned virtual
   points; target points with no assigned virtual point receive the nearest
   virtual point. Only the target rows are written.

Output: `predictions/<dataset>/SpatialZ.npz` — `pred` (n_target, n_genes) float32
in canonical target order plus echoed `coords`/`obs_names`/`slice`/`row_index`/`genes`;
`predictions/<dataset>/SpatialZ.status.json`. No volume is materialised.
Metrics/logs: `methods/SpatialZ/{status.json,run.log,compute_metrics.json}`.

## Caveats

- SpatialZ is a two-slice interpolation; with only the immediate bracket used, a
  target at the exact middle has `alpha = 0.5` and receives an even mixture of the
  two neighbouring clouds.
- Per dataset the target coverage (fraction of target coordinates receiving at
  least one virtual point) and the point-to-target NN distance are recorded; the
  remainder is nearest-point filled (a deterministic passthrough, not fabricated
  expression).
- No model is trained; there is no in-sample metric. Predictions are clipped at 0
  and written in canonical-X units (input X used as-is, never renormalised).

## Measured run (final)

- Coverage: **46/46 datasets DONE**, 0 FAILED, 0 UNSUPPORTED.
- Wall time 3.9 min; per dataset min/median/max = 2.5 / 3.8 / 20.7 s.
- GPU (device-wide): peak 2,487 MiB, util up to 100 %; torch alloc stays in the tens
  of MB (only the coordinate optimisation is on GPU).
- Output: 46 `SpatialZ.npz` + 46 `.status.json`, 0.08 GB total.
- Mapping: target coverage (fraction of target coordinates receiving >=1 virtual
  point) min/median/max = 0.312 / 0.682 / 0.913; point->target NN distance median
  0.014-0.379 mm. Unfilled target points take the nearest virtual point (recorded).
- Prediction mean tracks the visible mean per dataset (median 0.072) and the exact-zero
  fraction tracks the visible X zero fraction (predictions 0.58-0.98 vs visible
  0.79-0.99), as expected for a neighbour-sampling interpolator.
- Constant-gene audit (`../RUNNER_B_DIAGNOSTICS.json`): constant predictions are
  all-zero and sit on genes with median visible nonzero fraction 0.01-0.3 %; with only
  k_sam=3 same-class neighbours per virtual point such genes are frequently not sampled.

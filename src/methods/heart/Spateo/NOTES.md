# Spateo — heart 10-slice, target-only

Runner: `run_spateo_heart.py` (adapted from `Workspace/merfish_test/Spateo/run_spateo_50um.py`).

## Protocol
- 26 scenarios from `canonical/scenarios.json`; per scenario fit **only** on bins of the
  visible sections, predict **only** the hidden target bins listed in `canonical/target_gt.npz`.
- Output: `predictions/<scenario>/Spateo.npz` — `pred` (n_target_bins, 238) float32 in
  target_gt order + `row_index`, `bin_id`, `section`, `x`, `y`.
- No target expression is read (the runner loads only target_gt metadata keys; `gt` is never opened).
- No dense volume, no PCC/PCA/50PC.

## Model / config (unchanged from the 50um reference unless noted)
- Spateo `DeepInterpolation` + `interpolation_nn` (leaky-ReLU MLP 3→512×4→238), `weighted_mse`,
  smoothing/stability regularisers off, no SIREN.
- float64 on one A100; Adam(betas 0.5/0.9, wd 2.5e-5, amsgrad); batch 8192; cosine 3e-3→1e-5.
- Coordinates: per-axis min-max normalised to [-1,1] over the union of visible+target extents
  (geometry only, no expression).
- Deviations: **40,000 iterations** per scenario (reference full-grid run used 60,000 on 122k bins;
  here the training set is 31-56k visible bins and 26 scenarios must fit one lane). Negative
  predictions clipped to 0.

## Result
26/26 scenarios DONE, coverage 1.0, all finite. 198.6 s/scenario, 86.3 min total.
Output 321.1 MB. GPU device-peak 3563 MiB (co-tenants included), torch alloc peak ≤ 462 MiB,
process RSS peak 2240 MiB, mean 1.0 CPU cores.

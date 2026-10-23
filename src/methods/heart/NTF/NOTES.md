# NTF — heart 10-slice, target-only

Runner: `run_ntf_heart.py` (adapted from `Workspace/merfish_test/NTF/run_ntf_50um.py`).

## Protocol
- Per scenario: fit NTF on visible-section bins only; sample the hidden target bins only.
- Output: `predictions/<scenario>/NTF.npz` — `pred` (n_target_bins, 238) float32 in target_gt
  order + `row_index`, `bin_id`, `section`, `x`, `y`.
- `gt` is never loaded; no volume output; no PCC/PCA/50PC.

## Model / config (unchanged from the 50um reference unless noted)
- NTF `NeuralTranscriptomicField` (GeneINR hash grid, 13 levels, log2_hashmap 20, width 64,
  depth 1, 4 PSF samples, softplus expression, no dropout, single precision).
- Param-group Adam: hash grid 1e-2, rest 1e-3, betas (0.9,0.99); MultiStepLR 0.5/0.7/0.9 gamma 0.5;
  12,000 iters, batch 8192, weight_expr 1.0; patience disabled.
- Tight bbox = visible+target extent (geometry only); isotropic 50 um PSF (resolution [0.05,0.05,0.05] mm).
- The expression path (`GeneINR`) is coordinate-only, so hidden sections need no slice embedding;
  slice ids are used during training only for the visible sections' variance/bias heads.
- No renormalisation: canonical X is fit directly.

## Result
26/26 scenarios DONE, coverage 1.0, all finite, 0 constant genes, 0 exact zeros (softplus).
112.9 s/scenario, 50.7 min total. Output 322.0 MB. GPU device-peak 3173 MiB (co-tenants included),
torch alloc peak ≤ 1.5 GiB class, RSS peak 1604 MiB, mean 1.55 CPU cores.

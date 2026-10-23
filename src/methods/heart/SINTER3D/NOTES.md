# SINTER3D — heart 10-slice, target-only

Runner: `run_sinter3d_heart.py` (adapted from `Workspace/merfish_test/SINTER3D/run_sinter3d_50um.py`).

## Protocol
- Per scenario: train the official `SINTER3D.model_spatial.Model` on visible-section bins only;
  decode only the hidden target bins.
- Output: `predictions/<scenario>/SINTER3D.npz` — `pred` (n_target_bins, 238) float32 in
  target_gt order + `row_index`, `bin_id`, `section`, `x`, `y`.
- `gt` is never loaded; no volume output; no PCC/PCA/50PC.

## Model / config
- Official model + loss (Poisson deconvolution + feature-reconstruction terms), Adamax lr 1e-3,
  hidden [238,512,128], slice-emb 16, mid_channel 200, normalize=100 um, coord-emb 64,
  num_freqs 6, alpha_poisson 5, lambda_feature 4, gamma_feature 3, patience disabled.
- Predictions come from the decoder path (`inference_latent(..., decode=True)`), which is
  coordinate-only; slice embeddings only affect the deconvolution loss, so hidden sections
  need no slice label.

## Documented adaptations (dataset-specific)
1. **X used as the count-like target** (`count` matrix, `library_size` = row sums) with
   `adata_st.X = X` and no `log1p(CP10k)` renormalisation, because canonical X is already
   normalised and the plan forbids renormalising. The decoder therefore outputs canonical-X
   units directly — no `expm1`/library-size back-transform, and predictions are clipped at 0.
2. **Basis without cell-type labels**: canonical bins carry no class composition (leakage guard),
   so the deconv basis is built from KMeans (k ≤ 8, k-means++ n_init 4) over the **visible bins
   of that scenario**, using unit-normalised cluster-mean expression profiles. Target expression
   never enters the basis.
3. **6,000 training steps** per scenario (reference full-grid runs used 10,000 steps on 122k bins;
   here 31-56k visible bins × 26 scenarios on one lane).

## Result
26/26 scenarios DONE, coverage 1.0, all finite, 0 constant genes. 187.6 s/scenario, 81.3 min total.
Output 288.1 MB. GPU device-peak 4267 MiB (co-tenants included), RSS peak 1999 MiB,
mean 1.0 CPU cores. In-sample visible RMSE 0.19-0.23 at 6k steps.

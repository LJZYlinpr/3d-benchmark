# MERFISH 5-slice metric analysis plan

Status: prepared 2026-10-22; not launched yet.  
Root: `/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/Workspace/merfish_test/missing_combos_50um`

## Data

- MERFISH ABCA-3, sections `.006-.010` (5 slices), 50 um bins.
- 7 held-out scenarios: M1a, M1b, M1c, M2a, M2b, M2c, M3a.
- Existing predictions: NTF and Spateo only (14 method x scenario pairs).
- Target GT: `targets/<combo>/target.npz` with `X`, `coords`, `obs_names`, `slice`.
- Predictions: `predictions/<combo>/<method>.npz` with the same keys.
- 18 cell-type marker modules from S1/S5: `modules/modules_celltype.json`.
- Existing tables: `metrics.csv`, `module_metrics.csv`, `FINAL_THREE_METRICS.csv`.

This 5-slice set is a **metric-protocol testbed**, not a 10-method ranking: only NTF and Spateo have held-out predictions.

## Debug GPU

- Lane: job `9383994` on `gvna02`, currently 0% GPU / ~1.8 GB used; Spateo heart full run is on lane 1, so no conflict.
- Use `srun --jobid=9383994 --overlap --nodelist=gvna02 --ntasks=1 --cpus-per-task=4`.
- Analysis is small; GPU is for tensorized module maps / SSIM / perturbation warps. No training.

## Frozen configuration (pre-registered)

- Data: 50 um bins; same target bin index for GT and prediction; no renormalization.
- Gene modules: 18 modules from `modules/modules_celltype.json`; z-score mean/std from S1/S5 only; all-positive weights (mean); record gene coverage and module size.
- Correlation: Pearson per gene across target bins; genes with zero variance are N/A; report median, p10 and fraction > 0.5.
- RMSE: on the same normalized expression scale; report raw RMSE plus per-gene standardized RMSE.
- Module map: `M_k = mean(z_g)` over module genes; 2D per target section.
- Hotspot threshold: per-module 90th percentile of visible module scores; same numeric t for GT and prediction; report threshold sensitivity at 80/90/95th percentiles. Also report threshold-free average precision (AP) and PR-AUC.
- Boundary: Gaussian smooth module map (sigma=1 bin); gradient magnitude; threshold at visible 90th percentile; remove components <2 bins; exclude outer tissue contour; tolerance delta = 0 and 1 bin (50/100 um).
- MS-SSIM: 2D per target section, fixed window (7x7), 3 scales, masked to tissue; no 3D SSIM.
- Moran's I: kNN spatial weights (k=6) within each target section; absolute difference to GT; report Geary's C as secondary.
- Variance ratio: module score `var(pred)/var(GT)`; report log2 ratio.
- No target expression in module definitions, z-score stats, thresholds, or calibration.
- No dense-volume metrics and no 3D SSIM (no 3D GT here).

## Core metrics

A. Expression: median per-gene PCC; RMSE.  
B. Module: module PCC; module MS-SSIM.  
C. Spatial statistics: variance ratio; |Delta Moran's I|.  
D. Structure: hotspot recall/precision; boundary recall/precision.

Mandatory diagnostics (reported separately, not folded into a score):
- mean ratio and zero fraction;
- per-gene PCC p10 and fraction >0.5;
- hotspot AP/PR-AUC (threshold-free);
- boundary F1 and object-level component count / centroid distance;
- support/coverage and no-positive cases;
- split-half reliability of GT module maps (noise ceiling);
- paired bootstrap CIs by slice/section;
- runtime, GPU/RAM.

## Phases

1. **Input audit**: align genes, target rows, coords, slices, masks; verify GT/pred no NaN and no target leakage; write `input_audit.csv`.
2. **Freeze config**: write `metric_config.yaml` before seeing method scores; hash it.
3. **Core metrics**: compute 10 core + diagnostics for 7 combos x 2 methods x 18 modules; write raw long table.
4. **D-group implementation**: store GT/pred hotspot and boundary masks, TP/FP/FN, P/R, AP; strict and delta=1 bin; no-positive table.
5. **Perturbation validation**: on held-out GT module maps, apply
   - Gaussian blur sigma=1/2/3 bins;
   - added hotspot (radius 2/3 bins);
   - added boundary band;
   - spatial shift 1/2 bins;
   - spatially correlated noise with matched global variance;
   - merge/split connected components.
   Report metric response curves; do not retrain.
6. **Report**: per-combo and per-method table; module heatmap; P-R tradeoff; gap response; example GT/pred/error maps; reliability ceiling. No single composite score.

## Output

```text
missing_combos_50um/metrics_v1/
  metric_config.yaml
  input_audit.csv
  metrics_core.csv
  metrics_module.csv
  metrics_spatial_pr.csv
  metrics_diagnostics.csv
  perturbation_curves.csv
  figures/
```

## Statistical caveat

The 7 scenarios reuse the same 5 slices; they are **not independent biological replicates**. Report per-combo and per-slice results; use slice-level paired bootstrap or permutation, and do not treat the 7 combinations as 7 independent samples. With only 5 slices, CIs will be wide; the 5-slice experiment validates the protocol and characterizes NTF vs Spateo, not a population-level method ranking.

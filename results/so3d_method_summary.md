# So3D 46 x 8 metric summary (metrics_v1-adapted protocol)

- Generated: 2026-10-23T01:37:32Z
- Datasets: 46, methods: 8, cells: 368
- Metric rows: 164800 (OK 141531, NA 23269, FAILED 0)
- Held-out slice: fixed per dataset (`meta.target_slice`), target X used for scoring only.
- Raster: adaptive 2D bin from target xy geometry; module z-scores/hotspot/boundary thresholds from visible slices only.

## Gene-level (target spots, all genes)

Macro-average over datasets (only finite cells; N/A not imputed).

| method | gene_pcc_median | gene_pcc_mean | gene_rmse | gene_mae | gene_cosine_median | gene_spearman_median | gene_vr_log2_median | gene_pcc_frac_gt_0.5 | gene_vr_within_2fold_frac |
|---|---|---|---|---|---|---|---|---|---|
| ReHo | -0.004 (46/46) | 0.011 (46/46) | 0.577 (46/46) | 0.121 (46/46) | 0.018 (46/46) | -0.003 (46/46) | -1.033 (46/46) | 0.002 (46/46) | 0.538 (46/46) |
| PASTE-OT-Linear | -0.003 (46/46) | 0.014 (46/46) | 0.454 (46/46) | 0.115 (46/46) | 0.034 (46/46) | 0.008 (46/46) | -2.543 (46/46) | 0.003 (46/46) | 0.074 (46/46) |
| NTF | 0.007 (46/46) | 0.027 (46/46) | 0.419 (46/46) | 0.103 (46/46) | 0.076 (46/46) | 0.021 (46/46) | -7.595 (46/46) | 0.005 (46/46) | 0.014 (46/46) |
| SINTER3D | -0.004 (46/46) | -0.000 (46/46) | 0.437 (46/46) | 0.132 (46/46) | 0.090 (46/46) | -0.002 (46/46) | -7.209 (46/46) | 0.000 (46/46) | 0.004 (46/46) |
| SpatialZ | -0.003 (46/46) | 0.016 (46/46) | 0.467 (46/46) | 0.116 (46/46) | 0.030 (46/46) | -0.002 (46/46) | -1.790 (46/46) | 0.003 (46/46) | 0.156 (46/46) |
| isoST | -0.004 (46/46) | 0.009 (46/46) | 0.576 (46/46) | 0.124 (46/46) | 0.019 (46/46) | 0.002 (46/46) | -0.561 (46/46) | 0.001 (46/46) | 0.586 (46/46) |
| DeepSpatial | -0.004 (46/46) | 0.011 (46/46) | 0.576 (46/46) | 0.120 (46/46) | 0.018 (46/46) | -0.003 (46/46) | -0.925 (46/46) | 0.002 (46/46) | 0.572 (46/46) |
| UniST | 0.027 (46/46) | 0.044 (46/46) | 0.433 (46/46) | 0.108 (46/46) | 0.153 (46/46) | 0.031 (46/46) | -24.300 (46/46) | 0.006 (46/46) | 0.001 (46/46) |

## Module-level (visible-celltype marker modules)

Macro-average over datasets (only finite cells; N/A not imputed).

| method | module_pcc | module_vr_log2 | module_rmse |
|---|---|---|---|
| ReHo | 0.192 (46/46) | 0.007 (46/46) | 0.457 (46/46) |
| PASTE-OT-Linear | 0.210 (46/46) | -1.726 (46/46) | 0.369 (46/46) |
| NTF | 0.257 (46/46) | -2.466 (46/46) | 0.363 (46/46) |
| SINTER3D | -0.002 (46/46) | -5.418 (46/46) | 0.429 (46/46) |
| SpatialZ | 0.256 (46/46) | -1.022 (46/46) | 0.376 (46/46) |
| isoST | 0.198 (46/46) | 0.028 (46/46) | 0.452 (46/46) |
| DeepSpatial | 0.200 (46/46) | 0.052 (46/46) | 0.453 (46/46) |
| UniST | 0.237 (46/46) | -5.694 (46/46) | 0.410 (46/46) |

## Spatial (2D target slice, module maps)

Macro-average over datasets (only finite cells; N/A not imputed).

| method | ms_ssim | abs_delta_moran_i | abs_delta_geary_c | hotspot_recall | hotspot_precision | hotspot_f1 | hotspot_AP | boundary_recall_d0 | boundary_precision_d0 | boundary_f1_d0 |
|---|---|---|---|---|---|---|---|---|---|---|
| ReHo | 0.119 (44/46) | 0.138 (46/46) | 0.138 (46/46) | 0.214 (44/46) | 0.209 (44/46) | 0.201 (44/46) | 0.193 (44/46) | 0.260 (43/46) | 0.245 (43/46) | 0.237 (43/46) |
| PASTE-OT-Linear | 0.111 (44/46) | 0.254 (46/46) | 0.254 (46/46) | 0.219 (44/46) | 0.249 (44/46) | 0.176 (44/46) | 0.207 (44/46) | 0.111 (42/46) | 0.282 (42/46) | 0.133 (42/46) |
| NTF | 0.158 (44/46) | 0.328 (46/46) | 0.327 (46/46) | 0.330 (42/46) | 0.320 (42/46) | 0.224 (42/46) | 0.235 (44/46) | 0.175 (43/46) | 0.366 (43/46) | 0.199 (43/46) |
| SINTER3D | 0.022 (44/46) | 0.323 (46/46) | 0.323 (46/46) | 0.560 (38/46) | 0.134 (38/46) | 0.146 (38/46) | 0.113 (44/46) | 0.152 (6/46) | 0.171 (6/46) | 0.086 (6/46) |
| SpatialZ | 0.165 (44/46) | 0.248 (46/46) | 0.247 (46/46) | 0.213 (44/46) | 0.259 (44/46) | 0.217 (44/46) | 0.228 (44/46) | 0.170 (43/46) | 0.297 (43/46) | 0.196 (43/46) |
| isoST | 0.128 (44/46) | 0.113 (46/46) | 0.113 (46/46) | 0.283 (44/46) | 0.211 (44/46) | 0.209 (44/46) | 0.195 (44/46) | 0.254 (43/46) | 0.258 (43/46) | 0.236 (43/46) |
| DeepSpatial | 0.140 (44/46) | 0.116 (46/46) | 0.116 (46/46) | 0.219 (44/46) | 0.213 (44/46) | 0.209 (44/46) | 0.197 (44/46) | 0.262 (43/46) | 0.250 (43/46) | 0.244 (43/46) |
| UniST | 0.090 (44/46) | 0.430 (46/46) | 0.429 (46/46) | 0.500 (35/46) | 0.358 (35/46) | 0.240 (35/46) | 0.220 (44/46) | 0.112 (15/46) | 0.518 (15/46) | 0.157 (15/46) |

## Prediction diagnostics (mean across datasets)

Macro-average over datasets (only finite cells; N/A not imputed).

| method | gene_coverage_frac | pred_mean_ratio | zero_frac_pred |
|---|---|---|---|
| ReHo | 0.775 (46/46) | 1.060 (46/46) | 0.956 (46/46) |
| PASTE-OT-Linear | 0.854 (46/46) | 1.037 (46/46) | 0.234 (46/46) |
| NTF | 0.873 (46/46) | 0.858 (46/46) | 0.000 (46/46) |
| SINTER3D | 0.852 (46/46) | 1.229 (46/46) | 0.280 (46/46) |
| SpatialZ | 0.850 (46/46) | 1.064 (46/46) | 0.900 (46/46) |
| isoST | 0.871 (46/46) | 1.118 (46/46) | 0.479 (46/46) |
| DeepSpatial | 0.790 (46/46) | 1.057 (46/46) | 0.957 (46/46) |
| UniST | 0.423 (46/46) | 0.830 (46/46) | 0.636 (46/46) |

## N/A / support counts

| level | metric | N/A cells | top reasons |
|---|---|---|---|
| spatial | boundary_centroid_dist_bins | 1960 | boundary_pred_support_insufficient (1552); n_target<100 (152); no_visible_boundary_threshold (136) |
| spatial | boundary_recall_p80 | 1960 | boundary_pred_support_insufficient (1552); n_target<100 (152); no_visible_boundary_threshold (136) |
| spatial | boundary_recall_d1 | 1960 | boundary_pred_support_insufficient (1552); n_target<100 (152); no_visible_boundary_threshold (136) |
| spatial | boundary_precision_d1 | 1960 | boundary_pred_support_insufficient (1552); n_target<100 (152); no_visible_boundary_threshold (136) |
| spatial | boundary_precision_p80 | 1960 | boundary_pred_support_insufficient (1552); n_target<100 (152); no_visible_boundary_threshold (136) |
| spatial | boundary_precision_d0 | 1960 | boundary_pred_support_insufficient (1552); n_target<100 (152); no_visible_boundary_threshold (136) |
| spatial | boundary_f1_d0 | 1960 | boundary_pred_support_insufficient (1552); n_target<100 (152); no_visible_boundary_threshold (136) |
| spatial | boundary_recall_d0 | 1960 | boundary_pred_support_insufficient (1552); n_target<100 (152); no_visible_boundary_threshold (136) |
| spatial | hotspot_recall | 1508 | hotspot_pred_positive_insufficient (1292); n_target<100 (152); hotspot_gt_positive_insufficient (64) |
| spatial | hotspot_precision | 1508 | hotspot_pred_positive_insufficient (1292); n_target<100 (152); hotspot_gt_positive_insufficient (64) |
| spatial | hotspot_f1 | 1508 | hotspot_pred_positive_insufficient (1292); n_target<100 (152); hotspot_gt_positive_insufficient (64) |
| spatial | hotspot_precision_p80 | 781 | hotspot_p80_support_insufficient (629); n_target<100 (152) |
| spatial | hotspot_recall_p80 | 781 | hotspot_p80_support_insufficient (629); n_target<100 (152) |
| spatial | hotspot_PR_AUC | 216 | n_target<100 (152); hotspot_gt_positive_insufficient (64) |
| spatial | hotspot_AP | 216 | n_target<100 (152); hotspot_gt_positive_insufficient (64) |
| spatial | ms_ssim | 152 | n_target<100 (152) |
| spatial_macro | boundary_recall_d1 | 90 | all_modules_unsupported (90) |
| spatial_macro | boundary_precision_p80 | 90 | all_modules_unsupported (90) |
| spatial_macro | boundary_recall_p80 | 90 | all_modules_unsupported (90) |
| spatial_macro | boundary_precision_d0 | 90 | all_modules_unsupported (90) |
| spatial_macro | boundary_f1_d0 | 90 | all_modules_unsupported (90) |
| spatial_macro | boundary_recall_d0 | 90 | all_modules_unsupported (90) |
| spatial_macro | boundary_precision_d1 | 90 | all_modules_unsupported (90) |
| spatial_macro | boundary_centroid_dist_bins | 90 | all_modules_unsupported (90) |
| spatial_macro | hotspot_recall | 33 | all_modules_unsupported (33) |
| spatial_macro | hotspot_f1 | 33 | all_modules_unsupported (33) |
| spatial_macro | hotspot_precision | 33 | all_modules_unsupported (33) |
| spatial_macro | hotspot_precision_p80 | 21 | all_modules_unsupported (21) |
| spatial_macro | hotspot_recall_p80 | 21 | all_modules_unsupported (21) |
| spatial_macro | ms_ssim | 16 | all_modules_unsupported (16) |
| spatial_macro | hotspot_PR_AUC | 16 | all_modules_unsupported (16) |
| spatial_macro | hotspot_AP | 16 | all_modules_unsupported (16) |
| spatial | abs_delta_moran_i | 2 | spatial_unsupported (2) |
| spatial | abs_delta_geary_c | 2 | spatial_unsupported (2) |
| module | module_pcc | 2 | constant_module_score (2) |
| spatial | geary_pred | 2 | spatial_unsupported (2) |
| spatial | moran_i_pred | 2 | spatial_unsupported (2) |

## Failures

No failed dataset x method cells (all 368 predictions aligned, finite, non-negative).

## Sparse-target and support caveats

- 2/46 target slices have < 100 spots, so all spatial metrics (Moran/Geary, MS-SSIM, hotspot, boundary) are N/A for those datasets: Embryo__Drosophila__3Ddataset_3=99, Heart__Mouse__3Ddataset_1=52.
- 0 datasets have no valid marker modules (celltype categories with >=20 visible spots and >=5 markers); module/spatial metrics are N/A there.
- Dataset x method failures: 0 datasets with >=1 failed cell.
- Module macro values weight modules equally, not by size; hotspot/boundary use fixed visible-derived thresholds, so scale-shifted predictions lose recall by construction.
- Hotspot/boundary means are computed only on cells passing the support gates, and support differs by method (boundary d0: SINTER3D 6/46, UniST 15/46, others 42-43/46; hotspot d0: SINTER3D 38/46, UniST 35/46, others 42-44/46), so those columns are selection-biased and not directly comparable across methods.
- UniST and other methods can emit constant genes/spots; gene_pcc/gene_vr are computed only where both target and prediction are non-constant (see gene_coverage_frac), which favours methods that only predict a subset of genes.
- Sparse targets: median zero fraction across datasets = 0.978; gene PCC is computed only for genes non-constant in both target and prediction (n_genes_scored in long table).

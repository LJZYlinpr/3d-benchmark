# MERFISH 5-slice, all 10 methods: core metric table

Target-only held-out bins, 7 scenarios, 18 cell-type modules. Values are means over pooled (combo x module) rows; `combos_all_10_core` = scenarios where all 10 core metrics are finite.

**Completion:** all 10 methods have predictions for all 7 scenarios; no method is UNSUPPORTED/FAILED. Budgets: SINTER3D 3,000 steps; isoST 20 epochs/stage; DeepSpatial 20 epochs; UniST 50/100 GAE/INR epochs; PASTE-OT-Linear 6,000 anchors/bracket; Ours-EncMTP 3,000 steps. N/A cells are reported with reasons below (no GT hotspot/boundary or no prediction positive), never imputed.

| Method | Combos finite | Gene PCC (median) | RMSE | Module PCC | Module MS-SSIM | log2 var ratio | abs(delta Moran I) | Hotspot recall | Hotspot precision | Boundary recall d0 | Boundary precision d0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NTF | 7/7 | 0.06825 | 0.2522 | 0.1286 | 0.1404 | -3.007 | 0.2824 | 0.2251 | 0.1808 | 0.09414 | 0.1527 |
| Spateo | 7/7 | 0.4403 | 0.2144 | 0.4976 | 0.259 | -0.4063 | 0.1706 | 0.5055 | 0.3808 | 0.1863 | 0.352 |
| SINTER3D | 7/7 | 0.04843 | 0.2617 | 0.09205 | 0.1465 | -5.62 | 0.1555 | 0.1338 | 0.1641 | 0.0008171 | 0.1211 |
| isoST | 7/7 | 0.03426 | 0.5346 | 0.1153 | 0.0009175 | -0.1184 | 0.4663 | 0.6512 | 0.1242 | 0.3095 | 0.1322 |
| SpatialZ | 7/7 | 0.473 | 0.196 | 0.4976 | 0.4258 | -0.4634 | 0.04133 | 0.4341 | 0.4452 | 0.3201 | 0.3693 |
| DeepSpatial | 7/7 | 0.3523 | 0.2392 | 0.4024 | 0.356 | 0.1757 | 0.1706 | 0.4178 | 0.3885 | 0.3638 | 0.3123 |
| UniST | 7/7 | 0.3862 | 0.2142 | 0.3805 | 0.4201 | -4.332 | 0.1755 | 0.1505 | 0.6837 | 0.01196 | 0.665 |
| PASTE-OT-Linear | 7/7 | 0.4632 | 0.2048 | 0.4786 | 0.3279 | -0.352 | 0.04979 | 0.4421 | 0.4041 | 0.346 | 0.2986 |
| ReHo | 7/7 | 0.4107 | 0.2278 | 0.454 | 0.399 | 0.2132 | 0.03715 | 0.4489 | 0.4136 | 0.382 | 0.3348 |
| Ours-EncMTP | 7/7 | 0.4885 | 0.1969 | 0.5205 | 0.3436 | -0.3322 | 0.1414 | 0.4978 | 0.4403 | 0.2956 | 0.3722 |

## Diagnostics (means)

| Method | mean ratio | pred zero frac | gene PCC p10 | frac gene PCC>0.5 | coverage 10/10 | hotspot AP | boundary F1 d0 | boundary F1 d1 |
|---|---|---|---|---|---|---|---|---|
| NTF | 0.7758 | 8.772e-07 | -0.01156 | 0.005117 | 0.8413 | 0.1575 | 0.09664 | 0.1714 |
| Spateo | 1.36 | 0.3872 | 0.1586 | 0.4101 | 1 | 0.4005 | 0.2131 | 0.2814 |
| SINTER3D | 1.373 | 0.2865 | -0.05395 | 0.009375 | 0.1508 | 0.1407 | 0.01302 | 0.02772 |
| isoST | 3.748 | 0.4557 | -0.006959 | 0 | 1 | 0.1562 | 0.1483 | 0.2265 |
| SpatialZ | 1.009 | 0.6918 | 0.1663 | 0.4536 | 1 | 0.4102 | 0.3312 | 0.4726 |
| DeepSpatial | 1.012 | 0.8048 | 0.1056 | 0.262 | 1 | 0.3393 | 0.3299 | 0.4707 |
| UniST | 0.4949 | 0.8459 | 0.03067 | 0.3598 | 0.246 | 0.327 | 0.0841 | 0.1343 |
| PASTE-OT-Linear | 1.013 | 0.2127 | 0.1639 | 0.434 | 1 | 0.3794 | 0.3024 | 0.4336 |
| ReHo | 1.01 | 0.8063 | 0.131 | 0.3655 | 1 | 0.3689 | 0.3527 | 0.4969 |
| Ours-EncMTP | 1.132 | 0.3845 | 0.2194 | 0.4773 | 1 | 0.4274 | 0.3199 | 0.4442 |

## N/A reasons (pooled rows)

- **NTF**: {"boundary_precision(no_pred_boundary)": 19, "hotspot_precision(no_pred_positive)": 17}
- **Spateo**: {}
- **SINTER3D**: {"boundary_precision(no_pred_boundary)": 104, "hotspot_precision(no_pred_positive)": 76}
- **isoST**: {}
- **SpatialZ**: {}
- **DeepSpatial**: {}
- **UniST**: {"boundary_precision(no_pred_boundary)": 95, "hotspot_precision(no_pred_positive)": 84}
- **PASTE-OT-Linear**: {}
- **ReHo**: {}
- **Ours-EncMTP**: {}

## Perturbation validation (GT module maps only)

See `perturbation_curves.csv` / `perturbation_summary.json`; metric response curves are method-independent.

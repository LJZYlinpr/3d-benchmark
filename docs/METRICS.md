# Metrics

This package uses the **new metric suite**, not 50-PC features.

## Core metrics

### Expression

- median and mean per-gene PCC;
- per-gene RMSE, MAE, cosine and Spearman;
- per-gene variance ratio, log2;
- p10 per-gene PCC and fraction with PCC > 0.5.

### Modules

- per-dataset cell-type marker modules derived from **visible** cells only;
- module score = mean z-scored expression over module genes;
- module PCC, module variance ratio, module RMSE.

### Spatial

- masked 2D MS-SSIM per target section;
- global |Δ Moran's I| and |Δ Geary's C|;
- hotspot recall/precision/F1 and threshold-free AP where support is adequate;
- boundary recall/precision/F1 with δ=0 and δ=1 bin tolerance.

## Support and N/A

N/A is a first-class result. It must be reported with a reason:

- target slice has <100 spots;
- target/prediction has no positive hotspots;
- boundary support is insufficient;
- module score is constant;
- spatial coordinates are degenerate.

Do not replace N/A with 0, 1, or a copied value.

## No single score

Do not collapse all metrics into one weighted rank. Report:

- per-dataset tables;
- macro averages with the number of available datasets;
- support/N/A counts;
- selection-bias caveats when support differs across methods.

## Known metric caveats

- Gene-level PCC is unstable on sparse So3D data; use it only with gene coverage
  and zero-fraction context.
- MS-SSIM can reward smoothing and is secondary, not the primary module metric.
- Global Moran's I is insensitive to spatial shifts/local noise; pair it with
  hotspot/boundary metrics.
- Hotspot/boundary support differs by method; columns with fewer available
  datasets are not directly comparable without support reporting.

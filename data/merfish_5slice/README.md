# MERFISH 5-slice 50 µm input (compact, committed)

This directory holds the actual input consumed by the MERFISH 5-slice
benchmark, split into five GitHub-friendly shards.

| File | Slice | Bins | Size |
|---|---|---:|---:|
| `observed_50um_S1.h5ad` | C57BL6J-3.006 | 24,677 | 24.8 MB |
| `observed_50um_S2.h5ad` | C57BL6J-3.007 | 25,281 | 25.6 MB |
| `observed_50um_S3.h5ad` | C57BL6J-3.008 | 25,287 | 24.7 MB |
| `observed_50um_S4.h5ad` | C57BL6J-3.009 | 24,151 | 23.7 MB |
| `observed_50um_S5.h5ad` | C57BL6J-3.010 | 23,273 | 22.2 MB |
| **total** | | **122,669** | **~116 MiB (121 MB)** |

Each shard is an AnnData with:

```text
X                 (n_bins, 1122) float32 CSR, mean expression per 50 µm bin
obs               slice, x_mm, y_mm, z_mm, n_cells, ix, iy, iz, celltype, domain
obsm['X_aligned'] (n_bins, 3) float32, aligned CCF xyz in mm
var['gene_name']  1122 genes
uns               bin_size_um=50, grid_nx, grid_ny, grid_x0, grid_y0
```

## Why shards instead of one file

The raw cell-level file is 2.9 GB:

```text
Data/benchmark_data/merfish/ABCA-3_slices_006-010.h5ad
615,400 cells x 1,122 genes, dense float32 X (2.76 GB)
```

The benchmark never reads that file directly. It starts from the 50 µm binned
matrix (`observed_50um.h5ad`, 122,669 bins). A single lossless float32 CSR
version of that matrix is ~115–120 MB, above GitHub's 100 MB hard file limit.
Splitting by slice gives five ~22–26 MB files; the total is unchanged and every
file is below both the 100 MB hard limit and the 50 MB warning threshold.

The shards are **lossless**: the rebuild check reports
`max_abs_X_diff = 0.0` and `max_abs_X_aligned_diff = 0.0` against the full HPC
copy.

## Rebuild the single input file

```bash
python scripts/build_observed_50um.py \
  --shards-dir data/merfish_5slice \
  --out observed_50um.h5ad
```

The output is byte-value equivalent in `X`, coordinates, gene names, bin IDs,
`celltype` and `domain` to the full `observed_50um.h5ad` used for the reported
5-slice results. To verify against an HPC reference:

```bash
python scripts/build_observed_50um.py \
  --shards-dir data/merfish_5slice \
  --out /tmp/observed_50um.h5ad \
  --check /path/to/Workspace/merfish_test/observed_50um.h5ad
```

## What was removed

Compared with the raw `ABCA-3_slices_006-010.h5ad`:

- cell-level rows replaced by 50 µm bin means (615,400 → 122,669 rows);
- dense `X` replaced by float32 CSR (the dominant size reduction);
- removed `obsm['SpaLP']` (32-d embedding, ~79 MB), `X_CCF`, `X_original`,
  `X_umap`, `X_spatial_coords`, duplicate `spatial`;
- removed ontology IDs, label-transfer confidence scores, QC flags and other
  `obs` columns not used by the benchmark; kept 10 columns required by the
  canonical builder and module metrics.

The raw cell-level file is not committed. It can be obtained from the Allen
Brain Cell Atlas (`Zhuang-ABCA-3`, CC BY 4.0) or kept outside Git under
`Data/benchmark_data/merfish/`.

## Provenance and license

- Source: Allen Brain Cell Atlas, `Zhuang-ABCA-3`, sections `.006–.010`
  (C57BL6J-3), 1,122-gene MERFISH panel.
- ABC Atlas data are released under CC BY 4.0; retain attribution when
  redistributing these derived shards.
- Binning: 50 µm grid in x/y, mean expression per bin, bins with at least one
  cell; z is the section-level CCF-aligned depth.

See `scripts/build_observed_50um.py` for the exact concatenation logic.

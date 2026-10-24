# MERFISH 5-slice 50 µm data package (vendored)

The five per-slice shards and the query grid are **tracked in this Git
repository** under `data/merfish_5slice/`.  A release-asset copy also exists
(`data-v1/merfish_5slice_data_v1.tar.gz`) for consumers that prefer not to clone
binary blobs; `scripts/fetch_data_merfish5.sh` is optional and verifies its
SHA-256.

## Contents

| File | Slice | Bins | Size |
|---|---|---:|---:|
| `observed_50um_S1.h5ad` | C57BL6J-3.006 | 24,677 | 24.8 MB |
| `observed_50um_S2.h5ad` | C57BL6J-3.007 | 25,281 | 25.6 MB |
| `observed_50um_S3.h5ad` | C57BL6J-3.008 | 25,287 | 24.7 MB |
| `observed_50um_S4.h5ad` | C57BL6J-3.009 | 24,151 | 23.7 MB |
| `observed_50um_S5.h5ad` | C57BL6J-3.010 | 23,273 | 22.2 MB |
| `grid_50um.npz` | 50 µm query grid for NTF/Spateo | — | 36 KB |
| **total** | | **122,669** | **~112 MiB** |

Each shard is an AnnData with:

```text
X                 (n_bins, 1122) float32 CSR, mean expression per 50 µm bin
obs               bin_id (obs index; added as a column by the merge script),
                  slice, x_mm, y_mm, z_mm, n_cells, ix, iy, iz, celltype, domain
obsm['X_aligned'] (n_bins, 3) float32, aligned CCF xyz in mm
var['gene_name']  1122 genes
uns               bin_size_um=50, grid_nx, grid_ny, grid_x0, grid_y0
```

`SHA256SUMS` records the per-file hashes, including the grid.

## Rebuild input and canonical data

```bash
python scripts/build_observed_50um.py \
  --shards-dir data/merfish_5slice \
  --out "$MERFISH5_ROOT/observed_50um.h5ad"

MERFISH5_ROOT="$MERFISH5_ROOT" python src/canonical/make_targets.py
MERFISH5_ROOT="$MERFISH5_ROOT" python src/canonical/merfish_5slice.py
cp data/merfish_5slice/grid_50um.npz "$MERFISH5_ROOT/grid_50um.npz"
```

`scripts/reproduce_m1a.sh` runs all of the above and the ten methods.

## Why compact shards

The raw cell-level file is 2.9 GB:

```text
ABCA-3_slices_006-010.h5ad
615,400 cells x 1,122 genes, dense float32 X
```

The benchmark never reads that file. It uses the 50 µm binned matrix. The
five lossless shards are ~112 MiB and rebuild exactly:

```text
max_abs_X_diff = 0.0
max_abs_X_aligned_diff = 0.0
```

`bin_id` is the shard `obs` index; the merge script materialises it as an
explicit `obs['bin_id']` column for the canonical builder.

## Provenance and license

- Source: Allen Brain Cell Atlas, `Zhuang-ABCA-3`, sections `.006–.010`
  (C57BL6J-3), 1,122-gene MERFISH panel.
- ABC Atlas data are released under CC BY 4.0; retain attribution when
  redistributing these derived shards.
- Binning: 50 µm grid in x/y, mean expression per bin, bins with at least one
  cell; z is the section-level CCF-aligned depth.

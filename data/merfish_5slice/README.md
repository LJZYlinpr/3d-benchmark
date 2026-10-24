# MERFISH 5-slice 50 µm data package

The data are **not tracked in the Git code repository**.  They are a separate
GitHub release asset:

```text
tag:    data-v1
asset:  merfish_5slice_data_v1.tar.gz
url:    https://github.com/LJZYlinpr/3d-benchmark/releases/download/data-v1/merfish_5slice_data_v1.tar.gz
sha256: 187af1cc750634512b58c8333a86eb94dec5b619bd24b2074319f94fcc67ab08
```

Download and unpack it into this directory:

```bash
bash scripts/fetch_data_merfish5.sh
```

The script verifies the tarball SHA-256 and extracts the five shards plus the
query grid.  To use an already-downloaded tarball:

```bash
MERFISH5_DATA_TARBALL=/path/to/merfish_5slice_data_v1.tar.gz \
  bash scripts/fetch_data_merfish5.sh
```

## Contents

| File | Slice | Bins | Size |
|---|---|---:|---:|
| `observed_50um_S1.h5ad` | C57BL6J-3.006 | 24,677 | 24.8 MB |
| `observed_50um_S2.h5ad` | C57BL6J-3.007 | 25,281 | 25.6 MB |
| `observed_50um_S3.h5ad` | C57BL6J-3.008 | 25,287 | 24.7 MB |
| `observed_50um_S4.h5ad` | C57BL6J-3.009 | 24,151 | 23.7 MB |
| `observed_50um_S5.h5ad` | C57BL6J-3.010 | 23,273 | 22.2 MB |
| `grid_50um.npz` | 50 µm query grid for NTF/Spateo | — | 36 KB |
| **total** | | **122,669** | **~112 MiB (tarball)** |

Each shard is an AnnData with:

```text
X                 (n_bins, 1122) float32 CSR, mean expression per 50 µm bin
obs               bin_id (obs index; added as a column by the merge script),
                  slice, x_mm, y_mm, z_mm, n_cells, ix, iy, iz, celltype, domain
obsm['X_aligned'] (n_bins, 3) float32, aligned CCF xyz in mm
var['gene_name']  1122 genes
uns               bin_size_um=50, grid_nx, grid_ny, grid_x0, grid_y0
```

`SHA256SUMS` records the per-file hashes inside the tarball.

## Why a separate package

The raw cell-level file is 2.9 GB:

```text
ABCA-3_slices_006-010.h5ad
615,400 cells x 1,122 genes, dense float32 X
```

The benchmark never reads that file.  It uses the 50 µm binned matrix.  The
five lossless per-slice shards are ~112 MiB; keeping them out of Git makes the
code package small and lets the data package be versioned independently.

The shards are lossless: rebuilding `observed_50um.h5ad` gives
`max_abs_X_diff = 0.0` and `max_abs_X_aligned_diff = 0.0` against the HPC
reference.  `bin_id` is the shard `obs` index; the merge script materialises it
as an explicit column for the canonical builder.

## Rebuild the single input and canonical data

From the code repository (after downloading the data package):

```bash
python scripts/build_observed_50um.py \
  --shards-dir data/merfish_5slice \
  --out "$MERFISH5_ROOT/observed_50um.h5ad"

MERFISH5_ROOT="$MERFISH5_ROOT" python src/canonical/make_targets.py
MERFISH5_ROOT="$MERFISH5_ROOT" python src/canonical/merfish_5slice.py
cp data/merfish_5slice/grid_50um.npz "$MERFISH5_ROOT/grid_50um.npz"
```

## Provenance and license

- Source: Allen Brain Cell Atlas, `Zhuang-ABCA-3`, sections `.006–.010`
  (C57BL6J-3), 1,122-gene MERFISH panel.
- ABC Atlas data are released under CC BY 4.0; retain attribution when
  redistributing the derived shards.
- Binning: 50 µm grid in x/y, mean expression per bin, bins with at least one
  cell; z is the section-level CCF-aligned depth.

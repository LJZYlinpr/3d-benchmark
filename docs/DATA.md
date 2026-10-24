# Data

No `.h5ad`, prediction, or volume file is committed to this repository. The
MERFISH 5-slice input is a separate release asset (`data-v1`); this repository
contains only its manifest and hashes under `data/merfish_5slice/`. The
following describes all inputs used by the benchmark.

## MERFISH

Primary source:

```text
Data/benchmark_data/merfish/Reconstruction.h5ad
  8,380,288 cells x 1,122 genes
  239 slice categories
```

Derived 5-slice subset used for the current held-out matrix:

```text
sections .006-.010
50 um legacy bins
7 held-out scenarios M1a-M3a
```

Separate data package (no raw cells needed):

```text
data-v1 / merfish_5slice_data_v1.tar.gz
  observed_50um_S1.h5ad ... S5.h5ad + grid_50um.npz
  122,669 50 µm bins x 1,122 genes, float32 CSR, ~112 MiB tarball
```

Download and rebuild the single file consumed by the canonical builder:

```bash
bash scripts/fetch_data_merfish5.sh
python scripts/build_observed_50um.py \
  --shards-dir data/merfish_5slice \
  --out observed_50um.h5ad
```

The shards keep only the fields the benchmark uses (`slice`, `x_mm`, `y_mm`,
`z_mm`, `n_cells`, `ix`, `iy`, `iz`, `celltype`, `domain`,
`obsm['X_aligned']`, gene names). `bin_id` is the shard `obs` index; the merge
script materialises it as an explicit `obs['bin_id']` column because the
canonical builder reads it. The rebuild is value-identical in `X` and
coordinates to the full HPC `observed_50um.h5ad`. The 2.9 GB cell-level file
`ABCA-3_slices_006-010.h5ad` is **not** read by the benchmark and is not
committed; it can be obtained from ABC Atlas (`Zhuang-ABCA-3`, CC BY 4.0).

The historical 50 µm grid is kept only for reproduction of the older 5-slice
results. The v1 grid policy is 10/20/40/100 µm.

Grid files use:

```text
x_centers, y_centers, z_centers
obs_mask, query_mask, inside, query_near5
x0, y0, z0, bs, nx, ny, nz
```

## Heart

```text
Fig4to6_3DHeart.h5ad
  3,095,373 cells x 238 genes
  53 sections; z = obsm['X_spateo'][:,2]
  selected group: sections 21-30
```

`X` is normalized expression; raw counts are in `layers['counts']`.
The heart 10-slice experiment uses 26 contiguous-gap scenarios.

## So3D

```text
Data/benchmark_data/so3d/ge5_slices/h5ad/
  46 datasets with >=5 slices
  901,284 spots
  2k/3k/4k gene panels
  obsm['spatial'] = xyz
  obs['slice'], obs['z'], cell-type labels
```

The manifest is `manifests/so3d_46_manifest.tsv`.

Important caveats:

- only 7/46 datasets have clearly physical z spacing;
- 4 use slice index; 5 are irregular; the rest have ambiguous z units;
- target zero fraction is very high; gene-level PCC is unstable;
- only `Embryo,Drosophila,3Ddataset_16` has unannotated spots.

## Preprocessing rules

- never use target expression to compute normalization, PCA, modules, masks,
  thresholds, or checkpoint selection;
- target coordinates are allowed under the fixed-coordinate protocol;
- keep the gene order from `var_names` and save it with predictions;
- do not silently drop target rows; record support and N/A.

## Data availability

Raw data are not in Git. Place them under a local `Data/` tree and point the
scripts to that root. The package includes only manifests and small summaries.

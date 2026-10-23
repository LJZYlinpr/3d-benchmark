# NTF on MERFISH 50 µm grid — run notes (v2, valid)

## Result

- Output: `NTF_50um_uniform.h5ad` — **523,163 × 1,122**, float32, dense
  - rows = the grid `inside` mask (same 523,163 voxels as the isoST lane)
  - 400,677 query voxels (predicted) + 122,486 observed voxels (measured values copied verbatim,
    `max|Δ| = 0` vs input)
  - 183 observed voxels fall outside `inside` (SPEC: `query ∪ obs = 523,346` vs `inside = 523,163`);
    they are reported in `metrics.json` (`n_observed_outside_inside`) and are **not** added as rows
  - `obs`: `x_mm, y_mm, z_mm, ix, iy, iz, is_observed, slice`; `obsm['spatial']`; gene names identical
- In-sample PCC (observed slices, all 122,669 spots): **r(all) = 0.904**, r(nonzero) = 0.879
  (median 0.918), 1115/1122 genes with a defined PCC, RMSE 0.064. Query predictions vary spatially
  (per-gene std 0.152 vs 0.192 observed), min 0.0 / max 16.07, all finite.
- GPU: 1×A100-40GB, gvna02, job 9383994 (`srun --overlap`); process peak 1.51 GB alloc /
  1.63 GB reserved; device-wide peak 4.49 GB (co-tenant xpy job included).
- Time: train 281.8 s (12,000 iters) + sample 0.8 s + write 2.3 s = 302 s total.

## Why v1 was invalid (root cause)

`NTF.train()` uses **one Adam with lr=1e-4 for every parameter**. tinycudann initialises the hash
grid at `U(-1e-4, 1e-4)`, so with lr 1e-4 the encoding never grows: measured
`pe std = 2.8e-5`, `dL/d(encoding params) = 8e-10`, encoding params even shrink under weight decay.
The network output therefore stays at `softplus(0) = 0.693` (v1: constant 0.686), loss flat at
~0.45, and the library's early stopping fires at 3,600 iters on a flat curve.

Hypotheses checked (see `pilot_matrix.json`, 20k-spot in-sample eval, job 9383994):

| tag | change | iters | r_all | r_nz | pred_std |
|---|---|---|---|---|---|
| A | default lr 1e-4 (no bias net, no dropout) | 1500 | 0.079 | 0.135 | 0.006 |
| B | param-group lr | 2000 | 0.878 | 0.848 | 0.188 |
| C | B + dropout net (`no_dropout=False`) | 2000 | 0.832 | 0.874 | 0.189 |
| D | B + width 128 / depth 2 | 2000 | 0.876 | 0.849 | 0.189 |
| E | B + weight_expr 1.0 | 2000 | 0.879 | 0.849 | 0.185 |

- Bias net is **not** involved: `n_levels_bias=0` (default) means no `log_bias` module exists.
- Dropout / variance path is **not** the cause: collapse reproduces with `--no_dropout`; dropout
  is slightly worse on r_all and adds spurious exact zeros.
- More capacity did not help at this budget; `weight_expr=1.0` was marginally best (kept).
- Scale check: `SpatialOmicsDataset.v` is raw `adata.X`; only *coordinates* are bbox-normalised.
  In-sample `pred_mean 0.06655` vs `gt_mean 0.06655` → no missing de-normalisation.

## Final configuration

- Data: `obsm['spatial'] = X_aligned` (mm), `obs['slice']` (5 slices), raw dense `X`.
- Bounding box (monkeypatched, shared repo untouched): grid extent + 1 voxel
  `[-1.075, 0.625, 3.625] .. [14.325, 7.125, 4.575]`. NTF normalises **per axis**, so z
  (0.837 mm) spans the full [0,1] range and is not squashed; an explicit bbox avoids the
  library's `resolution.max()*2 = 0.4 mm` margin that wasted ~50 % of the z range.
- PSF resolution forced isotropic 0.05 mm (the data-derived z gap of 0.19–0.25 mm would blur z
  by ~6 voxels).
- Hash grid 13 levels, base res 2, scale 1.5, 2 features/level, log2 table 20; MLP width 64,
  depth 1; n_samples 4; `--single_precision --no_dropout`; `weight_expr=1.0`.
- Optimizer (wrapper loop, ~20 lines around the library model/loss): Adam with
  `{hash grid: 1e-2, rest: 1e-3}`, `weight_decay=0`, MultiStepLR γ=0.5 at 0.5/0.7/0.9,
  12,000 iters, no early stopping.
- Query `slice` = label of the nearest observed z layer (0/4/9/13/17 → 3.010/3.009/3.008/3.007/3.006).

## Files

- `run_ntf_50um.py` — wrapper (param-group training loop, bbox/PSF overrides, sampling, export)
- `NTF_50um_uniform.h5ad`, `metrics.json`, `status.json`, `pilot_matrix.json`, `run.log`, `NOTES.md`
- kept: `INVALID_NTF_50um_uniform.h5ad`, `status.INVALID.json` (v1 artifacts, untouched);
  `NTF_50um_uniform.valid_0937.h5ad` (previous 523,346-row run, renamed by the monitor)
- coordination mirror: `../run50_state.json`
- diagnostics outside the project: `/storage/yuhongtaoLab/linpeiran/tmp/run50_ntf/`

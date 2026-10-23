# PASTE-OT-Linear

## Implementation

- Package: `src/methods/merfish/PASTE-OT-Linear/run_paste_heart.py`
- So3D adapter: `src/methods/so3d/PASTE-OT-Linear/run_paste_so3d.py`

## Environment

`spateo` (contains POT/torch).

## Method

Custom baseline: entropic fused Gromov-Wasserstein alignment of two visible
bracket slices, followed by z-linear expression interpolation and inverse
distance weighting.

## Command template

```bash
srun --jobid=<job> --overlap --nodelist=<node> --ntasks=1 --cpus-per-task=4 \
  /path/to/envs/spateo/bin/python src/methods/merfish/PASTE-OT-Linear/run_paste_heart.py \
  --scenarios all --max-anchors 6000
```

## Output

Target-only predictions; no volume.

## Known deviations / P0 issues

- **Not faithful PASTE**: PPA solver, `max_iter=100`, `tol=1e-7`, 6,000 anchors,
  no nearest-anchor expansion. The reference uses OT/Frank–Wolfe and 20,000
  anchors.
- **M2b bracket bug**: the code uses one `1→5` bracket for target `{2,4}` and
  ignores visible section 3. M2b predictions equal corresponding M3a rows. M2b
  must be rerun before any paper table.
- Sinkhorn did not converge in 264/322 solves in the current 5-slice run.
- Some genes are constant when absent from both bracket slices.
- Target expression is not used; this is a method/implementation issue, not a
  target leakage issue.

## Status

Predictions exist for 5-slice, heart and So3D, but the M2b cells and the
"PASTE" label are **not release-ready**.

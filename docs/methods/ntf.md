# NTF

## Implementation

- Package: `src/methods/merfish/NTF/run_ntf_50um.py`
- Heart adapter: `src/methods/heart/NTF/run_ntf_heart.py`
- So3D adapter: `src/methods/so3d/NTF/run_ntf_so3d.py`

## Environment

`ntf`

## Inputs

- visible expression matrix, coordinates and slice labels;
- target coordinates;
- grid/metadata from `configs/resolutions.yaml` for 10/20/40/100 µm runs.

## Command template

```bash
srun --jobid=<job> --overlap --nodelist=<node> --ntasks=1 --cpus-per-task=4 \
  /path/to/envs/ntf/bin/python src/methods/merfish/NTF/run_ntf_50um.py
```

Run from the directory expected by the adapter or set its `DATA` path. The
historical 50 µm script is retained for the 5-slice result; v1 grid runs must
point to `grid_10um.npz`, `grid_20um.npz`, `grid_40um.npz` or
`grid_100um.npz`.

## Output

`predictions/<scenario>/<method>.npz`, target rows only. No dense volume.

## Dataset tracks

- MERFISH 5-slice: run with `ntf` env; 4,000 iterations, batch 4096.
- Heart: visible-only bounding box/PSF, 50 µm canonical bins.
- So3D: per-dataset resolution estimate; variable gene count.

## Known deviations / caveats

- GPU run-to-run nondeterminism was observed.
- NTF is sensitive to under-budgeting; So3D results should be interpreted with
  its reduced budget.
- NTF has high in-sample fit but poor held-out behavior on MERFISH gaps; do not
  use in-sample PCC as a ranking signal.
- `pred_zero_fraction` was 0 in the 5-slice run, while the target is sparse.

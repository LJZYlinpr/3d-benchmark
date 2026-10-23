# DeepSpatial

## Implementation

- Package: `src/methods/merfish/DeepSpatial/run_heart.py`
- So3D adapter: `src/methods/so3d/DeepSpatial/run_so3d.py`

## Environment

`deepspatial`

## Command template

```bash
srun --jobid=<job> --overlap --nodelist=<node> --ntasks=1 --cpus-per-task=4 \
  /path/to/envs/deepspatial/bin/python src/methods/merfish/DeepSpatial/run_heart.py
```

Use 10/20/40/100 µm grids for v1; 50 µm is legacy.

## Output

Target-only predictions; no dense volume.

## Dataset tracks

- Heart: 120 epochs, 15k samples, 50 µm thickness.
- So3D: 20 epochs, patch size `ceil(G/32)` instead of official patch 8;
  the official patch yields 250–500 tokens and quadratic attention (pilot 19 GB,
  ~200 s/epoch).

## Known deviations / caveats

- Patch-size and epoch-budget changes are significant; report them.
- In-sample PCC=1.0 claims are not held-out evidence: observed rows are copied
  by construction in some configurations.
- On So3D, module PCC ~0.20 but boundary recall is relatively good; use both.
- Do not use target-derived patch statistics.

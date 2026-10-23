# SpatialZ

## Implementation

- Package: `src/methods/merfish/SpatialZ/run_heart.py`
- So3D adapter: `src/methods/so3d/SpatialZ/run_spatialz.py`

## Environment

`spatialz`

## Command template

```bash
srun --jobid=<job> --overlap --nodelist=<node> --ntasks=1 --cpus-per-task=4 \
  /path/to/envs/spatialz/bin/python src/methods/merfish/SpatialZ/run_heart.py
```

Use 10/20/40/100 µm grid files for v1.

## Output

Target-only predictions; no dense volume.

## Dataset tracks

- 5-slice: official fast setting; lr=40, 1000 iterations, 200 projections,
  k_sam=3.
- Heart: same fast setting; 41 s/scenario.
- So3D: pseudo-mm xy frame; single class `all`; official code largely
  unmodified.

## Known deviations / caveats

- SpatialZ acts more like a fast splatting/reconstruction method than a learned
  held-out interpolator.
- On So3D, target coverage was 0.31–0.91; uncovered target points were filled by
  nearest virtual points. Report coverage, not just metrics.
- In-sample PCC can be 1.0 by construction when observed rows are copied; never
  use that as a held-out score.

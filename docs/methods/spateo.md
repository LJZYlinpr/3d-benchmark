# Spateo

## Implementation

- Package: `src/methods/merfish/Spateo/run_spateo_50um.py`
- Heart adapter: `src/methods/heart/Spateo/run_spateo_heart.py`

## Environment

`spateo`

## Role

Spateo is a **supplemental interpolation reference**, not one of the 8
common-task methods. It is not included in the paper's 8-method table.

## Command template

```bash
srun --jobid=<job> --overlap --nodelist=<node> --ntasks=1 --cpus-per-task=4 \
  /path/to/envs/spateo/bin/python src/methods/merfish/Spateo/run_spateo_50um.py
```

For 10/20/40/100 µm, point the adapter to the corresponding grid file. The
historical 50 µm script is for the older 5-slice result.

## Output

Target-only predictions; no dense volume.

## Known deviations / caveats

- 5-slice: 40k iterations/scenario (reference 60k).
- 10 µm: a post-hoc per-gene calibration was used in an earlier run. If the
  calibration used target/query statistics, it is an **oracle calibration** and
  must not be used in the main held-out table. Only visible-only calibration is
  valid.
- Spateo SIREN variants can memorize observed slices; use ReLU/validated
  configuration for held-out claims.
- Spateo is strong on held-out MERFISH gaps but is not part of the 8-method
  common task.

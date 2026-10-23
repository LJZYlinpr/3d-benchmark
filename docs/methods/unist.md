# UniST

## Implementation

- Package: `src/methods/merfish/UniST/run_heart.py`
- So3D adapter: `src/methods/so3d/UniST/run_unist_so3d.py`
- Generated per-dataset SUICA scripts are excluded from Git.

## Environment

`unist`

## Command template

```bash
srun --jobid=<job> --overlap --nodelist=<node> --ntasks=1 --cpus-per-task=4 \
  /path/to/envs/unist/bin/python src/methods/merfish/UniST/run_heart.py \
  --gae-epochs 50 --inr-epochs 100
```

For v1, use the 10/20/40/100 canonical data; 50 µm is legacy.

## Output

Target-only predictions; no dense volume.

## Dataset tracks

- Heart: GAE 50 / INR 100; 26/26 scenarios.
- So3D: GAE 150 / INR 300; 5.32 h for 46 datasets, 416 s/dataset mean.
- Official 1,000/2,000 epochs would be ~24 h for So3D.

## Known deviations / caveats

- Reduced GAE/INR budgets are documented.
- UniST can emit constant genes and has low target coverage on sparse data.
- On So3D, variance ratio collapsed severely; do not interpret high hotspot
  precision without recall/coverage.
- Never train on target expression.

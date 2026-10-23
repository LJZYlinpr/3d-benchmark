# isoST

## Implementation

- Package: `src/methods/merfish/isoST/run_heart.py`
- So3D adapter: `src/methods/so3d/isoST/run_so3d.py`

## Environment

`isoST`

## Command template

```bash
srun --jobid=<job> --overlap --nodelist=<node> --ntasks=1 --cpus-per-task=4 \
  /path/to/envs/isoST/bin/python src/methods/merfish/isoST/run_heart.py
```

Use 10/20/40/100 µm grids for v1; 50 µm is legacy.

## Output

Target-only predictions; no dense volume.

## Dataset tracks

- 5-slice: 40/40/40 epochs.
- Heart: 40/40/40 epochs, hidden 64.
- So3D: 10/10/10 epochs; the official 100×3 configuration diverged in a
  visible-only sweep (ratio 3.8–4.9), while 5–20 epochs were stable.

## Known deviations / caveats

- Reduced epoch budget on So3D.
- isoST is over-dispersed on heart (mean 0.444 vs GT 0.337).
- In the 5-slice metric table, isoST had very low module PCC and near-zero
  MS-SSIM in one configuration; treat as a stability/scale issue, not a clean
  method ranking.
- Validate any output scale correction on visible data only.

# SINTER3D

## Implementation

- Package: `src/methods/merfish/SINTER3D/run_sinter3d_heart.py`
- So3D adapter: `src/methods/so3d/SINTER3D/run_sinter3d.py`
- Upstream 50/100/20/10 scripts are available in the local workspace but are
  not duplicated in this package.

## Environment

`SINTER3D_env`

## Command template

```bash
srun --jobid=<job> --overlap --nodelist=<node> --ntasks=1 --cpus-per-task=4 \
  /path/to/envs/SINTER3D_env/bin/python src/methods/merfish/SINTER3D/run_sinter3d_heart.py
```

Point the adapter to the canonical visible/target data and grid for the desired
resolution (10/20/40/100 µm). Do not use the historical 50 µm grid as the v1
resolution.

## Output

Target-only predictions; no full volume.

## Dataset tracks

- MERFISH 5-slice: 6,000 steps; visible-only KMeans basis; canonical X treated
  as target-like expression without renormalization.
- Heart: 6,000 steps; lower budget than default.
- So3D: mini-batch SGD, 6,000 steps; visible-only coordinate rescale; separate
  patch setting.

## Known deviations / caveats

- Official full-batch training is replaced by mini-batch training.
- Reduced step budget and visible-only KMeans are documented deviations.
- So3D output is smooth and low-expression genes are often all-zero; module
  PCC can collapse toward zero.
- Do not claim official full-budget SINTER3D from this adapter.

# So3D 46 x 8 baseline run plan

Date: 2026-10-22  
Input: `/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/Data/benchmark_data/so3d/ge5_slices/h5ad/`  
Output root: `/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/Workspace/so3d_bench/`

## Scope

- 46 datasets with >=5 serial slices; 901,284 spots.
- One held-out middle slice per dataset (both sides visible); target coordinates retained; target expression is evaluation-only.
- Spot-level fixed-coordinate prediction; all genes per dataset (2k/3k/4k); X treated as provided normalized values (no renormalization).
- 8 methods: SINTER3D, UniST, isoST, SpatialZ, PASTE-OT-Linear, ReHo, NTF, DeepSpatial.
- Target-only outputs; no dense volume; no GT in training/calibration.

## Outputs

```text
Workspace/so3d_bench/
  canonical/<dataset>/visible.h5ad, target.npz, meta.json
  predictions/<dataset>/<method>.npz
  metrics/
  logs/
  methods/<method>/
```

## GPU assignment

| Runner | Job | Methods |
|---|---|---|
| A | 9383991 | ReHo, PASTE-OT-Linear, NTF |
| B | 9383992 | SINTER3D, SpatialZ |
| C | 9383993 | isoST, DeepSpatial |
| D | 9383994 | UniST |
| prep | CPU | canonical dataset/scenario builder |
| metrics | CPU | 10 metrics after predictions |

All use `srun --jobid=... --overlap`; no scancel/CONT; one method process per lane.

## Acceptance

- 46/46 datasets have a target and visible set with no overlap.
- 46 x 8 predictions attempted; failures explicit (UNSUPPORTED/FAILED), never fabricate.
- All prediction rows align to `target.npz`; finite; no target X use.
- Metrics output separate from paper's existing MERFISH 5-slice results.

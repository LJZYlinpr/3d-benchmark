# Environment

This repository itself is pure Python, but the 10 method adapters depend on
multiple upstream environments. The current HPC installation uses separate
Conda environments.

## Environment map

| Environment | Used by |
|---|---|
| `spateo` | Ours-EncMTP (5-slice/heart), PASTE-OT-Linear, Spateo |
| `ntf` | NTF, Ours-EncMTP (some So3D runs) |
| `SINTER3D_env` | SINTER3D, DeepSpatial |
| `isoST` | isoST |
| `spatialz` | SpatialZ |
| `deepspatial` | DeepSpatial |
| `unist` | UniST |
| `benchmark` | data preparation, metrics, generic analysis |
| `tools` | helper scripts |

`envs/*.yml` are lightweight `conda env export --from-history` files, not full
lockfiles. For exact reproduction, generate a complete lock on the machine that
produced the result:

```bash
conda env export -n spateo > envs/spateo.full.yml
pip freeze -r envs/spateo.full.yml > envs/spateo.pip.txt
```

## Current gaps

- No single unified environment exists yet.
- CUDA/PyTorch/POT versions are not pinned across environments.
- Some upstream method code is path-sensitive; see `docs/REPRODUCE.md`.
- `spateo` and `unist` are the most important environments for the current
  common task; `SINTER3D_env` is used for the historical 5-slice results.

## GPU

The current experiments use A100-40GB jobs through Slurm. ReHo is CPU-only.
Methods should be run on a GPU allocation for training/inference; metrics can
run on CPU. Do not submit heavy training onto the coding node.

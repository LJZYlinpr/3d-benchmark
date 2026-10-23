# Reproduction

Paths below are placeholders; set them for your machine. On the original HPC
installation, data live under
`/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/`.

## 0. Check installation

```bash
python scripts/check_install.py
python -m pytest tests/test_grid.py -q
```

## 1. Build 10/20/40/100 µm grids

```bash
python src/benchmark/grid.py \
  --cells cells.npz \
  --out-dir grids \
  --resolutions 10 20 40 100
```

`cells.npz` must contain `xyz` (mm) or `x`, `y`, `z`. The output schema is
documented in `docs/DATA.md`. The historical 50 µm grids are not produced by
this v1 command.

## 2. MERFISH 5-slice

Data root example:

```bash
export DATA_ROOT=/path/to/3d_benchmark/Data/benchmark_data/merfish
export WORK_ROOT=/path/to/3d_benchmark/Workspace/merfish_test
export ENV_BIN=/path/to/conda/envs/spateo/bin/python
```

Canonical builder:

```bash
$ENV_BIN src/canonical/merfish_5slice.py
```

Method runners live in `src/methods/merfish/<method>/`. Run each method on the
canonical scenario directory; keep target-only outputs under
`predictions/<scenario>/<method>.npz`.

The historical final table can be regenerated with:

```bash
cd $WORK_ROOT/missing_combos_50um/metrics_v1_all_final
$ENV_BIN compute_core.py
$ENV_BIN finalize.py
```

Inputs must be present before running; see `docs/DATA.md`.

## 3. Heart 10-slice

```bash
$ENV_BIN src/canonical/heart_10slice.py
```

Run the method adapters in `src/methods/heart/` for all 26 scenarios. The
canonical directory must contain `bins_all.h5ad`, `scenarios.json`,
`target_gt.npz` and `READY`. Do not use target expression in fitting.

## 4. So3D 46

Build canonical targets:

```bash
$ENV_BIN src/canonical/so3d_46.py
```

Then run the 8 common methods in `src/methods/so3d/` for each dataset. The
external metric adaptation is in `src/metrics/so3d/`:

```bash
$ENV_BIN src/metrics/so3d/so3d_metrics.py
```

Current results and per-dataset table are in `results/` and
`manifests/SO3D_BY_DATASET.md`.

## 5. Environment lock

`envs/*.yml` are `--from-history` exports, not full locks. Before publishing a
result, save complete environments and pip freezes:

```bash
bash scripts/freeze_envs.sh
```

## 6. What not to do

- Do not copy target expression into visible h5ad or training code.
- Do not use target-derived normalization or model selection.
- Do not overwrite existing predictions; use a new run directory.
- Do not present 50 µm legacy results as v1 10/20/40/100 results.
- Do not present the 5-slice result as the 54-slice paper result.

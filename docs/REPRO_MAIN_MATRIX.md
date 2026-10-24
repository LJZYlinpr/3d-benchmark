# Main server 10-method x 7-combo matrix (2026-10-24)

Completed on `main` (8x RTX 4090D, 384 CPU, 1.5 TB RAM).  Four GPUs were used
(0, 1, 4, 5); GPU3 had a co-tenant and was avoided.

```text
start:   2026-10-24 11:14 +0800
finish:  2026-10-24 13:16 +0800
wall:    ~2 h 02 min
runs:    70 method-combo predictions (M1a reused from the pilot)
```

All 70 predictions pass `validate_predictions.py` (shape, finite values and
scenario row alignment):

```text
M1a: 10  M1b: 10  M1c: 10  M2a: 10  M2b: 10  M2c: 10  M3a: 10
```

The matrix script had a validation-only bug: it omitted `--scenario "$s"` and
therefore checked every scenario against M1a rows.  The predictions were
correct; the script is fixed in `scripts/run_merfish5_matrix_4gpu.sh` and the
70 files were revalidated.

## Metrics: module-mean gene PCC

| method | M1a | M1b | M1c | M2a | M2b | M2c | M3a | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Spateo | 0.5996 | 0.5784 | 0.6100 | 0.5797 | 0.5138 | 0.4953 | 0.5273 | **0.5577** |
| Ours-EncMTP | 0.5414 | 0.4844 | 0.5088 | 0.4280 | 0.5183 | 0.4619 | 0.3987 | **0.4773** |
| SpatialZ | 0.4971 | 0.4825 | 0.4362 | 0.4261 | 0.4691 | 0.4071 | 0.3799 | **0.4426** |
| PASTE-OT-Linear | 0.4427 | 0.4948 | 0.4835 | 0.3964 | 0.4011 | 0.4224 | 0.3762 | **0.4310** |
| ReHo | 0.4148 | 0.4259 | 0.3779 | 0.4046 | 0.3989 | 0.4035 | 0.3770 | **0.4004** |
| DeepSpatial | 0.3478 | 0.3892 | 0.3469 | 0.3214 | 0.3519 | 0.3474 | 0.2985 | **0.3433** |
| UniST | 0.1538 | 0.2600 | 0.2361 | 0.1811 | 0.2566 | 0.2134 | 0.2024 | **0.2148** |
| NTF | 0.1987 | 0.2088 | 0.1707 | 0.1229 | 0.1901 | 0.1628 | 0.0875 | **0.1631** |
| SINTER3D | 0.0825 | 0.1487 | 0.1828 | 0.0900 | 0.0349 | 0.0643 | 0.0825 | **0.0980** |
| isoST | 0.0677 | 0.0553 | 0.0247 | 0.0528 | 0.0751 | 0.0651 | 0.0244 | **0.0522** |

Full 70-row table: `results/merfish5_main_metrics_all.csv`.
Per-method summary and HPC deltas: `results/merfish5_main_metrics_summary.csv`.

## Comparison with the HPC table

`gene_pcc_median` is the same module-mean definition as the HPC
`metrics_v1_all` table.

| method | mean diff vs HPC | max abs diff | reading |
|---|---:|---:|---|
| ReHo / PASTE / SpatialZ | 0.0000 | 0.0002 | exact |
| DeepSpatial | -0.0013 | 0.0030 | close |
| isoST | -0.0032 | 0.0135 | close |
| Ours-EncMTP | +0.0099 | 0.0202 | GPU/version variance |
| SINTER3D | +0.0240 | 0.0710 | near-zero signal, unstable |
| NTF | +0.0648 | 0.1255 | 3k-iteration pure-PyTorch fallback, not the HPC budget |
| Spateo | +0.1074 | 0.2868 | main uses the held-out export; HPC used the target-only adapter/config |
| UniST | -0.1279 | 0.1850 | from-scratch 300/400 in a different env; HPC reference is a different run/config |

The deterministic methods confirm exact reproducibility.  NTF, Spateo and
UniST remain **not like-for-like**; their absolute values should not be compared
to the paper table without freezing the same budgets/configs/seeds.

## Data / code separation

- Code + minimal vendored upstream: Git `main` (`vendor/methods/`).
- Data: five shards + grid tracked in `data/merfish_5slice/`; optional release
  mirror `data-v1 / merfish_5slice_data_v1.tar.gz` (SHA-256
  `187af1cc750634512b58c8333a86eb94dec5b619bd24b2074319f94fcc67ab08`).
- `scripts/fetch_data_merfish5.sh` is optional if the release mirror is used.

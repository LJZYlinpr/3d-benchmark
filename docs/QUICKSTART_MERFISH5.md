# MERFISH 5-slice quick start (fresh clone)

This is the complete, fresh-clone path to the ten-method M1a reproduction.
Data shards and the minimal upstream method code are **vendored in this
repository**; no HPC paths, release downloads, or separate upstream fetch are
required. Only the Python environments are installed from the network.

## 1. Clone

```bash
git clone https://github.com/LJZYlinpr/3d-benchmark.git
cd 3d-benchmark
export MERFISH5_ROOT="$PWD/work"
```

Requirements: Linux x86_64, an NVIDIA GPU, Python 3.9/3.10 (common profile) or
3.10 (Ada profile), and internet access for pip/PyTorch wheels.

## 2. Environment

Common profile (torch 1.13 + CUDA PyG + KeOps; tested on 2080 Ti):

```bash
bash scripts/setup_merfish5_env.sh
source "$MERFISH5_ROOT/env.sh"
```

RTX 4090 / Ada or a host without conda:

```bash
MERFISH5_PROFILE=ada bash scripts/setup_merfish5_env_ada.sh
source "$MERFISH5_ROOT/env.sh"
```

The scripts create `venv` (core/isoST/NTF/Spateo), `venv_omni`
(DeepSpatial/UniST), and a local CUDA toolkit if needed. They never modify the
system/conda base environment. Exact versions are pinned in `envs/merfish5/`.

## 3. Build data and run all ten methods

```bash
bash scripts/reproduce_m1a.sh
```

This runs the full pipeline:

```text
data/merfish_5slice/*.h5ad
  -> observed_50um.h5ad
  -> targets/<combo>/target.npz
  -> canonical/{bins_all.h5ad,scenarios.json,target_gt.npz}
  -> predictions/M1a/{ReHo,Ours-EncMTP,PASTE-OT-Linear,SpatialZ,
                       SINTER3D,DeepSpatial,isoST,NTF,Spateo,UniST}.npz
  -> metrics_M1a.csv
  -> strict validation against manifests/merfish5_m1a_expected.json
```

The data rebuild is value-identical to the HPC reference; canonical hashes are
recorded in `manifests/merfish5_m1a_expected.json` and
`docs/REPRO_SERVER1.md`.

Full budgets are the frozen reference budgets (NTF 12k iters, Spateo 60k,
UniST 50/100, Ours/SINTER3D 3k, isoST/DeepSpatial 20 epochs). Expect roughly
1–2 h on a 4090 and 4–5 h on a 2080 Ti; Spateo dominates.

For a quick smoke test (tiny budgets, no strict validation):

```bash
PILOT=1 bash scripts/reproduce_m1a.sh
```

For a subset:

```bash
bash scripts/reproduce_m1a.sh ReHo Ours-EncMTP SpatialZ
```

## 4. Check the result

The one-command run ends with:

```text
[ok] data shards + grid
[ok] canonical bins_all X <sha>…
[ok] target_gt <sha>…
[ok] ReHo / PASTE / SpatialZ / ...
[ok] M1a reproduction validated
```

Deterministic methods (ReHo, PASTE-OT-Linear, SpatialZ) must match exactly;
learned methods have documented tolerances in the manifest because of
GPU/library non-determinism and NTF/SINTER3D near-zero-signal variance.

Outputs:

```text
$MERFISH5_ROOT/predictions/M1a/
$MERFISH5_ROOT/metrics_M1a.csv
```

## 5. Package contents

```text
data/merfish_5slice/       five lossless 50 µm shards + grid + SHA256SUMS
vendor/methods/            minimal pinned upstream code for all methods
vendor/shims/torch_cluster CPU kNN shim for isoST
manifests/                 modules + expected M1a metrics/hashes
src/canonical/             observed -> targets -> canonical builders
src/methods/merfish/       target-only benchmark adapters
scripts/                   setup, reproduce, validate, evaluation
docs/                      environment, data, methods, metrics, provenance
```

`vendor/methods/README.md` lists the upstream commit anchors, trimmed paths and
licenses. `docs/THIRD_PARTY.md` records attribution and modifications.

## 6. Troubleshooting

- If `data.pyg.org` is blocked, the common setup prints the three required CUDA
  wheel names; download them elsewhere and install with `pip install --no-deps`.
  The Ada profile uses the system CUDA 12.1 toolkit instead.
- `isoST` peaks around 10.8 GB; run it alone on an 11 GB card.
- `SINTER3D` is numerically unstable on the near-zero M1a signal; its strict
  tolerance is intentionally loose.
- `NTF` uses the pure-PyTorch hash-grid fallback; tiny-cudann is optional.
- `Spateo` uses the target-only holdout export; do not substitute the
  full-volume h5ad (it contains measured values at observed rows).
- Status/log files are written under `$MERFISH5_ROOT/methods/<method>/`.

## 7. Reproduction provenance

The package was validated on server 1 (RTX 2080 Ti) and main (RTX 4090D); see
`docs/REPRO_SERVER1.md` and `docs/REPRO_MAIN_MATRIX.md`. The canonical data
chain and deterministic methods reproduce exactly; NTF/Spateo/UniST remain
configuration-sensitive and are reported with explicit tolerances.

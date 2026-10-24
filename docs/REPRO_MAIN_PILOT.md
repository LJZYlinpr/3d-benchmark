# Main server pilot (2026-10-24)

First reproduction attempt on `main` (8x RTX 4090D, 384 CPU, 1.5 TB RAM,
driver 595.99.02).  The goal was to validate the Git package on Ada GPUs
before a full 10-method x 7-combo run.

## Isolation

```text
/data/linpeiran/3d_benchmark_repro_20261024/
  repo/                 fresh clone of the Git package
  work/venv_ada         torch 2.3.1+cu121, PyG 2.3, KeOps
  work/venv_omni        torch 2.6.0+cu124, lightning/timm
  work/canonical        rebuilt from the committed shards
  upstream/methods      fetched at pinned commits (+ vendored SINTER3D)
  logs/pilot/
```

Only GPUs 0-3 were used.  GPU3 had a co-tenant job using ~19 GB; heavy methods
were moved to free GPUs.

## M1a pilot

Data chain (from the committed shards only):

```text
shards -> observed_50um.h5ad -> targets -> canonical
READY: 5 sections, 298,876 target rows
```

Predictions (target-only contract):

| method | status | wall | GPU peak (device) | pred mean |
|---|---|---:|---:|---:|
| ReHo | DONE | 7.4 s | CPU | 0.0660 |
| Ours-EncMTP | DONE | 52.2 s | 20,817 MB | 0.0780 |
| PASTE-OT-Linear | DONE | 443.4 s | 21,525 MB | 0.0672 |
| SpatialZ | DONE | 56.1 s | 1,716 MB | 0.0680 |
| SINTER3D | DONE | 290.9 s | OOM on shared GPU3; 4.7 min on free GPU0 | 0.1048 |
| NTF | DONE | 191.6 s | 21,525 MB (device); torch ~1.9 GB | 0.0739 |
| Spateo | DONE | ~5 min | ~1.1 GB (torch) | 0.0819 |
| isoST | DONE | 137.5 s | 21,278 MB | 0.2000 |
| DeepSpatial | DONE | 350.5 s | 10,224 MB | 0.0686 |
| UniST | RUNNING | started 10:45 | 1.3 GB | — |

`validate_predictions.py`:

```text
[ok] 9 prediction files
```

UniST is CPU-heavy in the SUICA phase; on server 1 it took ~52 min.  It was
still running when this pilot was recorded.

## Main-specific fixes now in Git

- `scripts/setup_merfish5_env_ada.sh`: torch 2.3.1+cu121 common env and
  torch 2.6+cu124 omni env; system CUDA 12.1; virtualenv fallback for hosts
  without ensurepip/conda; mirror URLs for the blocked PyTorch index.
- isoST upstream is missing a project-local `utils/` directory; added under
  `docs/upstream/new_files/isoST`.
- `fetch_upstream.sh` validates cached tarballs before extraction.
- SpatialZ requires `SOMENDER` (MENDER), `squidpy`, `shapely`, and a compatible
  numba; pandas pinned 2.0.0.
- SINTER3D can OOM on a 24 GB card shared with a co-tenant; schedule it alone.
- PASTE's OT step is slow on this stack (~425 s per 6000x6000 problem), so the
  full matrix needs ~7 min/combo for PASTE alone.

Latest package commits: `b9c2c42` (Ada profile), `579f000` (isoST utils +
tarball validation), `44acd08` (SpatialZ deps).  The pilot env was built from
`b9c2c42` plus the same manual package installs now encoded in `44acd08`; a
fresh clone at `44acd08` should reproduce the environment with
`scripts/setup_merfish5_env_ada.sh`.

## Full matrix

Not started.  Proposed four-GPU schedule:

```text
GPU0: UniST x 7
GPU1: DeepSpatial x 7, isoST x 7
GPU2: NTF x 7, Spateo x 7, SINTER3D x 7
GPU3: PASTE x 7, Ours/ReHo/SpatialZ x 7
```

Pick free GPUs (avoid co-tenants), keep heavy methods alone where possible,
and validate all 70 predictions before computing metrics.

# AI reproduction runbook (MERFISH 5-slice)

This is the operational guide for an AI agent on a fresh machine. The
scientific/data path is already self-contained in the repository; almost all
failures are environment or hardware issues. Follow the order below and do not
modify `data/`, `canonical/`, or `manifests/merfish5_m1a_expected.json`.

## 0. Golden rules

- Work in a separate `$MERFISH5_ROOT` (default `./work`); never write into the
  Git tree except temporary wheels.
- Do not change the target expression, budgets, seeds, or expected metrics to
  make a check pass.
- Deterministic methods must match exactly; NTF/Spateo/UniST are
  configuration-sensitive and are reported as `[info]`, not failures.
- If a method fails, keep the surviving predictions and report honestly.

## 1. Preflight (do this first)

```bash
git clone https://github.com/LJZYlinpr/3d-benchmark.git
cd 3d-benchmark
bash scripts/preflight_merfish5.sh
```

It checks Linux/x86_64, Python, GPU/VRAM/compute capability, CUDA/nvcc, free
disk, data presence and network reachability. It recommends `common` or `ada`
and exits non-zero only on hard blockers.

Minimum practical hardware:

| GPU | VRAM | Profile | Notes |
|---|---:|---|---|
| RTX 2080 Ti / 3080 | 11 GB | common | all 10 serial; isoST ~10.8 GB, DeepSpatial ~9.9 GB; Spateo 60k ≈ 3.3 h |
| RTX 3090 / A5000 | 24 GB | common | comfortable |
| A100 40/80 GB | 40/80 GB | common | fastest; CUDA 11.7 profile |
| RTX 4090 / H100 | 24/80 GB | ada | needs CUDA 12.1 toolkit for KeOps |
| CPU only | — | — | **not supported** |

Budget ~15 GB disk for environments + canonical data, and 1–2 h for a 4090
(Spateo dominates). A 2080 Ti can take 5 h.

## 2. Install the environment

Common (Turing/Ampere: 2080 Ti, 3090, A100):

```bash
bash scripts/setup_merfish5_env.sh
source work/env.sh
```

Ada (RTX 4090 / H100; auto-selected by `reproduce_m1a.sh` for sm_89/90):

```bash
MERFISH5_PROFILE=ada bash scripts/setup_merfish5_env_ada.sh
source work/env.sh
```

The scripts are idempotent. If setup is interrupted, rerun it; only remove
`work/venv*` if the venv is corrupted. Do not `pip install -r` the captured
`envs/merfish5/*.pip.txt` files over a working env: they are freezes for
diagnostics, not a single compatible install plan.

## 3. Run

```bash
bash scripts/reproduce_m1a.sh                    # all 10 methods, strict
PILOT=1 bash scripts/reproduce_m1a.sh            # tiny smoke test
bash scripts/reproduce_m1a.sh ReHo PASTE-OT-Linear SpatialZ
```

`reproduce_m1a.sh` builds observed → targets → canonical, runs the selected
methods, writes `metrics_M1a.csv`, and runs the strict validator. If one method
fails, it continues with the survivors, validates them with `--allow-subset`,
and exits 1 with a warning. Per-method logs are under
`$MERFISH5_ROOT/methods/<method>/run.log`.

## 4. Failure triage

### Environment / setup

| Symptom | Cause | Action |
|---|---|---|
| `python3.10: command not found` | Ada assumes Python 3.10 | install 3.10, set `PYTHON=/path/to/python3.10`; common also needs 3.9/3.10 |
| `python3.11` and anndata/numba/PyG install failures | pins target 3.9/3.10 | create a 3.10 env; do not force 3.11 |
| `nvidia-smi not found` / `torch.cuda.is_available() == False` | no GPU/driver | install NVIDIA driver; CPU-only is unsupported |
| `LD_LIBRARY_PATH: unbound variable` | old generated `env.sh` | `git pull`; or change to `${LD_LIBRARY_PATH:-}`; rerun setup |
| `ResolutionImpossible` | mixing lockfiles / old pins | use `setup_merfish5_env*.sh`, not `pip install -r`; `git pull` |
| `data.pyg.org` SSL/DNS error | CUDA PyG wheel index unreachable | see §5; install wheels manually or run the 9 non-isoST methods |
| no `nvcc` under `/usr/local/cuda-12.1` | Ada profile needs CUDA 12.1 toolkit for KeOps | set `ADA_CUDA_HOME=/path/to/cuda-12.x`; if unavailable, skip isoST |
| pip download stalls | slow default index | `export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`; rerun; wheels cache under `work/wheels` |
| `No space left on device` | envs + canonical > disk | set `MERFISH5_ROOT=/larger/disk/work`; remove `work/venv*` and retry |

### Data / canonical build

| Symptom | Cause | Action |
|---|---|---|
| missing `data/merfish_5slice/S1..S5` | incomplete clone | re-clone; `cd data/merfish_5slice && sha256sum -c SHA256SUMS` |
| `KeyError: bin_id` | old commit | `git pull` (merge script now materialises `bin_id`) |
| `numpy._core` load error | old target npz | regenerate with current `src/canonical/make_targets.py` |
| data/canonical hash mismatch | corrupted download/modified data | stop; re-clone/clean; do not patch expected hashes |
| `canonical/READY` missing | build step failed | run the three commands from `docs/QUICKSTART_MERFISH5.md` manually and read the first error |

### Method-specific

| Symptom | Cause | Action |
|---|---|---|
| PyG `torch_cluster.random_walk` / `knn` errors | some torch_cluster builds omit ops | setup calls `scripts/patch_pyg_torch_cluster.py`; ensure real `torch_cluster` precedes `vendor/shims` in `PYTHONPATH` |
| `SpatialZ`: `MENDER`, `spatialdata`, `dask legacy` | squidpy 1.3 pulls spatialdata/dask | install `squidpy==1.2.3 --no-deps` (setup does); rerun SpatialZ |
| `isoST`: `Not compiled with CUDA support` / `torch_sparse` | CPU-only PyG or shim only | use the documented CUDA PyG env; shim is a last-resort CPU kNN fallback |
| `isoST`: segfault at `coo training` | torch 2.8/PyG 2.6 CSR stack | use the pinned torch 2.3+cu121 Ada profile or torch 1.13 common; do not mix |
| `isoST`/`DeepSpatial` CUDA OOM | peaks ~21 GB / ~10 GB | run alone on a free GPU; `CUDA_VISIBLE_DEVICES=0`; small cards: skip isoST |
| NTF crash in hash grid | unpatched fallback | current vendor has `NTF_torch_fallback.patch`; `git pull` |
| NTF/Spateo/UniST metric differs | configuration/GPU-sensitive | expected `[info]`; check contract and trend, do not tune to ref |
| SINTER3D metric differs | near-zero signal, unstable | expected; do not quote its exact value |
| Spateo 3 h on 2080 Ti | 60k FP64 iterations | normal; use a 4090/A100 for speed |
| SSH drop / `BrokenPipeError` | relay reset | rerun; remove the incomplete method npz and rerun that method; use tmux/nohup |

### Validation

| Result | Meaning | Action |
|---|---|---|
| `[ok] ReHo/PASTE/SpatialZ` | deterministic reproduction | good |
| `[ok] Ours/DeepSpatial/isoST/SINTER3D` | within documented tolerance | good |
| `[info] NTF/Spateo/UniST` | contract only, metric config-sensitive | not a failure; report values and config |
| `[FAIL]` on a strict row | data/env drift or real regression | inspect hash, log, versions; do not edit expected metrics |
| `[missing] <method>` | method failed or not selected | inspect method log, rerun that method or accept subset |
| `[FAIL] data hash` | corrupted/modified data | re-clone; never patch `SHA256SUMS` |

## 5. Manual PyG wheel fallback

If `data.pyg.org` is blocked, download wheels on a machine with access. Match
the profile:

- common: torch 1.13.1+cu117, CPython 3.9/3.10
- Ada: torch 2.3.1+cu121, CPython 3.10

Required packages: `torch_scatter`, `torch_sparse`, `torch_cluster` (versions
compatible with the torch build). Copy them to the host and install:

```bash
pip install --no-deps /path/to/torch_scatter-*.whl \
  /path/to/torch_sparse-*.whl /path/to/torch_cluster-*.whl
```

Then rerun setup or use `SKIP_ENV=1` with an existing `env.sh`. If only isoST is
blocked, run the other nine:

```bash
bash scripts/reproduce_m1a.sh \
  ReHo Ours-EncMTP PASTE-OT-Linear SpatialZ SINTER3D DeepSpatial NTF Spateo UniST
```

## 6. How to report a partial/complete run

Create `REPRO_STATUS.md` with:

```text
commit:
machine / GPU / VRAM / driver:
python / torch / CUDA:
profile:
commands:
setup time / run time:
method status (rc, wall, GPU peak):
validator output:
deviations / skipped methods / blockers:
```

Do not report a method as successful if its prediction is missing or its
contract check failed. Do not re-label `[info]` as `[ok]`.

## 7. What not to do

- Do not use the full-volume Spateo/NTF outputs as target predictions; they
  contain measured values at observed rows.
- Do not compare a 50-PC PCA metric with the current full-gene suite.
- Do not change the target slice, bins, genes, or module definitions to make a
  metric pass.
- Do not run multiple GPU methods on the same 11 GB card concurrently.
- Do not commit `work/`, predictions, or generated canonical files.

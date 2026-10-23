# UniST on So3D (46 datasets, one held-out middle slice each)

Runner D / lane job `9383994` (gvna02, 4 CPU, 1 A100 via `srun --overlap`).

## Method

Official SUICA (`benchmark/methods/UniST/external/SUICA_pro`) 3D-sparse pipeline, code
unmodified and reused verbatim from `benchmark/scripts/rerun_unist_official.py`
(`PIPELINE_SCRIPT`):

1. GAE embedder (`embedder_gae_3d_sparse.yaml`: GAE 2048-512-128 -> 64, anisotropic
   KNN k=6, z_weight 2.0, z_scale_factor 1.5) fit on **visible spots only**;
2. INR (`inr_embd_3d_sparse.yaml`: FFN, 3x2048, enhanced/anisotropic positional encoding)
   fit on coordinates -> GAE embeddings, with the GAE decoder, **visible spots only**;
3. decoder reconstruction at the **held-out target spot coordinates only**.

Full gene set per dataset (2000/3000/4000 genes); no PCA/50-PC features for the model.
X is used as provided by the canonical data (no renormalization).

Adapted from `Workspace/merfish_test/UniST/run_unist_50um.py` and the heart target-only
adapter. Runner: `run_unist_so3d.py`.

## Contract / leakage

- Inputs: `canonical/<dataset>/visible.h5ad` + `meta.json`; target coordinates read from
  `canonical/<dataset>/target.npz` keys `coords, obs_names, slice, z` only.
- `target.npz['X']` (GT) is **never opened** by the runner or the verifier.
- Output: `predictions/<dataset>/UniST.npz` — `pred (n_target, n_genes) float32`,
  `obs_names`, `coords`, `slice`, `genes`, plus echo fields; row order exactly the
  canonical target order. Non-negative, finite, clipped at 0.
- Per dataset: `predictions/<dataset>/UniST.status.json` (DONE/FAILED/UNSUPPORTED).
- No full 3D volume is written. Large intermediates (`visible_data.h5ad`, `X_pred.npy`)
  are deleted after each dataset; `methods/UniST/run/<dataset>/pipeline.log` is kept.

## Config / deviations

| item | official | here |
|---|---|---|
| GAE epochs | 1000 | **150** |
| INR epochs | 2000 | **300** |

Measured on the lane (fitted from the timing runs): per-epoch GAE ~= 0.45 + 0.052*n_k s and
per-epoch INR ~= 0.25 + 0.026*n_k s, where n_k = visible spots in thousands. With 46
datasets / 832,849 visible spots on one GPU lane, 150/300 was chosen to fit ~5 h; the same
setting is applied to every dataset.
This is a documented throughput deviation, not a per-dataset tuning.

`torch.cuda` peak per process 3.4-4.0 GB (device-wide peak 4041 MiB).

## Results

- 46/46 datasets DONE, coverage 1.0; 0 FAILED, 0 UNSUPPORTED.
- Wall 19,142 s = 5.32 h (mean 416 s/dataset; min 159 s `Heart__Human__3Ddataset_2`,
  max 1069 s `Larvae__Drosophila__3Ddataset_2`).
- 46 prediction files, 3.2 GB total; every file: shape == target, float32, finite,
  min >= 0, `obs_names/coords/slice/genes` exactly aligned, all rows distinct.
- Independent verifier `verify.py` -> `verify.json`: 598 checks, 0 failures, 1 warning.

Warning: `Embryo__Drosophila__3Ddataset_30` has only 348/4000 genes varying across target
spots (all 2005 rows distinct; zero-fraction 0.95). This is SUICA's INR/decoder smoothing
on a weak-signal dataset, not a contract violation. Median across datasets: ~40% of genes
vary across target spots.

Small datasets handled without special casing: smallest visible sets
`Embryo__Drosophila__3Ddataset_4` (1654 spots), `Embryo__Drosophila__3Ddataset_3` (2167);
all above SUICA's KNN k=6 + 20% validation split requirements, so none needed
UNSUPPORTED.

## Files

```
methods/UniST/run_unist_so3d.py      runner
methods/UniST/verify.py              independent acceptance check
methods/UniST/verify.json            598 checks, 0 failures
methods/UniST/compute_metrics.json   per-dataset status/timing/pred stats + caveats
methods/UniST/status.json            last heartbeat (final DONE)
methods/UniST/run.log                driver log
methods/UniST/run/<dataset>/         pipeline.log + query_coords.npy
../../RUNNER_D_STATUS.json           aggregate runner status
```

# Method Changes Log

> Records all modifications made to third-party method code under `methods/`.
> Per AGENTS.md §5.1-2: only minimal adaptations allowed (path fixes, `__init__.py`, dependency pinning).
> Each entry must record: date, file changed, what was changed, reason, and the method repo commit hash at time of change.

| Date | Method | File(s) | Change | Reason | Upstream Commit |
|------|--------|---------|--------|--------|-----------------|
| 2026-08-06 | SINTER3D | benchmark/src/adapters/sinter3d_adapter.py | New adapter: subprocess-based, uses inference_latent(decode=True) to predict expression at query coordinates | Enables SINTER3D for MERFISH LOSO tasks without modifying method code | — |
| 2026-08-06 | — | benchmark/src/metrics/*.py | Metrics engine: P0 (TDA), P1 (zero-inflation), P2 (ranking stability), P3 (SSIM), P4 (error propagation), P5 (SVG recovery) | Enhanced evaluation beyond the initial metric set | — |
| 2026-08-06 | — | benchmark/scripts/eval_spatialz_z14.py | New evaluation script for SpatialZ with full metrics | Evaluates existing SpatialZ z14 results | — |
| 2026-08-06 | — | benchmark/scripts/run_merfish_baselines.py | New baseline framework for MERFISH LOSO_z14 | Standardized baseline comparison | — |
| 2026-08-07 | isoST | benchmark/src/adapters/isost_adapter.py | New adapter: PCA transform + isoST training pipeline; verified data format (batch_num=1 nested structure); CPU-only (PyTorch 1.12 cu113 incompatible with driver 580/CUDA 13.0) | Enables isoST for MERFISH LOSO tasks; blocked by GPU unavailability | — |
| 2026-08-07 | — | benchmark/src/adapters/naive_baselines.py | Fixed PASTE-OT-Linear exception handler: added fallback X_lower_mean/X_upper_mean assignment when OT fails | Bug fix: UnboundLocalError when paste_pairwise_align raises exception | — |
| 2026-08-07 | — | .gitignore | Changed methods/ to /methods/ to only match root-level methods/ directory | Fixes notebooks/methods/ being incorrectly excluded from git | — |
| 2026-08-07 | — | benchmark/env (benchmark conda) | Upgraded PyTorch: 2.13.0+cpu → 2.14.0.dev+cu130 (CUDA 13.0 nightly) | GPU enablement; blocked by GPU2 hardware error at system level | — |

## Large-scale 54-slice benchmark adapters (2026-08-11)

**Branch**: `feat/large-scale-benchmark`

### Z-axis coordinate fix (SINTER3D, UniST, isoST)
**Issue**: All three INR/SDE adapters used `coords[:, 1]` (Y column) as Z coordinate,
a leftover from the old 5-slice dataset's 2D spatial convention. The 54-slice
dataset has proper 3D spatial coordinates with Z in column 2.
**Fix**: Changed `coords[:, 1]` → `coords[:, 2]` in all three adapters. Updated
SINTER3D's 3D_coor construction to use obsm['spatial'] directly.

### isoST PCA leakage fix
**Issue**: PCA.fit_transform() was called on ALL data before train/test split,
leaking holdout expression information into the PCA basis.
**Fix**: Split visible/holdout first, then PCA.fit() on visible only,
PCA.transform() on holdout. Min-max normalization also fitted on visible only.

### Continuous-Z handling (naive_baselines, isoST)
**Issue**: `np.unique(z_all)` on continuous Z coordinates returned 994k values
instead of 54 slice-level values, causing O(n²) loops.
**Fix**: naive_baselines._validate_and_prepare now computes per-slice Z means
using obs['slice'] labels when Z has >1000 unique values. isoST iterates over
slice labels instead of unique Z values.

### Parameterized holdout_slices
**Issue**: All adapters hardcoded `holdout_z=14` (single float), unable to
handle multi-slice holdout (gap=2, gap=3).
**Fix**: All adapters now accept `holdout_slices` (list of slice label strings)
in config, with backward compatibility for `holdout_z`.

### SINTER3D generated script fix
**Issue**: `training_steps` variable not properly substituted in generated
script f-string (escaped braces `{{training_steps}}` instead of `{training_steps}`).
**Fix**: Changed to single braces for proper variable substitution.

| 2026-08-11 | SINTER3D | benchmark/src/adapters/sinter3d_adapter.py | Z-axis fix ([:,1]→[:,2]), 3D_coor uses spatial directly, holdout_slices support, training_steps substitution fix | 54-slice dataset compatibility | feat/large-scale-benchmark |
| 2026-08-11 | UniST | benchmark/src/adapters/unist_adapter.py | Z-axis fix ([:,1]→[:,2]), holdout_slices support | 54-slice dataset compatibility | feat/large-scale-benchmark |
| 2026-08-11 | isoST | benchmark/src/adapters/isost_adapter.py | Z-axis fix, PCA leakage fix, continuous-Z fix, holdout_slices support, added pandas import | 54-slice dataset compatibility + no-leakage compliance | feat/large-scale-benchmark |
| 2026-08-11 | Naive baselines | benchmark/src/adapters/naive_baselines.py | Continuous-Z fix: per-slice Z means when >1000 unique Z values | 54-slice dataset compatibility | feat/large-scale-benchmark |

**Issue**: PyTorch Lightning DataLoader uses num_workers=18 (hardcoded), causing 
multi-process deadlock on this system (40+ processes hanging during KNN graph construction).

**Fix**: Monkey-patch `torch.utils.data.DataLoader.__init__` to cap `num_workers` at 0 
(single-process loading). Does not affect algorithm — only changes data loading parallelism.

**Status**: Minimal adaptation per AGENTS.md §5.1 rule 2.

## UniST SUICA INR — PyTorch version incompatibility (2026-08-10)

**Issue**: `networks/ffn.py:FourierFeatureNet.__init__` passes `dtype=None, device=None` 
to `nn.Linear()`, which is incompatible with the installed PyTorch version in the `unist` env.

**Status**: F1_env — environment/dependency failure. 
GAE Embedder (stage 1) completed successfully. INR (stage 2) blocked by this issue.
Resolution requires either updating PyTorch or fixing the FFN constructor.

**Workaround applied**: DataLoader num_workers capped at 0 to fix multi-process deadlock.

## isoST GPU training fixes (2026-08-11)

### Issue
Training crashed on GPU with three linked errors:
1. `'float' object has no attribute 'item'` — `total_loss = 0.0` (Python float) in `base_model.py:101`
2. `'float' object has no attribute 'backward'` — SinkhornLoss from geomloss returned Python float
3. `element 0 does not require grad` — `torch.argmin` in `one_nn_mse` disconnects gradient chain

### Fixes applied

| File | Change | Reason |
|------|--------|--------|
| `methods/isoST/model/base_model.py:101` | `total_loss = 0.0` → `total_loss = torch.tensor(0.0, device=self.device)` | Initialize as tensor to preserve grad chain |
| `methods/isoST/model/base_model.py:116-117` | Add `isinstance(loss, torch.Tensor)` guard with tensor conversion | Handle geomloss returning Python float |
| `methods/isoST/model/base_model.py:127` | Add type check before `.item()` | Defensive: only call `.item()` on tensors |
| `methods/isoST/model/utils/losses.py` | Removed 2 duplicate `one_nn_mse` definitions (was defined 3×) | Code cleanup |
| `methods/isoST/model/utils/losses.py:SinkhornLoss` | Tensor wrapping: `torch.tensor(loss, device=..., requires_grad=True)` if geomloss returns float | Ensure gradient flow through Sinkhorn loss |
| `methods/isoST/model/utils/losses.py:one_nn_mse` | `torch.argmin` → `F.softmin(distances/temperature) @ true_features` | Differentiable soft nearest neighbor preserves gradient flow |

**Status**: All 5 GPU smoke tests pass (coo loss, joint loss, train_step, 5-step training, GPU memory). Verified on RTX 3090.

## UniST/SUICA adapter fixes (2026-08-11)

### Issues
1. **Recursion error**: Monkey-patched `torch.empty` and `torch.zeros` to strip `dtype=None, device=None` kwargs caused infinite recursion in PyTorch 2.5 internals.
2. **`empty()` argument error**: Using `target=embeddings` required pre-computed GAE embeddings. Raw visible data had no embeddings, so `dim_out=None` was passed to `nn.Linear`, causing `torch.empty((None, dim_hidden))`.
3. **Checkpoint mismatch**: `train_inr()` returns None; adapter used stale checkpoint path for prediction.

### Fixes applied

| Issue | Change | Reason |
|-------|--------|--------|
| Duplicate monkey-patches | Removed duplicate `torch.load` patching (defined twice) | Code cleanup |
| torch.empty/zeros recursion | Removed monkey-patches for `torch.empty` and `torch.zeros` | PyTorch 2.5 handles `dtype=None, device=None` natively; patches caused recursion |
| dim_out=None | Changed `target` from `embeddings` to `raw_representations` | Raw data has no pre-computed embeddings; FFN predicts expression directly |
| Checkpoint path | Find latest `.ckpt` file in logs directory after training | `train_inr()` returns None (no explicit return); need to glob for checkpoint |
| Decoder dependency | Set `decoder=None` in both training and prediction configs | No GAE decoder needed for raw expression prediction |
| GPU selection | Changed `CUDA_VISIBLE_DEVICES` from `0,1` to `0` | Other GPUs may be occupied by other users |

**Status**: INR training succeeds on GPU (RTX 3090, 150 epochs). Full benchmark 6/6 rounds complete. PCC=0.515-0.628. Known limitation: n_pred=5488 (subset of holdout points).

---

# Scientific-correctness audit (2026-08-13)

## isoST — third-party modifications REVERTED

| Change | Action |
|--------|--------|
| `methods/isoST/model/utils/losses.py` softmin `one_nn_mse` | **REVERTED to upstream** hard-argmin version (git checkout). Softmin was a core algorithm change (loss math), not a compatibility fix. |
| `methods/isoST/model/base_model.py` tensor-init / isinstance guards | **REVERTED to upstream**. Official code trains on GPU as-is (verified: gradients reach all params). The earlier "argmin breaks grad" diagnosis was wrong — the real crash came from passing `[slices]` (batch format) to `train_step`. |
| `methods/isoST/model/utils/losses.py` SinkhornLoss float wrap | **REVERTED to upstream**. geomloss 0.2.6 returns a differentiable tensor on GPU. |

Pre-revert files + diff archived at
`benchmark/results/large_scale_benchmark/AUDIT_2026-08-13/evidence/`.
Current `methods/isoST` diff vs upstream: **none**.

## UniST adapter fixes (benchmark/src/adapters/unist_adapter.py)

| Change | Reason |
|--------|--------|
| `pred_config.custom_coords_file` → set under `pipeline` (`pred_config.pipeline.custom_coords_file`) | SUICA `predict_inr()` reads `pipeline.custom_coords_file`; setting it at config root silently fell back to the official demo coords file (5488 points) — the origin of the "5488 predictions" |
| `pred_config.pipeline.predict_mode = "custom"` (explicit) | Default yaml already "custom", set explicitly for clarity |
| Query coords kept RAW (removed [-1,1] normalization) | INR trains on raw coords (`require_coordnorm: False`) |
| Adapter asserts output `obsm['spatial']` == query coords (atol 1e-4) | Proves query→prediction correspondence per round |
| GPU auto-selection (least-loaded) via config `gpu` | GPU 1 hardcode removed |

## isoST adapter alignment (benchmark/src/adapters/isost_adapter.py)

| Change | Reason |
|--------|--------|
| Hyperparams aligned to official `config/mouse_brain.yml` (alpha=0.1, warm_up_rate=1.0, std_x/y=0.01, std_z=0.1, beta 1→0.05/50/50, weight_decay=1e-8, lr=1e-3 NAdam) | Previous values were undocumented deviations |
| Official 3-stage schedule `epochs_list=[100,100,100]` (coo→seq→joint) | Official training protocol |
| Feature space: dataset 50 PCs used directly (no re-PCA) | Dataset is already PC50+minmax (isoST official recipe); re-PCA was unnecessary |
| Inference: one virtual slice per held-out z (official fine_infer protocol) | Previous adapter predicted only the gap-mean plane — unfair coverage for gap-2/3 |
| Frozen NN mapping of generated points to public query (x,y) | fixed_coordinate protocol; match distances recorded |
| Resource reductions (permitted): hidden_dim 64→32, 1000 pts/slice | Registered in metrics.json extra |

## SpatialZ adapter fix (benchmark/src/adapters/spatialz_adapter_v2.py)

| Change | Reason |
|--------|--------|
| `lr=1e5` (default) → `lr=1.0`, `nb_iter_max=1000` (official notebook value) | Default lr drifted generated coordinates out of the tissue (x∈[-4.3,0.3] vs input x∈[0.3,5.0]); lr=1.0 verified stable. Numerical hyperparameter, not an algorithm change. |
| Runner: frozen NN mapping of generated (x,y) → query (x,y) | Old summary paired generated cells to GT by arbitrary row order |

## PASTE-OT-Linear frozen OT implementation

| Item | Value |
|------|-------|
| OT | POT `entropic_fused_gromov_wasserstein` (PASTE2-style) in spateo env: KL expression dissimilarity, square_loss, α=0.1, ε=0.05, cost matrices max-normalized, marginal error ~1e-18 |
| Resource constraint | 3000 fixed-seed (42) anchor spots per slice; OT math unchanged |
| Blend | X_blend = w_lower·(Piᵀ@X_lower)·n_lower + w_upper·X_upper on the upper support, weights = z-distance |
| Projection | Frozen NN from upper support (x,y) to public query (x,y) |
| Previous implementation | Deprecated: spateo import failed → silent non-OT mean fallback (self-labeled "NOT a valid baseline") + arbitrary row-order pairing |

## ReHo

Frozen NN mapping of source-slice expression to query (x,y) support (previous: arbitrary row-order pairing). Internal sanity baseline only.

## Evaluation engine

`compute_v11_final.py`'s row-order pairing (`X_gt[:n]` vs `X_pred[:n]`) replaced by
`benchmark/scripts/_audit_common.pair_by_coords` — predictions are paired to GT
by exact coordinate match (or the documented frozen NN mapping), and the
correspondence is asserted before metrics are computed.

---

# Full-official-config compliance (2026-08-14, user directive)

## isoST — Sinkhorn evaluation backend (minimal adaptation, math-identical)

| Change | Detail |
|--------|--------|
| `methods/isoST/model/utils/losses.py:SinkhornLoss` backend `"tensorized"` → `"online"` | Streaming kernel evaluation via pykeops; **math identical** (same Sinkhorn divergence, same inputs/outputs). Verified on identical inputs: \|Δloss\|=2.8e-9, \|Δgrad\|=9.2e-9. Reason: official full-slice config (up to 36k pts/slice) needs ~19-30 GiB with tensorized (OOM on 24 GB RTX 3090); online backend uses ~0 GiB. |
| isoST env: geomloss 0.2.6→0.3.1, pykeops 2.2.3, nvidia-cuda-nvcc/nvrtc/runtime pip pkgs | Environment upgrades for the online backend |
| Project CUDA toolchain `/mnt/data1/linpeiran/005/envs/cuda-nvcc-11.8.89` (nvcc + headers) | keops JIT compiler needs nvcc + cuda.h/nvrtc.h; assembled from pip/conda caches, injected via `_isost_subprocess_env()` (CUDA_HOME/CUDA_PATH/PATH) |

Official-loss verification re-run with online backend: all checks pass
(gradients reach all params; loss value matches tensorized to 1e-7).

## isoST — full official config

| Param | Value |
|---|---|
| hidden_dim | 64 (official config/mouse_brain.yml) |
| points/slice | FULL slices (no subsampling; previous 1000-pt runs NON-COMPLIANT) |
| training | official 3-stage [100,100,100] |
| measured | 67.5 s/step at full slices (GPU 2, sole tenant) |

T_max per round = setup 600 s + 300×67.5×1.5 + inference/metrics 600 s ≈ **9.0 h**
(written to run manifest; round killed if exceeded).

## UniST/SUICA — GAE per-batch cleanup removal (computation-organization only)

| Change | Detail |
|--------|--------|
| `methods/UniST/external/SUICA_pro/systems/embedder_fitting_system.py:training_step` | Removed per-batch `del adj; del raw_rep; gc.collect()` (local no-ops — the dict retains the arrays) and `torch.cuda.empty_cache()` (pure memory-management call). Loss math and gradients are identical; ~3x faster per batch at the official full-slice scale (1M spots). |
| DataLoader num_workers | Official num_workers=18 verified working in the unist env (previous num_workers=0 patch dropped). |
| isoST env additions | geomloss 0.3.1, pykeops 2.2.3, nvidia cuda pip pkgs (see isoST entry) |

UniST full-pipeline measured costs (A1, GPU 1 sole-tenant where applicable):
- anisotropic KNN (1M visible spots, CPU): ~20 min/round
- GAE 1000 epochs x ~2000 batches: measured 2-epoch cost ~570 s (with cleanup) —
  re-measured after cleanup removal (see T_MAX.txt per run)
- INR 2000 epochs x ~200 batches + decoder

---

# GPU-acceleration campaign (2026-08-20, user goal: GPU-optimize all methods, no algorithm change, everything registered)

## SpatialZ — device=auto wired through (official GPU config; adapter fix)

| Change | Detail |
|--------|--------|
| `benchmark/src/adapters/spatialz_adapter_v2.py` | The 2026-08-14 "device=auto" commit only READ `config.get("device")` but the generated inner script hard-coded `device='cpu'` (dead variable) — every formal round would have run CPU-only. Fixed: `device='{device}'` is now interpolated into the inner script (default `auto` = official GPU path, same as the official notebooks' `device='cuda:*'`); added optional `config["gpu"]` → `CUDA_VISIBLE_DEVICES` pinning (same convention as the isoST/UniST adapters); subprocess env now passes the pin. No third-party code touched; `lr=1.0` / `nb_iter_max=1000` / `num_projections=80` audit fixes unchanged. Small-data smoke on GPU passed (300 cells, 49 s; GPU selection verified by code inspection of `Generate_spatialz` line `device=='auto' -> cuda:0`). |

## isoST — batched euler + discrete adjoint over slice pairs (math-identical, 2.30x step speedup)

| Change | Detail |
|--------|--------|
| `methods/isoST/model/utils/batched_ode.py` (new) | `BatchedEulerAdjoint` autograd.Function: all slice-pair euler trajectories of one training step integrated as one variable-size batch (torch_geometric batch semantics), with the exact torchdiffeq discrete adjoint (per-interval augmented-dynamics euler step; vjp via `autograd.grad` with `grad_outputs=-dt*adj`). No padding rows (variable-size graphs), so per-pair KNN/message-passing/pooling see exactly their own points. |
| `methods/isoST/model/utils/modules_ODE.py` | `PCooStep.forward` / `PUnetStep.forward`: optional `noise=` arg (pre-drawn raw SDE noise injection; `None` = upstream draw). `PUnetStep.forward`/`GraphUNet`: optional `batch=` arg (per-graph pooling). `PJointV2.forward_batched`: same p_coo/p_seq modules with caller-computed KNN + batch vector. All optional args default to upstream behavior. |
| `methods/isoST/model/base_model.py` | `USE_BATCHED_ODE=True` (module flag); `_compute_total_loss` dispatches to `_compute_total_loss_batched` when `stride==1` (the official config), upstream loop otherwise. Loss assembly (Sinkhorn/one_nn_mse per pair) unchanged. |

**Verification (2026-08-20, 8-slice smoke data, GPU 2):**
- A/B test (same seeds, 3 joint steps): global RNG stream bit-identical after both runs (noise draw order replicated exactly); loss values agree to ≤2.4e-7 (float32 accumulation-order noise from batched vs per-pair GEMM kernels — same class as cross-GPU-model variation; user ruling 2026-08-20: not precision loss); gradients agree to 1e-9..1e-6; no math, objective, dt, edge set, noise stream, or RNG change.
- Timing: 2.88 s/step (upstream loop) → 1.25 s/step (batched) = **2.30x** on the 8-slice smoke. The formal 9h/round T_max is unchanged (safety cap).

## UniST/SUICA — GPU subgraph construction (bit-identical, ~2.7x on the per-batch step)

| Change | Detail |
|--------|--------|
| `methods/UniST/external/SUICA_pro/utils.py` | New `construct_subgraph_cuda`: GPU twin of `construct_subgraph` — same sub_set_data values (GPU gather), same dense float32 sub-adjacency (scatter of 1.0s at the graph's edges, `accumulate=True` mirrors scipy's duplicate-summing), same `new_idx` list (identical Python dict/list code). |
| `methods/UniST/external/SUICA_pro/systems/embedder_fitting_system.py` | Data prep: builds CUDA copies of `raw_rep_train/val` and COO edge lists (`adj_*_src/dst`) of the full KNN graphs once; `training_step`/`validation_step` use the CUDA path when present (upstream path kept as fallback). The GAE/INR model, loss, adj values (connectivity 1.0s, float32), and neighbour ordering are untouched. |

**Verification (2026-08-20, synthetic 30k-node GraphST3D-shaped data, GPU 2):**
- A/B: 20/20 random batches bit-identical (sub_set_data, sub_set_adj, sub_set_idx).
- Timing: 10.2 ms/batch (scipy slice + toarray) → 3.72 ms/batch (GPU) = **2.7x** on the per-batch subgraph step; at the official 1M-spot scale this step also carries ~100MB CPU→GPU transfers per batch, so the end-to-end GAE gain is expected larger.

## Remaining methods — GPU-optimization assessment (2026-08-20)

| Method | Assessment | Action |
|--------|-----------|--------|
| SINTER3D | Model path already torch-GPU end-to-end (300-step smoke: 113 s on GPU 2, PCC 0.58); `utils.preprocess` is one-time CPU numpy; no CPU-bound loop remains | None needed (already GPU) |
| PASTE-OT-Linear | The frozen EFGW OT step runs in POT on CPU (~1 h/round at full-slice scale). The only GPU route is POT's torch backend, which changes the solver's dtype/implementation numerics — rejected under the no-precision-change rule; a bit-exact GPU rewrite of POT's solver is out of scope. Formal 6/6 runs already complete | None (documented) |
| ReHo | Pure numpy NN baseline, <1 s/round | None needed |
| Spateo (native-data) | Completed native-data alignment run used the official CPU pipeline; formal work complete | None (documented) |
| Spa3D (native-data) | GCN model is torch-GPU already; task is not_applicable for MERFISH expression | None needed |
| ASIGN (multimodal) | Official pipeline is torch-GPU (2.1.2+cu118, torch_geometric); task not_applicable (no H&E data) | None needed |

## 2026-08-20 new-machine campaign — runner parameterization (math-neutral)

| Date | Method | File(s) | Change | Reason | Upstream Commit |
|------|--------|---------|--------|--------|-----------------|
| 2026-08-20 | (all Common-task) | benchmark/scripts/_audit_common.py, rerun_spatialz_fixed.py, rerun_isost_fixed.py, rerun_unist_official.py, rerun_baselines_fixed.py, aggregate_fixed_results.py, post_campaign_finalize.py | Added `--results-dir` parameter (default: new campaign tree `large_scale_benchmark_rerun_20260820`); `save_round(results_root=...)`; old-evidence AUDIT_INVALID markers now written only if absent; per-run T_MAX.txt written pre-launch; PASTE OT subprocess timeout 54000s → 14400s (documented T_max basis: 80 min measured x 1.5 x 2 new-CPU factor) | New-machine full re-run campaign must never write into the pre-migration evidence trees (execution-safety rule); T_max discipline per resource policy §5 | — |
| 2026-08-20 | SINTER3D | benchmark/src/adapters/sinter3d_adapter.py, benchmark/scripts/rerun_sinter3d_campaign.py (new) | `timeout_sec` run-config key replaces hardcoded 2 h subprocess timeout (default unchanged); new campaign runner executes the official model_spatial + inference_latent path with the OFFICIAL training config: training_steps=10000 (official Drosophila_new.json / mouse_brain_new.json analog; lr=0.001, hidden_dims=[512,128], patience=200, mid_channel=200 already matched), full slices (no subsampling), probe-based T_max | Pre-migration predictions used training_steps=1000 — a resource shrink vs official configs (resource policy §1). Math of the model untouched | — |

## 2026-08-20 campaign — GPU memory organization at full 54-slice scale (math-neutral)

Both fixes are memory-organization only: no data/config/model-size change, no
objective change. Peak 24 GB exceeded only because the formal 1M-spot scale is
far larger than the official datasets (DLPFC/mouse-brain/embryo) and the
pre-migration smoke (5-8 slices).

| Date | Method | File(s) | Change | Reason | Upstream Commit |
|------|--------|---------|--------|--------|-----------------|
| 2026-08-20 | isoST | methods/isoST/model/utils/batched_ode.py | Per-training-step pair batch split into consecutive pair chunks; formal campaign default = ONE PAIR PER CHUNK (PAIR_CHUNKS=1, ISOST_CHUNK_POINTS/ISOST_PAIR_CHUNKS env overrides for A/B only). Pairs share no edges/tensors (per-pair KNN via batch vector), noise pre-drawn in upstream order and only indexed per pair. A/B (same seeds): multi-pair 120k chunks diverge ~0.7 rel by 9 steps (larger per-step GEMM/KNN-grouping noise amplified by hard-KNN SDE dynamics); per-pair chunks reproduce un-batched per-pair GEMM shapes and stay at the float32-noise floor — 3-epoch smoke generated-point values agree ~1e-4 rel vs single-batch (residual = amplification of ~1e-7 step noise; same noise class as the registered 2026-08-20 batching noise, user ruling: not precision loss). Param grads accumulate across chunks in deterministic order | Full 54-slice step (53 pairs, 1M spots) OOMs a 24 GB 3090 in the adjoint backward (17.85 GiB peak) | — |
| 2026-08-20 | SINTER3D | methods/SINTER3D/SINTER3D-master/SINTER3D/networks_spatial.py | Progressive-encoder layers wrapped in torch.utils.checkpoint (env SINTER3D_NO_CHECKPOINT=1 restores plain loop for A/B only). Deterministic recompute; CUDA RNG state is snapshotted/restored around the recompute so the ResidualSineBlock dropout masks reproduce exactly — math-identical, only peak-memory/compute trade-off changes | Encoder activations at 1M spots exceed 24 GB (CUDA OOM at 20.77 GiB allocated, backward) | — |

## 2026-08-20 campaign — ASIGN dataloader prefetch workers (computation organization)

| Date | Method | File(s) | Change | Reason | Upstream Commit |
|------|--------|---------|--------|--------|-----------------|
| 2026-08-20 | ASIGN | benchmark/scripts/rerun_asign_campaign.py (generated training script) | DataLoader num_workers 0 -> 8 via an init monkey-patch. DataLoader sampling (shuffle permutation, batch composition, RNG draws) happens in the MAIN process regardless of workers — the model sees identical batches in identical order; only item loading (PIL decode + transforms, ~150ms/item measured on this shared machine) is prefetched in parallel | Official num_workers=0 made per-batch ~25s and the 51-epoch official run infeasible on the shared machine (>> T_max); pure computation reorganization, seeds/inputs/outputs unchanged | — |

## 2026-08-21 campaign — UniST GAE sparse-edge graph convolution (math-neutral, ~1.9x GAE step)

| Date | Method | File(s) | Change | Reason | Upstream Commit |
|------|--------|---------|--------|--------|-----------------|
| 2026-08-21 | UniST | methods/UniST/external/SUICA_pro/utils.py, networks/gae.py, systems/embedder_fitting_system.py | New `construct_subgraph_cuda_edges` returns the in-batch COO edge lists (src_local, dst_local) instead of the dense [n,n] sub-adjacency; `content_graph_conv.forward` accepts the edge tuple and computes adj @ x via `y[src] += x[dst]` (scatter_add) instead of `torch.mm(dense_adj, x)`. Enabled by env `UNIST_SPARSE_GCN=1` in the campaign driver; dense path unchanged and kept as default/fallback. The set-union neighbourhood ordering, log1p, BN/ReLU, encoder/decoder, loss and Adam all untouched — only the adj-multiplication's summation order changes (registered float32 noise class, cf. isoST 2026-08-20 ruling) | Measured at the official 1M-spot scale (2026-08-21, probe on idle GPU): dense step = 15.49 ms (subadj build 3.1 + dense mm fwd/bwd 7.4 + model rest), sparse step = 8.05 ms; epoch wall 47.8 s -> ~25-26 s projected; GPU was 22-34% SM-busy while the CPU/Lightning orchestration dominated the other half — the dense [n,n] construction + matmul is the removable GPU cost, NOT data transfer (the 50-gene batches are ~100 KB; expression matrices are pre-resident on GPU since 2026-08-20) | — |

**Verification (2026-08-21):**
- A/B on the REAL anisotropic-KNN graph of A1's visible data (official call: n_neighbors=6, z_weight=2.0, include_self=True, N=1,023,466, E=7,136,146, built in 11s): 3 real batches (n=3594/3576/3566) — edge sets **identical** (dense nnz pattern == sparse edge list, 17,978/17,754/17,838 edges), GCN output max |dense−sparse| = **4.77e-07** (rel 1.5e-07, float32 summation-order noise), forward_loss rel diff = **0.0e+00 at float32** on all 3.
- Timing probe (fabricated closed-neighbourhood graph, official scale): V1 (official step, python+subadj+fwd+bwd+Adam) 15.49 ms vs V2 (sparse) 8.05 ms = **1.92x**; python set/dict orchestration measured 1.5 ms/step (15.49 vs 13.96 without). GPU 1 was 22-34% SM-busy during A1 (dmon 2-min sample) with epoch-boundary 0% dips; epoch wall 47.8 s at 1600 steps/epoch.
- Scope: applies from UniST A2 onward (A1 ran the dense path, already registered 2026-08-20); per-round code variant recorded in run logs AND in metrics.json as `gcn_variant` (`sparse_edge`/`dense`) — added to `rerun_unist_official.py` extras on 2026-08-21 so A2+ record it at save time; A1's metrics.json gets a provenance-only patch (`gcn_variant: "dense"`, no metric values touched) after it completes, since the running A1 process predates the field.

## 2026-08-21 campaign — UniST INR Lightning-checkpoint resume (math-neutral recovery)

| Date | Method | File(s) | Change | Reason |
|------|--------|---------|--------|--------|
| 2026-08-21 | UniST | methods/UniST/external/SUICA_pro/systems/inr_fitting_system.py; benchmark/scripts/rerun_unist_inr_resume.py (new) | `trainer.fit` accepts optional `pipeline.optimization.resume_from_checkpoint`, passed through as `ckpt_path=`; config key absent -> original behavior byte-for-byte. New single-shot runner resumes a killed round's INR stage from its official Lightning `last.ckpt` (full optimizer + RNG state), then executes stage 3 (predict_inr at query coords) verbatim from the official template; per-run resume T_max written to run/T_MAX.txt | 2026-08-21 22:58 the A2 subprocess was killed by the pre-calibration timeout (47279 s frozen at launch) while healthy at INR epoch 798. Resume is a pure computation reorganization (resource policy §3): same env, same seed 8848, optimizer + torch/numpy RNG states restored from the checkpoint -> exact continuation of the official trajectory; total official steps unchanged (GAE 1000 + INR 2000 + decode). Verified pre-flight: Lightning 2.6.5 `pl_load` defaults `weights_only=False` (source-verified), so no new load path; checkpoint self-produced in the same env. Resume metadata (`resumed`, `resume_from_epoch`, `resume_reason`) recorded in metrics.json; full incident in SCHEDULE.md 2026-08-21 22:58 | — |

## 2026-08-23 campaign — UniST B2 duplicate-round incident: GAE evidence reconstruction + stage-3 rescue (math-neutral recovery)

| Date | Method | File(s) | Change | Reason |
|------|--------|---------|--------|--------|
| 2026-08-23 | UniST | benchmark/scripts/rerun_unist_gae_recon.py, benchmark/scripts/rerun_unist_stage3.py (new) | Two recovery runners, no algorithm change. (1) GAE-only reconstruction: stage 1 of the official template verbatim (embedder_gae_3d_sparse.yaml, visible data, 1000 epochs, official seed); refuses to overwrite existing embedded-all.h5ad; preserves the duplicate's partial artifacts by rename (logs_gae -> logs_gae_dup_partial_20260823). (2) Stage-3-only rescue: stage 3 verbatim (coords normalization + predict_inr at query coords + X_pred verification) using the reconstructed GAE checkpoint + the official INR's completed `last.ckpt`; save_round records `resumed=True`, `resume_from_epoch`, `resume_reason`, `gcn_variant=sparse_edge` | 2026-08-23 13:45 the GPU0 chain's skip-check missed the in-progress official B2 on GPU1 and launched a duplicate; the official template's stage-1 `rmtree(logs_gae)` then deleted GPU1's completed GAE evidence. The GAE is deterministic under the official seed (cross-GPU verified: A2 resume continuity, byte-identical ckpt content) — reconstruction reproduces the evidence at float-noise level; the INR ran the full official 2000 epochs on GPU1 and its checkpoint is intact. Total official steps unchanged (GAE 1000 + INR 2000 + decode); pure computation reorganization per resource policy §3. Full incident: SCHEDULE.md 2026-08-23 13:45–17:38 | — |

## 2026-08-29 — machine migration: NVIDIA 3090 → 2× RTX 5090 (Blackwell), paths & envs (math-neutral)

| Date | Method | File(s) | Change | Reason | Upstream Commit |
|------|--------|---------|--------|--------|-----------------|
| 2026-08-29 | (all) | methods/* (9 symlinks) | Re-pointed broken symlinks to new data disk `/data/coding/xiepeiyan/data/{data,results,spa3d_data}` equivalents (all verified OK) | Old-machine `/mnt` paths absent on new machine | — |
| 2026-08-29 | (all) | 41 files under benchmark/scripts/, benchmark/src/, benchmark/configs/tasks/ | Hardcoded `/mnt/nfs/linpeiran/data/AIgnite/005/baseline` → `/data/coding/xiepeiyan/lab/baseline`; `/mnt/data1/linpeiran/005/` → `/data/coding/xiepeiyan/data/`. Pure path logistics — task yaml scientific content (holdout slices, gene set, protocol) unchanged | Migration | — |
| 2026-08-29 | (all) | benchmark/scripts/_audit_common.py | `REPO` now derived from `__file__` (`Path(__file__).resolve().parents[2]`); `DATASET` → `/data/coding/xiepeiyan/data/benchmark/datasets/MERFISH_ABCA-2_1_of_1_standardized.h5ad` (verified present, 484 MB) | Migration | — |
| 2026-08-29 | SpatialZ/SINTER3D/Spa3D | benchmark/src/adapters/spatialz_adapter.py, sinter3d_adapter.py, benchmark/scripts/rerun_spa3d_campaign.py | Env python paths → `/data/coding/xiepeiyan/.conda/envs/{spatialz,SINTER3D_env,spa3d}/bin/python` (envs to be rebuilt on Blackwell) | Migration | — |
| 2026-08-29 | ASIGN | benchmark/scripts/rerun_asign_campaign.py | subprocess cwd → `Path(tempfile.gettempdir())` (was an old-machine Claude job tmp dir; the local-disk-cwd requirement for dataloader speed is moot — new machine is all-local) | Migration | — |
| 2026-08-29 | (all) | benchmark/envs/benchmark.yaml, benchmark conda env | Added persim>=0.3 + ripser (pip) — present on old machine but unrecorded; without them 2 topology tests fail. pytest 183/183 green on new machine | Env parity | — |

**NOT changed**: third-party algorithm code under `methods/` (only symlink targets);
frozen task contracts' scientific content (only dataset paths); executed notebooks
and old evidence trees (`large_scale_benchmark*`, `_fixed`) remain read-only.

**Pending**: rebuild of per-method conda envs (unist / isoST / spatialz / SINTER3D_env /
spa3d / asign) for Blackwell sm_120 — old envs were built for 3090 / CUDA 11.8 and are
not present on the new machine. Compatibility matrix under study; any upgrade is a
math-neutral compatibility fix and will be registered here before formal use.

## 2026-08-29 — Blackwell (RTX 5090, sm_120) env rebuild & validation (math-neutral compatibility)

| Date | Method | File(s) | Change | Reason | Upstream Commit |
|------|--------|---------|--------|--------|-----------------|
| 2026-08-29 | (all) | /data/coding/xiepeiyan/.conda/envs/{SINTER3D_env,spa3d,spatialz,unist,asign,isoST} (new); benchmark/envs/*.yaml (new) | Rebuilt all method conda envs for sm_120: SINTER3D_env/spa3d/spatialz/unist/asign = py3.10 + torch 2.13.0+cu130; isoST = py3.10 + torch 2.9.0+cu128 + PyG 2.7.0 + torch_cluster/scatter/sparse (data.pyg.org wheels) + geomloss 0.3.1 + pykeops 2.2.3 + POT 0.9.5 + torchdiffeq 0.2.4. Old envs were 3090/CUDA 11.x-era (torch 1.10/1.12/2.1/2.5) and have NO sm_120 kernels (torch >=2.7 required; verified: "no kernel image" class failure). Env fixes along the way: spatialz scipy==1.10.1 (POT 0.9.3 armijo API), squidpy==1.2.3, setuptools<70, SOMENDER==1.1; SINTER3D_env +scanpy/pot/joblib; unist +tensorboard; asign +tensorboardX; isoST +pyyaml | Hardware migration: old envs unusable on Blackwell; algorithm code untouched | — |
| 2026-08-29 | isoST | benchmark/src/adapters/isost_adapter.py | ISOST_CUDA_HOME → /usr/local/cuda (CUDA 12.8); + LIBRARY_PATH/LD_LIBRARY_PATH=$CUDA_HOME/lib64 in _isost_subprocess_env (keops 2.2.3 links `-lnvrtc` without -L) | Old nvcc-11.8 toolchain cannot compile sm_120; system CUDA 12.8 headers verified complete (cuda.h + crt/host_defines.h); keops JIT link fix is env-only | — |
| 2026-08-29 | (all) | /data/miniconda/.condarc (envs_dirs += /data/coding/xiepeiyan/.conda/envs); /data/coding/xiepeiyan/.conda/environments.txt | Registered new env dir so `conda run -n <env>` resolves (adapters' invocation contract) | New machine layout | — |

**Validation evidence (2026-08-29, GPU 0, idle-card policy):**
- SINTER3D: official 300-step probe at full 1M-spot scale, 0.323 s/step (old 3090: 0.57) = **1.77×**; PCC-level sanity fine.
- UniST: official pipeline smoke (GAE 2ep + INR 2ep + decoder + custom-query predict) end-to-end PASS (Lightning 2.6.5 × torch 2.13 OK).
- isoST: keops 2.2.3 JIT compiled and ran on sm_120 with nvcc 12.8 (Sinkhorn online backend, fwd+bwd PASS); end-to-end 3×3-epoch smoke at full official scale — see run log.
- spa3d/spatialz/asign: import + forward validation PASS (incl. MENDER, ST_GTC, simple_GC_DEC).
- Numerical note: torch version/hardware migration changes kernel selection and float32 accumulation order (PyTorch does not guarantee cross-version bitwise identity; noise class precedented by 2026-08-20/21 rulings). All future formal runs will register hardware (run manifest `gpu`) and this env section.

## 2026-08-29 — Blackwell validation: cross-hardware A/B for isoST (accidental full official run)

| Date | Method | File(s) | Change | Reason |
|------|--------|---------|--------|--------|
| 2026-08-29 | isoST | benchmark/scripts/rerun_isost_fixed.py | Fixed `--smoke` argparse trap (`--epochs` had a default so `not args.epochs` was always False — smoke silently ran the FULL official 300-step training). Now `--epochs` default None; smoke = [3,3,3] only when not given | Runner bug (our code, not method code) |

**Evidence**: the accidental full run (probe_5090_20260829/A1, official config,
300 steps, 1.94 h on 5090 vs ~6 h on 3090 = **3.1×**) reproduced the campaign
A1 metrics (pcc 0.3241→0.3242, rmse 0.1712→0.1711, moran_delta 0.169→0.170)
— cross-hardware float-noise-level agreement, same registered noise class as
2026-08-20/21 rulings. Kept as validation evidence (not a formal result).

## 2026-08-30 campaign — UniST GAE all-GPU subgraph build (math-neutral, construction 2.75×)

| Date | Method | File(s) | Change | Reason | Upstream Commit |
|------|--------|---------|--------|--------|-----------------|
| 2026-08-30 | UniST | methods/UniST/external/SUICA_pro/utils.py (new `construct_subgraph_cuda_fast`), systems/embedder_fitting_system.py (training/validation fast branches + `e_row_start_train/val` indptr + reused `local_map_buf_train/val` precompute), benchmark/scripts/rerun_unist_official.py (FAST_SUBGRAPH marker + `gcn_variant` recording) | Per-step subgraph build moved fully onto GPU: `torch.unique` sorted dedup replaces the Python set/dict union (removes 2 per-step D2H syncs), candidate edges come from the subgraph nodes' CSR row blocks (~21k) instead of scanning all ~7.16M COO edges, the full-size local map is a pre-allocated reused buffer, and no per-step `.item()`/metadata syncs remain. Enabled by env `UNIST_FAST_SUBGRAPH=1` (campaign marker `campaign_runs/unist_gpu1/FAST_SUBGRAPH`); dense and sparse paths unchanged and kept as fallbacks | User request "GPU0 只有很小负载，能不能提提速" (2026-08-30). Measured (probe, GPU 0): the official GAE step (11.3 ms) is **latency-chain-bound** — subgraph construction 2.08 ms/step (dense: 51 MB [n,n] alloc + full-COO filter over 7.16M edges + Python set/dict over ~4k nodes + 2 D2H syncs) vs **0.76 ms all-GPU fast** (2.75×), but the official GAE forward+backward (~9.3 ms/step) dominates, so the total step gain is ~1.3 ms/step ≈ 12% (epoch 18.1 → 16.3 s). Earlier "~80% of 14.3 ms is orchestration" diagnosis was an overestimate (construction is ~18% of the step; the rest is official GPU math). Pure computation reorganization — same induced subgraph (same node set, edge multiset; per-source edge order preserved vs the CSR ground truth); node ORDER differs (sorted vs set-hash) = graph isomorphism, content_graph_conv scatter_add + BatchNorm1d are permutation-equivariant and the loss reads only batch-center rows (registered float32 noise class, cf. 2026-08-21 ruling) | — |

**Verification (2026-08-30, CPU — GPU 0 busy with the official probe):**
- Subgraph equivalence, 30/30 random batches on a 30k-node k=6 GraphST3D-shaped graph: node set, center mapping (`idx_local`), edge multiset + per-source edge order all identical to `construct_subgraph_cuda_edges` AND to the CSR-derived ground truth (multiset check alone would miss order; sequence compared elementwise). The test caught and fixed a real bug (CSR row-block intra-block offset must restart per row — a global arange offset overshoots E).
- Full official GAE forward (dim_in=50, dim_hidden=[2048,512,128], dim_latent=64), 20/20 batches with non-negative count-like data: `forward_loss` rel diff **0.0e+00** (bitwise identical at float32, stronger than the ~1e-7 registered noise class).

**Verification (2026-08-31, GPU 0 — official A1 data, official config; evidence `ab_fast_subgraph_20260830/evidence/`):**
- Same-seed smoke A/B (runner `--smoke` protocol, dense vs fast, official anisotropic kNN k=6 on the full 1,023,466-point graph): GAE epoch 18.1 s → **16.3 s**; end-to-end 227 → 222 s (fixed costs dominate); loss epoch-0 means identical at 1e-6, epoch-1 mean diff 2.98e-4 (see divergence characterization below); final metrics pcc_global 0.251754 vs 0.251020 (Δ7.3e-4), rmse 0.301070 vs 0.300842 (Δ2.3e-4) — registered noise class.
- Per-step probe (1600 official steps, both paths on identical fresh GAE/Adam, same fixed batches): subgraph outputs **identical every step** (node set / center map / edge multiset OK/OK/OK); losses **bitwise identical** for early steps; per-step |loss diff| ≤1.7e-6 over the first 100 steps, mostly ≤1e-5 over 1600 with **rare single-step spikes to ~4e-3 that self-recover** (step 699: 1.9e-3, step 799 back to 7e-8 — per-batch rounding sensitivity, not a structural difference; a dropped-edge bug would persist). Epoch-mean loss diff 1.99e-5 (same phenomenon as the smoke's 2.98e-4 — batch-content-dependent). Construction timing: dense mean 2.08 ms vs fast 0.76 ms/step (**2.75×**). All within the registered float32 accumulation-order noise class; no divergence growth beyond the noise floor.
- `benchmark/tests` pytest: 183/183 pass; both edited modules import cleanly in the `unist` env.
- **Verdict**: math-neutral; the fast path is enabled by env var and OFF by default — formal runs keep the registered dense path unless the user opts in.

## 2026-09-01 stress campaign (SPARSE_54 / STRESS_54) — PASTE engineering & environment (math-neutral)

| Date | Method | File(s) | Change | Reason |
|------|--------|---------|--------|--------|
| 2026-09-01 | PASTE-OT | benchmark/scripts/rerun_baselines_stress.py | OT subprocess env: `spateo` → `benchmark` (POT 0.9.7.post1). The 2026-08-20 campaign's `spateo` env was deleted; the frozen OT script imports ONLY POT + numpy (never spateo), so the math is unchanged. Reused campaign couplings are loaded from files (no computation). Registered: environment-only change | spateo conda env no longer exists (conda env list, 2026-08-31) |
| 2026-09-01 | PASTE-OT | benchmark/scripts/rerun_baselines_stress.py | Coupling reuse cache rewritten: campaign metrics.json carries no `extra.ot_pair` (fields added later), so pairs are now identified by matching each round's `run/ot_lower.npz` / `ot_upper.npz` point sets against per-slice coordinates (full slices, float32 round-trip identical) | cache previously always missed → fresh full-slice OT per round (would have been ~2.9 h/pair) |
| 2026-09-01 | PASTE-OT | benchmark/scripts/rerun_baselines_stress.py | `OT_ANCHORS` = None (was 3000 in draft runner): official campaign config is FULL slices (rerun_baselines_fixed.py 2026-08-14 ruling) — anchor subsampling would shrink OT data volume and change the optimization problem; forbidden by experiment-resources policy | A/B test `test_reuse_reproduces_campaign_a1_pcc` caught the dimension mismatch (3000×3000 pi vs full-slice X) |
| 2026-09-01 | all stress runners | benchmark/env (psutil) | Added psutil dependency to the `benchmark` conda env for `ResourceSampler` (P7 resource sampling, 10 s nvidia-smi + process tree) | ResourceSampler thread crashed with ModuleNotFoundError in pytest |
| 2026-09-01 | PASTE-OT | benchmark/scripts/rerun_baselines_stress.py | **Race-condition fix**: parallel OT workers (ProcessPoolExecutor, max_workers=6) previously wrote the SAME `ot_lower.npz` / `ot_upper.npz` paths; a worker's conda subprocess then read whatever the last writer left — silently cross-contaminated couplings (first T3_A1_s2 attempt: 5/6 pairs failed OT or saved another pair's shape; identical rowerrs across different pairs in the log are the fingerprint). Fix: per-pair paths `ot_lower_{sl}_{up}.npz` / `ot_upper_{sl}_{up}.npz` + shape assertions in the subprocess; regression `TestParallelPairIsolation` (parallel == serial ≤1e-6, exact shapes, per-pair files). Pure computation organization — coupling math (POT PGD solver, efgw params, reuse logic) unchanged | first parallel attempt (2026-08-31, pre-fix) failed with `RuntimeError: OT failed` for 5 pairs (contaminated shared npz); fixed before the production rerun (per-pair files verified in production, 2026-09-01) |
| 2026-09-01 | ReHo, SINTER3D | rerun_baselines_stress.py, rerun_sinter3d_stress.py | `--tag` CLI arg: sensitivity runs (ReHo k=5/10, SINTER3D steps 5000/20000) write to a SUFFIXED round dir (`T3_A1_s2_k5`, `T3_A1_s2_steps5000`) and registry run_id — without it, `save_round`'s round-name-keyed path would silently overwrite the official T3_A1_s2 metrics.json/pred.h5ad. Output-dir/registry bookkeeping only; no algorithm or parameter-default change | P7 sensitivity plan (PLAN.md §11.3.1); execution-safety rule forbids silent overwrites |

**Verification (2026-09-01, benchmark env, GPU):**
- `test_paste_ab.py` 4/4 pass: (1) GPU NN (batched float64 cdist) == frozen cKDTree elementwise (indices equal, dist rtol 1e-9 / atol 1e-6); (2) parallel fresh coupling == serial coupling, max |dpi| ≤ 1e-6 (200-anchor reduced machinery, production scale deterministic by construction); (3) A1 reuse through the full stress-runner path (cache discovery → reuse → blend → NN → save) reproduces the campaign `pcc_global` 0.553326 within 1e-3, 0 fresh / 1 reused coupling; (4) TestParallelPairIsolation: 3 parallel pairs with distinct sizes → per-pair npz inputs exist, parallel couplings == serial ≤1e-6, exact shapes.
- Production: per-pair files verified in the T3_A1_s2 rerun (2026-09-01) — couplings written with per-pair names, no cross-pair contamination.
- Full suite: **209/209 pytest pass** (67 s, benchmark env).
- The A1 reuse test caught a real config bug (OT_ANCHORS=3000) — anchor-level couplings silently diverged from the campaign protocol. Fixed before any formal stress round ran.
| 2026-09-01 | SpatialZ | benchmark/src/adapters/spatialz_adapter_v2.py | Per-slice subprocess safety cap `timeout=21600` (6 h) parameterized as env var `SPATIALZ_SLICE_TIMEOUT_HOURS` (default 6.0 — unchanged for all existing callers). INC-005: full-density T3_A1_s2 (13 holdout slices) hit `TimeoutExpired` at 6 h on one slice (pair started 01:48, killed 07:48); the adapter's exception path cleaned the tempdir and exited unregistered, orphaning the subprocess. Timeout is orchestration only — generation math/params (alpha, num_projections, lr, nb_iter_max) untouched. Stress T3 rounds set 12 h | 6 h is too tight for a full-density Blackwell slice pair when the host is under concurrent campaign load |

## 2026-09-04 — machine migration: 2× RTX 5090 → 8× Tesla V100 (sm_70), paths & envs (math-neutral)

| Date | Method | File(s) | Change | Reason |
|------|--------|---------|--------|--------|
| 2026-09-04 | all | benchmark/envs/*.yaml (7 files) | torch line re-pinned to cu126 wheels (torch 2.12.1+cu126; isoST 2.9.0+cu126 + PyG wheel set); setuptools pin widened <70 → `>=78,<81`; headers document rationale | cu130 wheels need driver≥580 (this box: 570.195.03); cu128 wheels dropped sm_70 kernels; setuptools<70 conflicts with torch 2.12+ metadata |
| 2026-09-04 | isoST | isoST env (post-install) | `ln -s libnvrtc.so.12` → `libnvrtc.so` in `nvidia/cuda_nvrtc/lib/` | pykeops keops JIT binder link fails with `ld: cannot find -lnvrtc` once torch is imported (keopscore resolves libnvrtc via the already-loaded torch copy; its wheel dir ships no unversioned `.so`); verified full JIT + geomloss sinkhorn on cuda:0 |
| 2026-09-04 | all | 90 files under benchmark/scripts, benchmark/src, benchmark/tests, benchmark/configs, methods/, PROJECT_PATHS.md etc. | Path logistics: `/data/coding/xiepeiyan/data` → `/data/home/xiepeiyan/xiepeiyan/data`, `/data/coding/xiepeiyan/lab/baseline` → `/data/home/xiepeiyan/xiepeiyan/lab/baseline`, `/data/coding/xiepeiyan/.conda/envs/` → `/data/home/xiepeiyan/miniconda3/envs/` (bulk sed, prefix-only; no science touched). Historical artifacts under benchmark/results/ and prior CHANGELOG/METHOD_CHANGES entries intentionally left at their recorded paths | repo + data migrated to the new V100 box under the user home; tests/specs/adapters hardcoded old-machine absolute paths |

**Verification (2026-09-04, benchmark env):**
- Full pytest suite: **209 passed, 1 skipped, 0 failed** (162 s) — `TestA1ReuseProjection::test_reuse_reproduces_campaign_a1_pcc` (real MERFISH_ABCA-2 data, full stress-runner reuse path) reproduces the campaign A1 PCC within 1e-3.
- All 7 conda envs rebuilt and verified on real V100s (idle GPUs only, nice): cuda matmul smoke; per-env import batteries; isoST PyG knn_graph + pykeops JIT; spatialz squidpy 1.2.3/open3d/MENDER; unist lightning 2.6.5.

## 2026-09-05 — stress 战役 V100 恢复：STRESS_B3_f20 零 GPU finalize 救援（math-neutral）

| Date | Method | File(s) | Change | Reason |
|------|--------|---------|--------|--------|
| 2026-09-05 | SINTER3D | benchmark/scripts/rescue_sinter3d_finalize.py (new) | CPU-only finalize rescue for interrupted rounds: re-runs the post-prediction tail of `rerun_sinter3d_stress.run_round` (feature_subset truncation → `pair_by_coords` → `save_round` → `register_round`) on the EXISTING `run/X_pred.npy` + `query_coords.npy` + `ground_truth.npy`. Integrity gates: refuse if metrics.json/pred.h5ad already exist; assert X_pred finite and rows==query rows; assert spec GT (after truncation) == saved ground_truth.npy (allclose); require full exact-coordinate pairing. `runtime_sec`/peaks derived from the campaign ResourceSampler `resource_timeseries.csv` and marked `runtime_sec_source` | STRESS_B3_f20/SINTER3D prediction finished 2026-09-01 03:00 (X_pred.npy written, finite, official 10000-step config) but the process died before finalize. Rescue reuses that prediction verbatim — no retraining, no math change. Result: PCC_global 0.5837 / RMSE 0.1140, 75732/75732 paired; registry row 2026-09-05 |

**Verification (2026-09-05, benchmark env, CPU only):**
- `rescue_sinter3d_finalize.py STRESS_B3_f20` → verified 75732/75732; metrics.json + pred.h5ad written under STRESS_B3_f20/SINTER3D/; single registry row appended (run_id …20260905).
- GT integrity gate passed: spec-loaded GT (truncated to 20 features) == saved `ground_truth.npy` (exact).
- runtime_sec=2139.9 derived from sampler timeseries (original run, 09-01); peak_gpu_mib=17621 (sampler).

## 2026-09-05 — stress 战役 V100 恢复：probe 基础设施（T_max 定标，math-neutral）

背景：V100（8×SM_70）上尚无正式轮。资源政策第 3 条要求续跑前以 probe 实测步时重定
T_max。旧 probe（2026-09-01，5090 机）不可用于 V100，且旧 runner 的 probe 选择逻辑
（全树 sorted 第一个 `_probe_*/PROBE.txt`）会把旧硬件/异任务 probe 误用于新轮。

| Date | Method | File(s) | Change | Reason |
|------|--------|---------|--------|--------|
| 2026-09-05 | SINTER3D | benchmark/scripts/rerun_sinter3d_stress.py | ① probe 目录名可加后缀：`--probe-suffix v100_YYYYMMDD` → `_probe_<round>_<suffix>`（旧机器 probe 文件原样保留为已背靠轮次的溯源）；PROBE.txt 增记 machine/gpu_idx/gpu_name/probe_date。② probe 选择逻辑改为：目标轮名前缀匹配 `_probe_<round>*` 中 mtime 最新者；无匹配才回退全树最新；每轮独立选择（原逻辑：全树 sorted 第一个，单值全局复用） | 每轮 T_max 必须基于本机同任务实测；防止 5090 旧值或 STRESS/T3 异任务值污染；回退情形在 T_MAX.txt 标注。无算法/数学改动 |
| 2026-09-05 | SINTER3D | 同上 | `run_round` 在 run/T_MAX.txt 增写 `probe_source=<实际所用 PROBE.txt 全路径>`，metrics extra 增 `probe_source` | 单轮 T_max 依据可审计（政策第 5 条）；回退选择透明可见 |
| 2026-09-05 | isoST | benchmark/scripts/rerun_isost_stress.py | 新增 `--probe-epochs N` probe 模式：官方全量数据 × N epoch/阶段 跑通（不 save、不注册），写 `_probe_isost_<round>_<suffix>/PROBE.txt`（per_epoch_avg_sec、estimated_formal_sec=×300 官方 epoch 总数）；正式轮 T_max = 本轮 probe 文件 estimated×1.5+1800 s，basis 与 probe 来源写入 run/T_MAX.txt；无 probe 才回退 `--tmax-hours` | isoST runner 原无 probe 机制，T_max 硬编码 4 h（5090 依据）；V100 前必须先实测。调度/记录改动，epochs 等算法配置不变 |
| 2026-09-05 | isoST | benchmark/src/adapters/isost_adapter.py | run_impl 内 T_MAX.txt 写入改为 first-writer-wins：仅当文件不存在才写（去掉硬编码 "measured_step_sec=67.5" 陈旧依据文本，改为中性 adapter-default 文本） | runner 先写的完整 T_max basis（含 probe 来源）曾被 adapter 以陈旧硬编码文本覆写；顺序语义即记录语义 |
| 2026-09-05 | SpatialZ | benchmark/scripts/probe_spatialz_slice.py (new) | 单片 probe：官方全量配置（full slices、num_projections=80）对默认中间 holdout 片生成 1 张虚拟片（逐片 subprocess 机制不变），写 `_probe_spatialz_<round>_<suffix>/PROBE.txt`（per_slice_hours、estimated_formal_sec=×n_holdout×1.5）；正式轮以 `--per-slice-hours` 喂给 rerun_spatialz_stress.py | SpatialZ 逐片生成可 >6 h/片（INC-005），13 holdout 全密度轮启动前必须实测单片 V100 成本。无算法改动（SPATIALZ_SLICE_TIMEOUT_HOURS 默认上限提为 12 h 仅为编排护栏） |

**Verification (2026-09-05):**
- 语法自检 4 文件通过；旧 PROBE.txt 解析兼容（key 行不变，仅新增 machine/date 行）。
- 选择逻辑单测（pick_probe / pick_isost_probe 前缀 + mtime）：待 V100 probe 落盘后以实物核验。

## 2026-09-06 — benchmark env: psutil pin（ResourceSampler 依赖补装，math-neutral）

| Date | Method | File(s) | Change | Reason |
|------|--------|---------|--------|--------|
| 2026-09-06 | — | benchmark/envs/benchmark.yaml | `psutil>=5.9` 加入 dependencies；benchmark env 已 `pip install psutil`（7.2.2） | 09-04 env 重建遗漏 psutil，战役 ResourceSampler 采样线程 import 失败静默死亡：2026-09-06 首启 6 条正式 lane（9 轮）的 resource_timeseries.csv 仅剩表头、metrics peak_* 归零。已终止首启并重启全部 lane（重启时各轮训练 <3 min，无结果污染；registry 无行写入）。纯观测/记录依赖，无算法影响 |

**Verification (2026-09-06):** 重启后 6 lane ResourceSampler 均正常写 resource_timeseries.csv（t_sec,gpu_mib,gpu_util_pct,ram_mib,cpu_pct 逐行增长）；GPU 空闲确认（GPU5/6 为 linpeir+ 既有 llama-server，未触碰）。

## 2026-09-06 — SpatialZ 适配器取证改造：逐片子进程日志持久化（INC-006，math-neutral）

背景：SpatialZ T3_A1_s2 重跑（09-06 r2）第 10/13 片 s023 的子进程被 signal 击杀（rc≠0 且
captured stderr 为空，疑似 cgroup OOM，见 INCIDENTS.md INC-006）。适配器原实现仅把子进程
stdout/stderr 存内存、失败时截尾拼进异常——空 stderr 场景零取证。INC-005 教训"重跑必须带
独立持久 log"仅落实在 runner 层（父进程 stdout），未覆盖子进程输出。

| Date | Method | File(s) | Change | Reason |
|------|--------|---------|--------|--------|
| 2026-09-06 | SpatialZ | benchmark/src/adapters/spatialz_adapter_v2.py | 每片 subprocess.run 后（无论成败）将完整 stdout/stderr + cmd + elapsed_sec + returncode 持久化至 `<out_run>/logs/slice_<sl>.log`；失败分支 AdapterError 消息与 print 增带 `rc=` | 空 stderr + rc≠0（signal 击杀）场景下仍保留子进程输出与退出码证据；仅编排/记录层改动，Generate_spatialz 调用与配置逐字节不变 |

**Verification (2026-09-06):** 语法自检通过；pytest 全量（benchmark env）绿后随正式提交；
s028 单片 probe 运行中首份 `logs/slice_ABCA2_s028_log_PC.log` 实落盘即核验点。

## 2026-09-06 — stress 收尾分析：stage 脚本 `--stamp` + dashboard notebook 路径修复 + env jupyter 工具链（orchestration，math-neutral）

背景：`stress_campaign_20260901.ipynb`（dashboard）从未执行；cell 1 硬编码迁移前机器路径
`/data/coding/...`；stageC/D 脚本输出固定写 `STAGED_20260901`/`COST_TABLE_20260901.md`
（09-01 快照已 git 跟踪，须保留为历史）；执行 notebook 需 nbconvert/ipykernel（09-04 env
重建未含）。以下变更均为编排/路径层，不触碰方法算法与任何指标计算。

| Date | Method | File(s) | Change | Reason |
|------|--------|---------|--------|--------|
| 2026-09-06 | — | benchmark/scripts/stageC_cost_table_20260901.py, stageC_sensitivity_20260901.py, stageD_domain_recovery_stress.py | 各增 `--stamp YYYYMMDD`（默认 `20260901` → 裸跑行为不变）；输出路径随 stamp；cost 表标题/Notes 补 GPU 双类窗口（5090 轮 09-01→09-04、V100 轮 09-05→09-06，probe-based T_max） | 最终矩阵产物落新快照 `STAGED_20260906/` + `COST_TABLE_20260906.md`，09-01 快照保留 git 历史；两代 GPU 成本不可混为跨方法速度比较（P7 operational only） |
| 2026-09-06 | — | notebooks/dashboards/stress_campaign_20260901.ipynb | cell 1 `REPO` → 当前机器路径；`STAGED` → 20260906；`run_script` 支持透传参数；stageC/D 调用与 md 产物路径同步 20260906 | 执行环境迁移后路径修复 + 最终快照输出定位 |
| 2026-09-06 | — | benchmark/envs/benchmark.yaml | dependencies + `nbconvert>=7`、`ipykernel>=6`；benchmark env 已 `pip install nbconvert ipykernel` | executed-notebook 工具链缺失（同 psutil 先例 2026-09-06）；无算法影响 |

**Verification (2026-09-06):** 三脚本 `--stamp` argparse 冒烟通过；notebook 首次全量执行成功
（nbconvert + ipykernel，benchmark env，pandas 2.3.3），0 error cell。执行中修掉两处 notebook
兼容问题（均 orchestration）：cell 1 `REPO` 需仓库全路径双 xiepeiyan 段（`/data/home/xiepeiyan/
xiepeiyan/lab/baseline`）；`SeriesGroupBy.astype` 在 pandas 2.3.3 不存在 → cell 3 改
`pd.to_numeric(errors='coerce')` 分组均值。产物核验点全部落盘：`STAGED_20260906/` 全套
（sensitivity_table/summary、robustness_curves + figures + GATE4_NOTES.md 3 flagged rows、
domain_recovery_stress.tsv 53 rows + 24 轮 GT labels）、`COST_TABLE_20260906.md`（含 V100 双类
GPU 注记）、`stress_campaign_20260901_executed.ipynb`（首 cell 后附执行注记 md）。

## 2026-09-07 — SpatialZ runner T_max 编排参数（PLAN §11.3.1 重定标公式，math-neutral）

背景：INC-005/006 后 SpatialZ T3_A1_s2 唯一正式缺口需按 PLAN §11.3.1 公式重跑：
`est = 4.69 h（09-06 r2 前 9 小片实测）+ 4×P（P = s028 最大留出片 probe 实测 per_slice_hours）`；
`T_max = 1.5×est + 1800 s`。原 runner 恒用 `per_slice_hours × n_holdout × 1.5 + 1800`
（全部 13 片按同一 P 算）并无条件覆写 run/T_MAX.txt——按旧公式反推 P 写 basis 会造假，
故加显式编排入口。无算法/配置改动（官方 config 行不变：max_cells_per_slice=None、
num_projections=80、逐片 subprocess 顺序生成）。

| Date | Method | File(s) | Change | Reason |
|------|--------|---------|--------|--------|
| 2026-09-07 | SpatialZ | benchmark/scripts/rerun_spatialz_stress.py | `run_round` 增可选 `tmax_est_hours`/`tmax_basis`：给定时 `T_max = 1.5×est+1800 s` 且 T_MAX.txt basis 行写 `--tmax-basis` 文本（含 probe_source 与 PLAN 公式）；缺省路径与既有 basis 文本逐字不变 | PLAN §11.3.1 尾片重定标公式入口；T_max 溯源可审计。纯编排参数 |

**Verification (2026-09-07):** `py_compile` 通过；`--help` 两参数可见；缺省行为回归由
`--help`/代码路径审查确认（tmax 计算分支仅当 tmax_est_hours is not None 生效）；
注册表 benchmark_commit 语义不变（运行前提交，取 HEAD）。

## 2026-09-07 — Multi-slice-missing 战役编排泛化（EXTENSION_DESIGNS_20260907.md 提案 A，用户批准）

背景：契约冻结（`benchmark/configs/tasks/MISSING_54/` 4 轮，track `Multi-slice-P2`，
留出=z 中位窗口 A1/B3 中部连续 2/3 片 run，其余片全可见、官方全量、无扰动）。原
`_stress_common` 的注册表与 run_id 生成硬编码指向 2026-08-31 stress 战役（58/59 行计数
叙事）；新战役若混入会污染历史行计数，故加 env 覆盖（默认值不变 → 历史行零影响）。
方法侧零改动（SINTER3D/ReHo/isoST/PASTE 官方配置与算法不动）。

| Date | Method | File(s) | Change | Reason |
|------|--------|---------|--------|--------|
| 2026-09-07 | —（benchmark 编排） | benchmark/scripts/_stress_common.py | `TASK_DIRS` 增 `MISSING_54`；`REGISTRY`/`CAMPAIGN_TAG`/run_id 家族改读 `STRESS_REGISTRY`/`STRESS_CAMPAIGN_TAG`/`STRESS_RUNID_FAMILY` env（缺省=原值）；新增 4 份 task yaml 只加 `split.gap`/`split.location` 展示字段 | 扩展战役 self-contained 注册（新结果树/新注册表 `state/run_registry_multimissing.tsv`，run_id `MERFISH__MISSING_54_<round>__<method>__MULTIMISS20260907__<date>`），历史 58/59 计数不受污染。纯编排/注册参数，无算法改动 |

**Verification (2026-09-07):** pytest 209 passed / 1 skipped（补跑全绿）；4 yaml
round-trip + 窗口/run 连续性断言通过；历史 SPARSE_54/STRESS_54 任务加载回归通过。

| 2026-09-07 | —（benchmark 编排） | benchmark/scripts/rerun_baselines_stress.py | `main` 增 `--skip-reho`（与 `--no-ot` 对称）：PASTE-only 运行不再执行 ReHo 阶段，json 记录按实际执行的方法组装 | 扩展战役 PASTE-OT-Linear 2 轮（M2_A1/M3_A1）启动需要：runner 恒跑 ReHo 会重复登记已入库的 4 行 ReHo（registry 污染）。纯编排门控，无算法/配置改动 |

**Verification (2026-09-07):** `py_compile` 通过；`--help` 新参数可见；缺省行为（两者
都跑）由既有 ReHo 4 轮 json/注册行确认不受影响。

| 2026-09-07 | —（benchmark 编排） | benchmark/scripts/rerun_spatialz_stress.py | `run_round`/`main` 增 `--out-suffix`：仅追加输出目录名（如 `T3_A1_s2_20260907/`），registry run_id/round 保持纯净（`register_round` 仍用原 round_name） | INC-006/007 旧部分证据目录（`T3_A1_s2/SpatialZ/run/`）只读红线：正式 13 片轮重跑须新目录（PLAN §11.3.2"新目录勿复用旧失败目录"），同时 registry 59 行 round 名必须为纯 `T3_A1_s2`。纯编排目录参数，无算法/配置改动 |

**Verification (2026-09-07):** `py_compile` 通过；`--help` 新参数可见；缺省（无 suffix）行为
不变——`out_name or round_name` 全等路径。

## 2026-09-09 — PASTE-OT-Linear 全 NaN 预测守卫（INC-008 事件处置，math-neutral）

背景：Stage-2 首发轮实测发现，注册噪声（n-class，`additive_gaussian_logPC`，
log-PC 存储空间直接加性高斯）使 OT pair 片内出现少量负表达式元（n01 α=0.1 时每片
4–5 个；n03 ~150–300；n05 ~700–1600）。`_run_ot.py` 的 KL 代价 `log(pA+eps)` 对负质量
profile 产生 NaN（np.log 仅警告），efgw 输运计划 100% NaN 并静默落盘 → fusion/NN →
预测 100% NaN → save_round 出 pcc=0.0/NaN 指标 → register_round 将垃圾记为
"success"（registry 59→61 两行污染，已删除恢复）。方法算法/配置零改动。

| Date | Method | File(s) | Change | Reason |
|------|--------|---------|--------|--------|
| 2026-09-09 | —（守卫，系统性） | benchmark/scripts/_audit_common.py | `save_round` 入口增非有限预测守卫：`yp` 含 NaN/inf 即 `raise RuntimeError`（rc≠0，不写 metrics.json/pred.h5ad，不注册）；isoST/PASTE/ReHo/UniST/SpatialZ 全部经此保存路径 | INC-008 系统性收口：任何 runner 的静默垃圾（全 NaN 预测 rc=0 "success"）必须响亮失败；方法算法/配置零改动，合法（有限）预测零影响 |

| 2026-09-09 | —（守卫） | benchmark/scripts/rerun_baselines_stress.py | `run_paste_round` 组装 `X_pred_paste` 后增守卫：`np.isfinite(...).all()` 不成立即 `raise RuntimeError`（rc=1，不 save/register），错误信息注明 NaN coupling 与 KL 域原因 | 拒绝把全 NaN 垃圾注册为 "success"；rc≠0 → 调度器 ERROR_CELL（no auto-rerun, alert）。纯失败-显式化守卫，非算法/配置改动；对合法（有限）预测零影响 |

**Verification (2026-09-09):** `py_compile` 通过；B3_n05/A1_n05 已跑垃圾（守卫前）人工
复核删除注册行；n-class 6 格判定：pair 片内负元确定性存在（seed 0，逐片计数证据）→
KL 输入域不可行 → 队列 HOLD（`/tmp/stage2/HOLD_CELLS.txt`：paste:B3_n01, paste:B3_n03）+
已运行 2 格（A1_n01/A1_n03）killpg 终止（rc=15, ERROR_CELL）；处置上报用户待批。
| 2026-09-09 | —（启动修复） | benchmark/scripts/rerun_unist_stress.py | `main()` round 解析改为跨全部 task family 预解析：先在 `TASK_DIRS` 全集找含该 round 的 family，找不到才 `SystemExit`（不再按序逐 family 校验、首个不含即退出） | INC-009：SPARSE_54 先于 STRESS_54 在 `TASK_DIRS` 中，STRESS_54 的 d/f/n rounds 全部在启动瞬间死于 rc=256（"not in SPARSE_54"），永远到不了含它们的 family。启动/编排修复，非算法/配置改动；无参数时默认 T3 机会型行为不变 |

**Verification (2026-09-09, INC-009):** `py_compile` 通过；bogus round 报新错误路径
`STRESS_A1_zzz not in any task family (SPARSE_54, STRESS_54, MISSING_54)`；loader 断言
`STRESS_A1_d10`/`STRESS_B3_d10` ∈ STRESS_54 specs；3 个受害者 cell（A1_d10/B3_d10/A1_d25）
re-queued，HOLD 全解除，scheduler 重启后依 lane cap 重发。d10 管线语义核对：density_frac
扰动手 manifest 与 runner slice-holdout 机制兼容（probe 已验 SUICA 全管线 + GPU-OOM 门）。


<!-- New-method campaign (NTF, DeepSpatial) registration 2026-09-11 -->
# METHOD_CHANGES.md — pending entry (staged 2026-09-11)

`benchmark/METHOD_CHANGES.md` is root-owned and not writable by the benchmark user
(no sudo); this is the owner-runnable registration text for the new-method campaign.
No file under `methods/` was modified: NTF @ f0b4941 and DeepSpatial @ fdb98657 are
byte-identical to their official clones.

## NTF

- Adapter: `scripts/new_methods/ntf_pipeline.py` (benchmark I/O boundary only).
  Imports the official `NTF.config/models/train/sample` unchanged.
- Formal configuration: official defaults (`n_iter 20000`, batch 8192, lr 1e-4, hash grid
  12x2 / 2^19, width 64, depth 1, PSF 4, official early stopping) plus the user-required
  `--no_dropout` and the **official** `--single_precision` flag.
- Reason for `--single_precision`: the official default mixed-precision (fp16) path is
  numerically broken in this environment. At small scale (smoke / 2048-point probe) it drives
  the tiny-cuda-nn hash-grid encoding to ~1e-6 and returns a coordinate-independent constant
  field; at benchmark scale (1,000,000 cells, official simulated data) it keeps accuracy
  (per-gene Pearson 0.8514 vs 0.8521 in fp32) but quantises the field to 9,628/100,000 distinct
  rows, i.e. an fp16 solution that is not a faithful expression field. Reproduced on the
  **official simulated dataset** (`analysis/simulate_data.py`) with the untouched official
  code, so it is a numerics issue in the fp16 path, not a benchmark artefact. With
  `--single_precision` the field is live and fully resolved. Both flags exist upstream; the
  model core is untouched. Evidence: `resume_20260911/diag/`,
  `logs/new_methods_agent/diag_ntf_simulated_control_20260911G/REPORT.md` (arms A/B/C),
  `logs/new_methods_agent/diag_ntf_ablation_20260911H/REPORT.md` (arm V5).
- Guard: the adapter refuses to emit a formal prediction when `unique_prediction_rows < 2`.
- No `normalize_adata`, no PCA/scaling fitted on expression; visible-only CSR training input;
  query carries xyz/ids/slices/features only.

Iterations H-I (2026-09-11) added no change here: the NTF under-fit mechanism diagnostics and
the independent HDF5-level re-verification touched no method code, and `methods/NTF` is still
byte-identical to the official clone @ `f0b4941a`.

## DeepSpatial

- Adapter: `scripts/new_methods/deepspatial_pipeline.py`. Imports the official
  `deepspatial` package (`setup_data` / `build_model` / `fit` / `reconstruct_between_slices`)
  with the official tutorial hyperparameters.
- `load_official_checkpoint`: upstream `DeepSpatial.load_checkpoint` forwards `train_config`
  keys that `build_model` rejects (`prediction`, `train_eps`, `sample_eps`, `ema_decay`), so
  the official loader raises `TypeError` on checkpoints the official trainer itself writes.
  The adapter rebuilds the identical architecture from the saved `config.json` (restricted to
  `build_model`'s signature) and loads the same Lightning `state_dict`. Network, weights,
  hyperparameters and sampler are identical; `methods/DeepSpatial` is unmodified.
- Common-task contract: the frozen task supplies expression + coordinates + slice only, so the
  per-cell-type channel (used by the official tutorial) is a single constant class. Registered
  in each `run_manifest.json` as `label_input`; this matches the information given to every
  other common-task method.
- Output conversion (math-neutral, post-processing only):
  1. Serialization clamp of tiny negative values (unbounded official regression head) to 0,
     with the exact counts/magnitudes recorded in `resources.json` (`output_clamp`).
  2. Coordinate-only projection clamps the per-bin reference z to the generated z support and
     records `target_outside_z_support` / `max_target_z_gap` (a hold-out slab is thick and
     overlaps the bracket slabs; up to ~21% of target z lie outside the generated volume on
     the posterior rounds).
- Harness fix (not a method change): `scripts/new_methods/watch_command.py` no longer aborts
  when a stale monitor directory exists (it appends a UTC suffix).

### DeepSpatial — projection reproducibility and uniformity (2026-09-11, second pass)

Found while independently re-verifying the staged artifacts; both issues are in the
benchmark-side *projection* stage, not in the official method, training or checkpoint.

- **Unseeded generator.** `deepspatial.core._generate_and_prune_optimized` draws the time
  distribution (`torch.rand`) and the source cells (`torch.randint`) for the neural-ODE point
  cloud, and the adapter's `predict` did not seed them. Every round was therefore a different
  random draw: two projections of the same A1 checkpoint correlated only r = 0.53.
  Fix: `predict` now seeds numpy/torch/CUDA with `--seed` (default 0) immediately before
  generation. The sampling algorithm, the official module and the checkpoint are unchanged;
  two seed-0 projections are bit-identical in content. Metric sensitivity was measured on A1
  with three seeds: pcc_global 0.4954/0.4925/0.4943, rmse 0.1173–0.1176, cosine 0.9635–0.9637
  (evidence: `results/new_methods_20260911/DEEPSPATIAL_SAMPLING_SENSITIVITY_A1.json`).
- **Non-uniform projection revision across rounds.** A3's projection used an earlier revision
  without the generated-z-support handling (no `target_outside_z_support` / `n_fallback_bins`
  in its `nn_match`). All six rounds were re-projected with the single current revision
  (`ed46c48e…`, seed 0, steps 100) reusing the existing trained checkpoints; the old
  projections are preserved under
  `resume_20260911/superseded/deepspatial_unseeded_20260911T1145Z/`.
- **Projection provenance.** Each round now writes `projection_provenance.json` (adapter
  sha256, seed, steps, density/z-slab/chunk settings, checkpoint name + sha256,
  generated-segment sha256) and `validation.json` additionally records
  `prediction_array_sha256`, because identical arrays can serialize to different `.h5ad`
  bytes (the container sha256 is not a content fingerprint).
- **Generated point-cloud density (harness parameter, documented).** The official API takes an
  absolute `thickness` (µm between generated cells); the tutorial's own value (10 µm) is tied
  to its dataset's slice spacing. The benchmark needs the hold-out slab sampled at the frozen
  query resolution, so the adapter fixes a per-gap layer count,
  `thickness = gap / density_factor` with `density_factor = 5`, chosen a priori and identical
  for all six rounds. A 4× sensitivity probe (2.5 / 5.0 / 10.0) moves pcc_global by ≤0.0033
  (0.4971 / 0.4954 / 0.4938) — comparable to the seed spread — and denser is slightly worse,
  so the value does not inflate the method. Note that the first-pass `run_manifest.json` calls
  this field `gap_thickness_ratio` under `official_tutorial_hyperparameters`; it is the
  benchmark density factor, not a literal official parameter name.

## Metric engine

Unchanged. Both methods are scored with the existing campaign evaluator
`benchmark/scripts/_audit_common.compute_v11_metrics` on the frozen PC_1..PC_50 space with
exact-coordinate pairing; the evaluator reads held-out expression only after a hash-bound
prediction validation receipt is persisted.

## 2026-09-13 — new methods in the stress families: additive runner layer (NTF / DeepSpatial)

**Context.** NTF and DeepSpatial were registered on the LOSO_54 Common-task track only. The
six established methods additionally cover the STRESS_54 / SPARSE_54 / MISSING_54 families.
Reaching parity needs those families to be reachable, but the new methods' drivers could not
provide it:

- `scripts/new_methods/ntf_pipeline.py` and `deepspatial_pipeline.py` are **byte-frozen** —
  their sha256 is bound into the registered evidence (`benchmark_script_sha256` in every
  `run_manifest.json`, `projection_code_sha256` in every `projection_provenance.json`), so
  editing them would invalidate frozen records.
- Both hardcode the Common-task identity: `prepare` reads
  `benchmark/configs/tasks/LOSO_54/task_<round>.yaml`, `evaluate` resolves the holdout through
  `_audit_common.ROUNDS` (six LOSO rounds only), and the output's `uns['task_id']` is
  `'MERFISH_LOSO_54_' + round`.
- No existing stress runner can host them: `rerun_{baselines,sinter3d,isost,unist,spatialz}_stress.py`
  are single-method and dispatch through `conda run`, whereas these two methods run in venvs.

**Change (all new files; no frozen file and no method code touched).**

| File | Role |
|---|---|
| `benchmark/scripts/_new_methods_stress_common.py` | Stress-side I/O, evaluation and provenance for the new methods. |
| `benchmark/scripts/rerun_ntf_stress.py` | NTF runner for `--task {STRESS_54, SPARSE_54, MISSING_54}`. |
| `benchmark/scripts/rerun_deepspatial_stress.py` | DeepSpatial runner (stages `run` / `predict` / `evaluate`). |

**Math-neutrality argument.** The method computation is not restated. Both runners import the
frozen Common-task driver and invoke its stages **verbatim** (`ntf_pipeline.train`,
`deepspatial_pipeline.train/predict`). The runners own exactly four things, none of which
touches the estimator:

1. **Input construction.** The perturbation is not reimplemented — it is imported verbatim
   from `_stress_common.build_perturbed_adata`, so the new methods see byte-for-byte the same
   degraded input as the six established methods. For unperturbed rounds the builder uses the
   same h5py row-span reader as the frozen `prepare`, so the bytes match the LOSO input.
2. **One declared metadata correction**, `relabel_task_id()`: the frozen stages label their
   output with the Common-task id, which is factually wrong on a stress round. The helper
   rewrites `uns['task_id']` (and adds `uns['task_family']`), asserts the prediction array is
   unchanged, re-derives the hash-bound `validation.json` receipt, and records before/after in
   `postprocess_<stage>.json`. Nothing numeric is altered.
3. **Evaluation** against the family's holdout, through the same campaign engine
   (`_audit_common.compute_v11_metrics`) — the frozen `evaluate` cannot be used because it
   indexes `ROUNDS`.
4. **An equality gate** (below).

**Correctness gate (this is the load-bearing evidence).** `STRESS_{A1,B3}_base` are
unperturbed and their split is byte-identical to `LOSO_54 A1_gap1_anterior` /
`B3_gap3_posterior`. Verified input identity:

| round | adapter `visible.h5ad` / `query.npz` | registered LOSO input | |
|---|---|---|---|
| `STRESS_A1_base` | `6366e8ab…9346` / `650ba7d8…87ed` | `6366e8ab…9346` / `650ba7d8…87ed` | identical |
| `STRESS_B3_base` | `06bb3c77…5678` / `392fdfa2…b6b9` | `06bb3c77…5678` / `392fdfa2…b6b9` | identical |

With identical input bytes and fully seeded pipelines the stress prediction must reproduce the
registered LOSO prediction. It does, to float32 tolerance — not bitwise, because tiny-cuda-nn's
hash grid uses atomic accumulation and is not run-to-run deterministic. Measured on the NTF
anchors: `max|delta|` 2.2e-06 and 9.2e-07 on log-PC values in [0,1]. `CANON_EQUAL_TOL = 1e-3`
separates that from a real change: the same runner in `--mode smoke` (5 iterations,
subsampled data) differs by 0.36, i.e. four orders of magnitude larger. A round exceeding the
tolerance is **refused**, not recorded.

**Registration.** Anchor rows were staged to `state/run_registry_stress_anchor_newmethods.tsv`
rather than the shared `state/run_registry_stress_anchor.tsv`, because the 2026-09-13 anchor
launcher was still appending to the latter (UniST lanes). The launcher finished
(`ALL LANES DONE`, 2026-09-14T17:42Z), the 4 rows were folded into the shared registry, and the
staging file was removed — two copies of registry rows can drift, and there is only one
authoritative file.

**Feature-subset rounds (`f10/f20/f30`) — resolved 2026-09-14, without touching a frozen file.**
Those rounds truncate the visible matrix to 10/20/30 PC columns, which the frozen stages reject
(`assert list(a.var_names) == FEATURES`). The *methods* have no problem with a narrower input —
`methods/NTF/NTF/data.py:32` (`self.n_genes = self.v.shape[1]`) sizes every output head from the
data, and the official DeepSpatial model reads the same way; only the adapters' width assertions
blocked the rounds. Mechanism (`benchmark/scripts/_new_methods_stress_nfeat.py`):

* **DeepSpatial** — every width reference in the frozen file goes through its module-level
  `FEATURES`, so the round needs only a rebind. Zero code copied.
* **NTF** — four assertions embed the width. Rather than hand-copying the frozen `train` body
  (which invites silent transcription drift), the variant is **generated from the frozen source
  at import time** by anchored textual substitution, each anchor asserted to occur exactly once,
  so an upstream change fails loudly instead of diverging quietly. Two subtleties surfaced only
  by running it: `FEATURES` serves two different widths (the `metadata()` check validates the
  *source* dataset and must stay 50), and `train()` ends by calling an internal `validate()` that
  carries its own width pair with different spacing — missing it killed every feature-subset
  round at the very end of training, after the GPU work was already spent.

No arithmetic changes: the only edits are the width used by four shape assertions; model, loss,
optimiser, seeding, scaling and sampling sequence are the frozen file's own code. **Boundary
validation** (`benchmark/scripts/validate_nfeat_variant.py`) is what makes a variant result
trustworthy: the generated variant must reproduce the frozen pipeline exactly at the shared width
`nfeat = 50`, checked against the registered LOSO prediction — it passed at `max|delta|`
1.7e-06, the same order as the anchor gate's own float noise.

**Tool-chain parity (2026-09-14).** Bringing the two methods onto the established tracks exposed
four places where existing tooling silently recognised only the six established methods. All four
are fixed in tooling, not in method code:

| Gap | Fix |
|---|---|
| DeepSpatial's frozen stages write into `pred/`, every other method (and the Common-task tree) keeps the deliverable at the cell root — so `aggregate_stress54_family.py` / `build_cross_method_table.py`, which read `<cell>/metrics.json`, would have **silently omitted every DeepSpatial row**. | `run_new_methods_parity.py flatten` **moves** `pred/*` up to the cell root (never copies — two `metrics.json` files would leave no way to tell which one a reader took). `round_dir()`/`is_done()` accept both layouts so no completed cell is re-run. Both stages write `method_config.json` / `run_manifest.json` / `resources.json`; the first two are byte-identical (deduplicated), while `resources.json` really differs (the run stage records `training_sec` and the trajectory-tensor shapes, the predict stage the projection settings). Neither is discarded: the predict-stage file keeps the plain name — the already-merged Common-task deliverable holds that schema — and the run-stage record is preserved as `resources_train.json`, printed rather than renamed silently. |
| The notebook builder hardcoded the STRESS root; MISSING_54 rounds live under a different campaign root. | Resolve through the driver's family → root map. |
| The notebook builder hardcoded `pred/`. | Detect the layout instead. |
| `audit_stress54_registry_evidence.py` required a `run/` directory, which only the six runners create. | Teach the audit the new methods' equivalent per-cell record (`driver_snapshot.py`, `resources.json`, `run_manifest.json`) rather than fabricate a `run/` directory to satisfy the checker. `NTF`/`DeepSpatial` added to its `METHODS` so `by_method` counts and orphan detection cover them too. |

**Capability limit, recorded not filled.** DeepSpatial generates a point cloud strictly between two
visible slices (`bracketing()` asserts a visible slice on each side of the target z).
`T3_B3_s3` and `T3_B3_s4` hide `s060`/`s061` while the highest visible slice is `s059`, so 2 of
18 (resp. 20) holdout slices have no upper bracket. Recorded as **`not_applicable_task`** with the
reason in `run_new_methods_parity.py::NOT_APPLICABLE` — never silently skipped, never filled.
This matches how the track already treats partial T3 coverage: isoST, SpatialZ and PASTE-OT-Linear
each ran one of the six T3 rounds; only SINTER3D and ReHo ran all six.

# Adapter housekeeping (2026-09-15)

## `spatialz_adapter.py` — retired P1 prototype deleted; the 54-slice adapter takes the name

**Not a method change.** No adapter logic, config, or number moved; this entry exists so that a
reader meeting the old filename in an earlier record can find where it went.

Two files existed under one name:

| File | What it was |
|---|---|
| `spatialz_adapter.py` | The original 2026-08-04 prototype (`91c24e0`): one function `run_spatialz_p1()` on the retired **5-slice P1 protocol** (hardcoded z=4,9,14,19,24; holdout z∈{9,14}), with its own data path and a nearest-neighbour match against ground truth. |
| `spatialz_adapter_v2.py` | The live adapter (`b26420d`, 2026-08-11): `class SpatialZAdapterV2(BaseAdapter)`, 54-slice unified dataset, fixed-coordinate protocol — the one used by every SpatialZ round in all three task families. |

**Deleted:** `spatialz_adapter.py` (the P1 prototype). Evidence it was dead, on three independent
counts: `run_spatialz_p1` had **no caller** anywhere in the repository (its definition was its only
occurrence); **no module imported** the file; **no test covered it**. It also implemented only the
P1 protocol, which no current task family uses — all three now run `fixed_coordinate` on the
54-slice dataset. Its last execution left two artefacts in the data directory dated 2026-08-04, the
day the prototype was written, and nothing since. Git preserves it at `91c24e0`.

**Renamed:** `spatialz_adapter_v2.py` → **`spatialz_adapter.py`**, class `SpatialZAdapterV2` →
`SpatialZAdapter`. The `_v2` suffix existed only to distinguish the file from the prototype above;
with the prototype gone it was a version label the project does not use (provenance is carried by
git). Reference sites updated so every executable path still resolves:

- `benchmark/scripts/rerun_spatialz_fixed.py`, `rerun_spatialz_stress.py`, `probe_spatialz_slice.py`
- the two committed probe drivers under `benchmark/results/large_scale_benchmark_stress_20260831/_probe_spatialz_*`
- the prose/output mentions in `notebooks/methods/SpatialZ_T3_A1_s2_20260907.ipynb`

**Left as-is on purpose** — these are dated records of what was true when written, and rewriting a
path inside them would misstate history: the entries above in this file, `benchmark/CHANGELOG.md`,
`INCIDENTS.md`, the `memleak_audit_20260907.md` report, the `phase_0_audit/inventory.json`
snapshots, a raw `events.jsonl` diagnostic log, and one captured directory-listing output in
`notebooks/benchmark_control/00_project_audit.ipynb`. Read the old name in any of them as this file.

---

## 2026-09-20 · SINTER3D：官方 `utils.preprocess` 在**全部**基准运行中被跳过（未登记，现补登）

**这不是一次改动，而是一次此前未被记录的既成事实。按项目规则（`EXPERIMENT_RESOURCE_POLICY`
"Any deviation from official config requires explicit user approval and registration"），
此处补登，并交由用户裁决。**

### 观测

六个主表（LOSO_54，A1–B3）运行日志中，每一轮都出现：

```
Basis: 1 cell types
Preprocessing with SINTER3D utils...
Finding highly variable genes...
WARNING: preprocess failed ('cell_type'), continuing with manual prep
Training SINTER3D (10000 steps, hidden_dims=[512, 128])...
```

`utils.preprocess` 抛出 **`KeyError: 'cell_type'`**，被 `sinter3d_adapter.py:285` 的
`except` 接住，走 **manual prep** 回退：

```python
except Exception as e:
    print(f"WARNING: preprocess failed ({e}), continuing with manual prep")
    adata_st = all_adata.copy()          # 原始拼接，未经官方预处理
    adata_basis_processed = adata_basis.copy()
```

### 证据（四层，互相独立）

| # | 层 | 事实 |
|---|---|---|
| 1 | 源码 | `sinter3d_adapter.py:272-280` 传 `celltype_ref_col='cell_type'`，但 `adata_basis.obs` 只有 `'group'`（第 254 行）→ `adata_ref.obs['cell_type']` 必 `KeyError` |
| 2 | 数据 | 标准化数据 obs 为 `['slice','annotation']`，**无** `cell_class`/`cell_type` |
| 3 | 模型 | `model_basic.py:121` `n_celltype = adata_basis.shape[0]` = 1 → `deconvolutioner` 输出退化为平凡 1-单纯形 |
| 4 | 运行日志 | 6 个主表日志全为 `cell_types=1`；全库 `cell_types=<数字>` 共 66 处，**无一例外** |

**且全库 `Calculate basis for deconvolution`（`utils.py:71` 的成功标记）出现 0 次**——
即**没有任何一次运行真正跑通过官方预处理**。

### 关键补充：这份数据**根本没有细胞类型标注**

上面第 2 层只说"列名不匹配"，那会让人以为"改个列名就好了"。**实测不是这样**：

```
源数据 obs['annotation']：1 个取值，'unannotated'，覆盖全部 1,056,520 个细胞
```

`annotation` 是**常量占位符，零信息量**。所以：

- **不存在"该映射而没映射"的可用列**——这份 benchmark 数据里**没有细胞类型标签**。
- **因此不构成泄漏**：可见输入里带着 `annotation` 也无信息可泄
  （`spatialz_adapter.py:63` 的 `cell_type_key = 'annotation'` 读到的是常量）。
- **因此 (b) 修复只能恢复预处理、不可能恢复去卷积**——类型数永远是 1，
  因为**没有类型可用**。想要多类型参考，只能引入**外部**图谱，
  而那才会让 SINTER3D 真正变成 Multimodal（见 D1）。
- 也解释了领域指标为何用 `fit_reference(X_gt)` + `cluster_in_reference` 而非读标注：
  **因为没有标注可读**。

### 影响范围（实测，2026-09-20）

| 结果树（族） | 命中回退 | 说明 |
|---|---|---|
| `large_scale_benchmark_rerun_20260820`（**LOSO 主表**） | **6 / 6** | 主表全部轮次 |
| `large_scale_benchmark_stress_20260831`（**Task 4 压力族**） | **28 / 28** | 全部压力轮 |
| `large_scale_benchmark_multimissing_20260907`（**多片缺失族**） | **4 / 4** | 全部轮次 |
| `large_scale_benchmark`（迁移前的旧树，已被 20260820 重跑取代） | **6 / 6** | 同样全部命中 |
| `large_scale_benchmark_fixed` | **0 / 6** | ⚠️ **不是阴性结果**——这 6 个目录里只有 `metrics.json` + `pred.h5ad`，**没有日志可查**。它们的状态是**未知**，不是"没回退" |

**合计 44 / 50 个运行有回退证据；6 个无日志可查。**（**更正**：我先写过"38"，
那是只数了四棵树、漏掉旧树 `large_scale_benchmark` 的结果；全树清点为 44/50。）
即**凡是有日志可查的 SINTER3D 运行，无一例外**都在回退配置下产生。

**并且这是现行路径**：`rerun_sinter3d_stress.py:63,118` 与 `rerun_sinter3d_campaign.py:59,92`
**直接实例化 `SINTER3DAdapter`**——回退不是历史遗留，而是当前战役所用的那条路。
（对比：`UniSTAdapter` 在全库**只有它自己的类定义一处**，无任何调用方；
UniST 实际由专用 runner `rerun_unist_official.py` 驱动。）

### ⚠️ 但严重性有限：那份预处理的四步里，三步在这份数据上**本就不适用**

**本条先写成了"官方管线被跳过 = 结果被降级"。检查"它在这份数据上是否适用"之后，
我自己把它降级了**：

```
var_names = PC_1 … PC_50      uns['feature_space'] = 'PC'
X 取值    = [0.0258, 0.9841]  无负值（min-max 标度）
```

**这份 benchmark 跑的是 50 个主成分，不是基因表达计数。**

| 步骤 | 位置 | 在 50-PC 数据上 |
|---|---|---|
| 线粒体基因剔除 | `utils.py:18-21, 40-42` | **不适用**——没有基因，无从判 `mt-` |
| `select_hvgs` 分组选高变基因 | `utils.py:63-69, 124` | **不适用**——没有基因 |
| `normalize_total(1e4)` + `log1p` | `utils.py:122-123` | **不适用且有害**——PC 是 min-max 标度、非计数，归一化到总和 1e4 会**破坏**它 |
| 由 `slice_dist_micron` 重建 z 轴 | `utils.py:138-151` | **本来适用**——特征空间无关，**是唯一真正的损失** |

**所以回退更可能是"官方管线在此特征空间上无法运行"的合理处置，而不是静默降级。**
`KeyError` 是这种不匹配的症状。回退用 `spatial` + 逐片 z 构造了 `3D_coor`
作为 z 轴的替代——**有损，但不是"没有"**。

**因此不要"修好它再重跑"**：补上 `cell_type` 列会让 `preprocess` 跑通，
然后它会对 PC 特征做**计数专用的**归一化与 HVG 选择——那才会真的把结果弄坏。

### 影响

`utils.preprocess` 中被跳过的步骤（逐条对到函数行号，可复核）：

| 步骤 | 位置 |
|---|---|
| 线粒体基因剔除（ref 与 ST 两侧） | `utils.py:18-21, 40-42` |
| 官方 HVG 选择 `select_hvgs`，并把 ST 子集到 hvgs | `utils.py:63-69, 124` |
| ST 标准化 `sc.pp.normalize_total(target_sum=1e4)` + `log1p` | `utils.py:122-123` |
| 由 `slice_dist_micron` 驱动的 z 轴重建 | `utils.py:138-151` |

回退路径把这些一律跳过：`adata_st` 就是原始拼接矩阵，只补了 `obsm['count'] = X`
与 `library_size = sum(X)` 两个字段，然后直接进 `model_spatial`。
（注意：模型自身的 `normalize=100` 参数仍按官方配置传入，与本项不冲突——
**跳过的是 `utils.preprocess`，不是模型内建的那一步**。）

**不改变的事实**：模型/损失/训练步数（10000 步官方配置）未变；2026-08-29 的独立审计
已确认 36/36 指标可复算、无泄漏。**故本项不是"数字错了"，而是"跑的不是官方预处理管线"。**

### 与 track 归属的关系（D1）

`needs_scrna: true` 描述的是**已发表代码的接口**（`preprocess` 要求 `adata_ref_input` 位置参数），
属实。但本项目实际每次运行喂进去的"参考"是：
① **由 visible 片自身**构造的细胞类型均值（`sinter3d_adapter.py:243-250`），**不是外部 scRNA 图谱**；
② 且退化为**单一细胞类型**。
**即：按实际运行，SINTER3D 没有获得任何额外模态。**

### 待裁决

- **(a) 登记为偏差并披露**：结果保留，在 SINTER3D 出现的表格/正文注明"manual-prep 回退"。
- **(b) 修复后重跑**：给 `adata_basis.obs` 补 `cell_type` 列即可让官方 `preprocess` 跑通
  （构造函数类型标签**不需要标签列**，`adata_basis.obs.index` 本身就是类型名）——
  但**只能恢复预处理，不能恢复去卷积**（类型数仍为 1）。需 T_max + 用户批准。
- **(c) 先在一轮上试修**：验证 (b) 的数值影响，再决定是否全量重跑。

## 2026-09-23 · 新数据集（心脏 / Allen 官方两套）逐方法复核：四处适配，均为 math-neutral，**修复待实施**

背景：三套新数据集已摄入（**基因空间**，238 / 1122 个特征），六个 LOSO 轮的任务族已登记，
ReHo 已在三套上各跑通一格。复核方式为**逐行读代码 + 读三份新 h5ad 的结构**（未跑正式轮）。
四处发现按"会不会静默出错"排序——前两处会**静默**，不是崩。

| # | 方法 | 位置 | 问题 | 修法（math-neutral） | 状态 |
|---|---|---|---|---|---|
| 1 | PASTE-OT-Linear | `benchmark/scripts/rerun_baselines_stress.py:84-93` | KL 矩阵按 `(chunk=2048, nB, n_feat)` 广播、**同时存在两个临时量**；`n_feat` 从 50 涨到 238/1122 后单 worker 需 **306–612 GiB** → **三套新数据的全密度轮都会超限**（原先只当心脏有问题） | chunk 按**字节预算**定：`chunk = max(1, budget // (2*nB*n_feat*4))`；和式只在特征轴内求和，**数值不变** | **已实施 2026-09-23**（`rerun_baselines_stress.py`，`KL_CHUNK_BUDGET_BYTES=30 GiB`）。验证：M 在 chunk=1/7/64/299/300/1e6 下**逐位相同**；参考数据的最坏对仍取 2048 → 25.2 GiB，**已注册轮的分块与运行时长一律不变**；三套新数据落在 ~30 GiB/worker |
| 2 | NTF / DeepSpatial | 冻结的 `scripts/new_methods/ntf_pipeline.py:44`（`metadata()` 断言 `var_names == PC_1..PC_50`）与 `:64`（`read_rows()` 硬编码 `shape=(len(span),50)`），经适配层 `_new_methods_stress_common.py:157,174,346` 调用 | 基因空间**在输入构造阶段就 hard fail**；适配层 `_new_methods_stress_nfeat.py:87,96` 连变体的特征名也钉成 `PC_{i}` | **不改冻结件**：新建一个适配模块，宽度与特征名从任务规格 / `var_names` 派生（沿用 `_new_methods_stress_common.py` 本身的生成式手法） | **已实施 2026-09-23**（`_new_methods_stress_nfeat.py` 新增两处读取器锚点 + **存储布局分支**；`_new_methods_stress_common.reader_for()` 按数据集特征轴二选一；两个 runner 传该轮真实特征名）。验证：① 宽度等价校验 `validate_nfeat_variant.py` **PASS**（max\|Δ\| 5.4e-06 ≪ 1e-3）；② 三套新数据的输入**全部建成**（心脏 3,005,403×238 基因符号 / ABCA-1 2,794,712×1122 / ABCA-2 1,193,648×1122，皆有限值非负）；③ 参考数据同一轮重建与改动前**逐元素相同**（X/var/obs/query 全等） |
| 3 | SINTER3D | `benchmark/src/adapters/sinter3d_adapter.py:97,231` | 片身份写成 `str(int(z))`；ABCA-1/2 的片内 z 不是整数 → 约 10 个物理片被**合并成同一个身份**（模型的分片嵌入随之退化，**静默**） | 用**片标签本身**（`visible_slice_labels[i]`，由已算好的列表显式传入，不靠文件名解析） | **已实施 2026-09-23**。**但严重度经追码后下调，必须写清楚**：`obs['slice']` 在现行路径上不进入任何算术——模型只在一个地方读它（`model_{basic,spatial,complex}.py` 的 `use_type=='evaluate'` 分支），用途是拼出唯一的 obs 索引 `{index}-slice{value}`；训练分支用的是 enumerate 下标。所以合并不改任何数值。**参考数据同样有碰撞**（54 片 → 41 个不同整数，13 对相邻片同桶）——即这个缺陷**一直存在**，只是无害。修它的理由是"输入不该自我描述错误"，以及官方 `preprocess`（若将来修好）会自己重设该列 |
| 4 | isoST | `benchmark/src/adapters/isost_adapter.py:77` + 其生成脚本 `:371` | `delta_d=0.2` 是按参考数据的归一化尺度定的常数 → 心脏上变成**积分 1501 步** ODE、ABCA-1 上只有 **2 步** | `delta_d` 按**该数据集自己的片中位间距**缩放，锚定到参考数据（`REFERENCE_MEDIAN_Z_GAP = 0.928329`，实测 54 片逐片中位 z） | **已实施 2026-09-23**（`isost_adapter.py` 生成脚本内）。验证（无需跑 GPU）：参考数据 **53 对相邻片的 N_steps 逐一相同**（比值 1.0 → delta_d 仍为 0.2）；心脏 751 → 6 步、ABCA-1 **1 → 6 步**、ABCA-2 2 → 6 步 |

**另记两条边界（不是方法问题，但会让结论失真）：**

- 冻结的独立校验器 `scripts/new_methods/audit_rounds_gt.py:31-34,105` 同样钉住 `PC_50`，
  所以第 2 项的验证**不能只靠那一个校验器**，须另做参考数据集上的等价性对照。
- 心脏的 `ot.dist` 是**全量 float64** 距离矩阵（(84,173)² ≈ 56.6 GiB，且 C1/C2 并存再加临时
  ≈ 170 GiB），**没有 math-neutral 的拆法** → 心脏的 PASTE-OT-Linear 走 **d25 轮**
  （基准设计内的轮型，不是为跑通而造的捷径）。

**四处均已实施（2026-09-23）**，各自的边界验证见上表右列。**冻结件一律未改**
（`scripts/new_methods/{ntf,deepspatial}_pipeline.py`、`methods/` 下的第三方代码保持逐字节不变）：
NTF 那处靠**生成式副本**，PASTE/SINTER3D/isoST 改的是**本项目自己的适配层与 runner**。

**一处必须记录的运行环境事实（不是本次改动造成的）**：`_inputs/STRESS_A1_base` 的
`visible.h5ad` 在**当前环境下重建与 2026-09-14 记录的 sha256 不同**
（`6366e8ab…` → `7d214dee…`）。已用 `git stash` 归因：**改动前的代码给出同样的新哈希**，
所以这是**环境漂移**（V100 迁移后重建的 conda 环境里 h5ad 写入的容器层差异），不是本次改动的后果。
**内容逐元素相同**（X 矩阵、var_names、obs_names、query 四个数组全等，已实测），
而锚点门 `assert_matches_canonical` 比的是**预测内容的数值**、不是输入文件的字节，
所以门不受影响。**但任何将来想按 sha256 复核 09-14 那批输入的人要知道这一点。**

相关更正同时写进了 `benchmark/reports/CAMPAIGN_PLAN_newdatasets.md`
（§2 规模律缺 `n_feat` 项、§3 可行性表）。

## 2026-09-24 · 选卡改为受显式绑定约束（四处，math-neutral）

**事故（当天实测）**：oldgaps 车道的 `MISSING_54 M2_B3 × isoST` 格跑到了 **GPU 1**，
而队列给它的绑定是 **GPU 3**。证据：该格子进程 `_run_isost.py` 的环境里同时有
`GPU=3`（队列的变量）与 `CUDA_VISIBLE_DEVICES=1`（被覆盖后的值），nvidia-smi 也把它的
6.2 GiB 记在 GPU 1 上。这不是"队列没绑"，而是**绑了以后被 adapter 改掉**。

**根因**：`IsoSTAdapter._pick_free_gpu()` 与 `UniSTAdapter._pick_free_gpu()` 都在
`config["gpu"] is None` 时按 **`nvidia-smi memory.free` 全局最空的卡**挑一张，然后把
`CUDA_VISIBLE_DEVICES=<它挑的>` 写进子进程环境（`isost_adapter.py:_isost_subprocess_env`）。
队列只给了环境变量、没给 `--gpu`，于是"挑一张最空的"覆盖了"这张卡是分给你的"。
两个 UniST runner（`rerun_unist_official.pick_free_gpu`，被 `rerun_unist_stress` import；
以及 `rerun_unist_fixed.pick_free_gpu`）是同一形状。
2026-09-23 为 SINTER3D 修的是**队列侧导出环境变量**，对这类"adapter 自己挑卡"的路径无效
—— 这是同类缺陷的**第二个实例**。

**改动（4 处 + 1 个共用判定，零算法改动）**：
- `benchmark/src/adapters/base.py` 新增 `assigned_gpu_indices()` 与
  `pick_from_assigned_or_free()`：`CUDA_VISIBLE_DEVICES` 是**数字索引**时即为权威
  （多值则在其中挑最空的）；未设 / 空 / UUID-MIG 形式则**完全保持原行为**，
  所以独立运行（本次事故之外的用法）逐字节不受影响。
- `isost_adapter.py`、`unist_adapter.py`、`rerun_unist_official.py`、`rerun_unist_fixed.py`
  的挑卡函数改为先问 `assigned_gpu_indices()`。
- **验证**：`benchmark/tests/test_resource_scope.py` 新增一项，逐值钉住
  `3 → GPU3`（即使 GPU1 更空）、`3,5 → 其中较空者`、`"" → 原行为`。

**同时补上"事后核验"**（这类越界的隐蔽之处在于启动环境看起来是对的）：
`lib_resource_scope.sh` 新增 `rs_card_of_pid` / `rs_assert_card_discipline`，
两条队列在 cell 启动约 2 分钟后**核对该 cell 内的进程实际落在哪张卡**，越界即
state log 记明并停掉该格（卡纪律是红线，不是偏好）。

**教学**：绑定的"生效"必须在**子进程实际使用设备之后**核验，不能只看启动时的环境变量。

## 2026-09-24 · heart × ReHo 格的两个调度缺陷（零算法改动，A/B 实测数字中性）

**这不是"heart 上 ReHo 要 147 GiB"，而是这一格从没跑到 ReHo。** 该格在队列里是
`heart ReHo A1_gap1_anterior`，**既没有 `--no-ot` 也没有 `--cpu-nn`**，两次尝试各跑约 28 分钟后
被上限截停（105.71 / 146.96 GiB 且仍在爬升），我先前把它记为"待查的测量值、不许当墙"——
现在查清了，两个缺陷都是**命令配置**，不是方法的资源需求：

1. **缺 `--no-ot`** → `rerun_baselines_stress.py` 的 `main()` 是
   `run_paste_round()`（PASTE）**先于** `run_reho_round()`（ReHo），于是这一格整条命都花在
   **heart 的 PASTE-OT 阶段**上，而那个阶段在本战役里是**已登记的不可行格**
   （计划估 277 GiB > 250 GiB 上限，路线是 d25 密度轮）。
   **证据是它自己的产物树**：`..._heart_.../A1_gap1_anterior/PASTE-OT-Linear/run/` 里有
   19:17 写下的 `_run_ot.py`、`ot_lower_9_26.npz`、`ot_upper_9_26.npz` 和一直写到 19:44 截停
   时刻的 `resource_timeseries.csv`；而 `ReHo/` 目录**根本不存在** —— 连 `run_reho_round()`
   头两句就写的 `T_MAX.txt` 都没有。
2. **缺 `--cpu-nn`** → 战役用 `benchmark` conda 环境跑 ReHo/PASTE，该环境
   `torch 2.12.1+cpu`（`cuda_build None`、`cuda.is_available() False`），而默认 NN 路径是
   `nn_map_gpu`，于是必然抛 `AssertionError: Torch not compiled with CUDA enabled`
   （当天实测两次：abca2 的 A/B 首次运行、heart 这格本身，state.log 19:51:51）。
   反推：**已注册的 abca1/abca2 ReHo 格也不可能走 GPU 路径**（同一个环境没有 CUDA），
   它们的数字来自 cKDTree 路径 —— 所以 `--cpu-nn` 是**与已注册格对齐**，不是偏离。

**改动（仅 `benchmark/scripts/newdata_queue.sh` 的格表，属"仅调度、不影响数值"）**：
两条车道的 heart ReHo 行都改为 `106 --no-ot --cpu-nn`（claim 决定谁跑，因此两处必须一致）；
`M_max=106` 的依据是本数据集的**共享前缀**：heart NTF 实测 59.44 GiB（载入+表示，ReHo 不训练）
× 1.5 ≈ 89，上取到本数据集既有的 106 基准。abca1/abca2 的 ReHo 行保留其原样文本作为
"当时排了什么"的历史记录（它们已注册，车道不会重跑；注册产物才是事实来源）。

**验证（`benchmark/scripts/abtest_reho_no_ot.sh`，2026-09-24 实测）**：在 abca2 上把已注册的
ReHo 格用 `--no-ot --cpu-nn` 在**临时结果树 + 临时注册表**（`STRESS_REGISTRY` 改道，
不碰 `state/`、不覆盖任何已注册产物）重算一遍，与已注册产物逐字段比对：
**37 个指标字段全部一致**（pcc_global/pcc_gene_mean/rmse/cosine/ssim/moran 全家），
预测矩阵 `max|ΔX| = 0.000e+00`，查询坐标逐位相同，`rc=0`。
即：这两个标志在该格上**数字中性**，并**逐位复现**了已注册的 abca2 ReHo 数字。

**教学**：① 一格"内存爆了"先问**爆在哪个阶段** —— 跑过的阶段会留下文件，没跑的连开场文件都没有，
产物树比日志更可靠（本次日志因 stdout 块缓冲只剩三行，靠树才定的案）。
② 一个 flag 的默认值依赖环境能力（CUDA）时，那个能力必须**核对**而非假定；
同一战役里 PASTE 格早就显式带 `--cpu-nn`，ReHo 格却没带，说明"同类格默认一致"是错觉。
③ 判"硬件墙"之前先排除"调度缺陷"：前者要做计算重组（改数学等价性需 A/B），后者只改一行。

---

## 2026-09-30 · 三处**调度/记账**改动：零算法改动、零输入输出改动、零随机性改动

> 位置说明：这三处都改在**我们自己的战役代码**（`benchmark/scripts/`），
> **没有一处触及 `methods/` 下任何第三方方法**。登记在此是因为其中一处
> **变更了先前的资源申报值**，应可被检索到；三处均不改变任何方法的数学、
> 输入、输出或随机性，故**不需要 A/B 数值等价性验证**（对比 2026-09-24 的 ReHo 那条，
> 那条改了运行参数所以做了逐位 A/B）。

### 1. `catchup_queue.sh` 新增 `has_product()` —— 完成判据由"注册表有行"扩为"或产物已落盘"

* **问题**：注册是独立的周期步骤，格跑完后 `metrics.json` 已落盘而注册表尚未写行；
  这个空档里队列读 `registered()` 得到"未完成"并重启该格。`clear_stale_cell_dir.sh`
  **正确地**拒绝搬走有产物的目录，于是冻结管线 `mkdir(exist_ok=False)` 撞上刚写好的目录，
  **秒失败、峰值 ~1.4 GiB**，且新收据覆盖了成功那次的收据。
* **实测**（时间戳定因）：abca1 A2/A3/B1 的**产物**写于 23:42/23:36/23:23，
  而**失败尝试**始于 23:58:21/23:57:53/23:38:37 —— 尝试**晚于**产物。
* **改动**：完成 = `registered()` **或** `has_product()`。后者**只认 `metrics.json`、
  绝不认空目录** —— 空目录正是被内存闸切断的尝试留下的形状，那些必须保持可重试。
* **验证**：四例判定（abca1 A2→已完成、B2/B3→未完成、abca2 B2→已完成）全部符合预期；
  `bash -n` 通过。
* **算法影响**：无。仅"何时不重复启动一个已完成的格"。

### 2. `probe_cell.sh` 的车道声明名与方法名折叠规则统一

* **问题**：心跳按方法原名写申报（`probe_abca1_PASTE-OT-Linear_…`），
  而窗口发布者按 **unit 名**写（`-`→`_`，`probe_abca1_PASTE_OT_Linear_…`）——
  同一车道**两个文件、被算两份**；发布者的 GC 每 60 s 删前者，心跳又把它写回来。
  实测该车道 `25 + 60 = 85` 对真值 `60`。
* **改动**：`LANE` 改用与 unit 名**逐字相同**的折叠规则。
* **验证**：四个方法名（`PASTE-OT-Linear`/`isoST`/`DeepSpatial`/`SINTER3D`）折叠后与发布者
  写出的名字逐字一致；`bash -n` 通过。
* **算法影响**：无。仅内存窗口记账的名字。

### 3. `catchup_queue.sh`：abca1 × DeepSpatial 的申报 `M_max` 由 55 → 130

* **依据**：55 = 36.6 × 1.5，系按 **ABCA-2 尺度**推导，罩不住 ABCA-1。各 scope 自己的收据实测：
  ABCA-2 三个成功轮峰值 `38.74 / 42.53 / 41.13 GiB`（1,168,219 可见 cells，62 片）；
  ABCA-1 的 B2/B3 在 `50.22 / 49.61 GiB` 被软闸（0.9×55=49.5）切断（2,769,305 cells，141 片）。
  后续实测证明**真峰值压在软闸附近**（B2 44.12 / B3 47.82 一次即过），
  即那是**掷硬币而非确定的墙**；抬高申报值消除这枚硬币。
  按每可见 cell ~36.9 KiB 外推 ≈102 GiB、按每可见片外推 ≈94 GiB，取 130（软闸 117 / 硬闸 140）。
* **这不是缩水**：申报量与硬闸都是**上调**，官方全量点、官方 epochs、官方模型尺寸一律未动。
* **验证**：ABCA-1 × DeepSpatial 随后达 **6/6**（B2 pcc 0.172079、B3 0.168443，均 rc=0）。
* **算法影响**：无。仅 `MemoryMax` 这个守卫值。

### 附：本次**没有**改的两处（已诊断、未动手，等用户裁决）

* `_stress_common.register_round()` 的**幂等化**（现为无条件 append，且 `campaign_run_id()`
  把当天日期嵌进 run_id，故跨天重登记必产生重复行）。详见 `state/registry_duplicate_rows_note_20260930.md`。
* `rerun_unist_official.py` 驱动模板每阶段开跑前**无条件 `rmtree`** 自己的输出目录，
  使 checkpoint 无法用于续跑；是否改造以支持 Lightning resume 会**改变运行协议**，须用户裁决。

---

## 2026-09-30 · SpatialZ 适配器：α 取向回归修复（**缺陷修复，非算法改动**）

`benchmark/src/adapters/spatialz_adapter.py`

### 问题：虚拟切片被造在了错误的深度上

官方约定是 **adata1 = ABOVE 片**，两处可证：

* `methods/SpatialZ/SpatialZ.py:21` —— 默认参数名 `def Generate_spatialz(adata1, adata2,
  adata1_id='above', adata2_id='below', ...)`
* `methods/SpatialZ/SpatialZ.py:397` —— `Generate_multiple_spatialz` 里 `alpha = 1 - i/num_sim`
  随 i 递减，而生成的切片按 i 递增排列，即切片从 adata1 走到 adata2
  ⇒ **alpha 是 adata1 的权重**

适配器算出的 `alpha = (hz - z_lower)/(z_upper - z_lower)` 是**上片**的权重，
却把**下片**当作 adata1 传入。于是虚拟切片落在
`alpha*z_lower + (1-alpha)*z_upper`，而真值在 `alpha*z_upper + (1-alpha)*z_lower`
（后者是 alpha 定义本身的恒等式）。

**溯源**：被删除的 P1 原型（`91c24e0`，2026-08-04）用的是
`Generate_spatialz(a_above, a_below, adata1_id=f'z{z_above}', ...)` —— 与官方一致。
v2 重写（`b26420d`，2026-08-11）把两片顺序对调，却未同步调整 alpha 的取向。
P1 的 alpha 系外部传入，故**不存在补偿性改动**。**这是一次回归。**

### 改动：两行

生成脚本内 `a1` / `a2` 的读入顺序与 `adata1_id` / `adata2_id` 对调，使 **adata1 = 上片**。
`alpha` 的算法不变。文件顶部与调用处均加了说明注释。

### 实测代价

16 张隐藏切片（A1_gap1_anterior ×1、M2_A1 ×2、T3_A1_s2 ×13）逐张测量，
深度偏差占本切片上下间距的比例：

| | 修复前 | 修复后 |
|---|---|---|
| 偏差 / 切片间距 | **0.04% – 52.7%**（中位约 11%） | **0.004% – 0.15%** |

残差与子样均值的标准误之比 ≤ 0.76，即残差完全可被子样抽样噪声解释。
两个 α≈0.5 的切片两种取向本就等价，正确地显示"无改善"。

### 验证（两道闸门，均通过）

1. **Phase 0（离线，16/16）**：`PRE/figgen/geom/FINDINGS.md`，判据为"残差/间距 < 1% 且
   残差/标准误 < 3"。
2. **Phase 1（端到端，走真实适配器）**：A1_gap1_anterior，`max_cells_per_slice=1500`，
   conda 子进程 + 官方 `Generate_spatialz`。
   生成 z 均值 **25.491**，真值 z **25.491**（偏差 0.0000）；旧取向预期值 24.015（距离 1.4754）。
   `PRE/figgen/verify_adapter_fix.py`、结果 `PRE/figgen/geom/verify/verify_result.json`。

### 影响范围

**2026-08-11 起的所有 SpatialZ 结果**（`RERUN20260820` / `stress_20260831` /
`multimissing_20260907` / 新数据集各族）均带此问题。已注册格需按新世代重跑，
**不原地覆盖**。手稿 §5.1(i) 对 SpatialZ 的表述须待重跑后一并改述。

### 定性

**对齐官方约定的缺陷修复，不是算法改动。** `lr` / `num_projections` / `nb_iter_max`
等超参未动；官方 `SpatialZ.py` 一行未改（`methods/` 零改动）。

**未解决（独立问题，勿与此条混淆）**：即便取向修正后，SWD 坐标优化在红抽投影下
仍几乎不移动点云（净位移极小）。那是另一个问题，见
`PRE/figgen/geom/FINDINGS.md` 的"一处需要更正的归因"。

---

## SE 集群迁移（2026-10-18）— 仅可移植性，无算法改动

**范围**：把项目从 8×V100 工作站搬到 Slurm 集群；**没有**改任何方法的数学、超参、
输入输出定义或评价口径。`methods/` 下第三方代码本次**零改动**（唯一相关动作是
`methods/` 内 9 个数据软链重新指向本机 `$DATA005`）。

| 文件 / 组件 | 改动 | 理由 | 验证 |
|---|---|---|---|
| 182 个代码/配置/脚本/文档 | 旧绝对路径前缀替换（275 处） | 旧机 `/data/home/xiepeiyan/...` 在本机不存在 | 182 文件清单由 `relocate_paths.py` 输出；`pytest` 全绿 |
| `benchmark/src/utils/hpc.py`（新） | `cuda_home()`、`allocated_cpus()`、`allocated_gpus()`、`job_memory_gib()`、`subprocess_env()` | 集群的 CUDA 在 module 里、CPU/内存来自分配 | 环境自检 + 端到端 |
| `benchmark/src/adapters/isost_adapter.py` | `ISOST_CUDA_HOME` 由硬编码 `/usr/local/cuda` 改为 `hpc.cuda_home()` | 本机 `/usr/local/cuda` 是空壳，nvcc 在 `/soft/devtools/cuda/cuda-12.9` | isoST keops JIT 自检 |
| `benchmark/scripts/run_method.py` | `--threads` 默认 = `SLURM_CPUS_PER_TASK`（无分配时仍 8） | 避免 8 核作业里按 16 线程跑 | 单测 + e2e |
| `benchmark/scripts/rerun_baselines_{fixed,stress}.py` | OT 线程 `16` → `XPY_OT_THREADS` / `min(16, SLURM_CPUS_PER_TASK)` | 线程数只影响浮点归约顺序（已登记噪声类），旧值在无分配时保留 | e2e ReHo PCC 与注册值一致 |
| 同上（`fixed`） | `SPATEO_ENV` `"spateo"` → `"benchmark"` | 本机无 `spateo` env；冻结 OT 脚本只 import POT+numpy，`benchmark` 有 POT 0.9.7.post1（stress runner 2026-08-20 已作同样切换） | 未改 OT 参数（α=0.1, ε=0.05, KL, square_loss） |
| `benchmark/scripts/pick_free_card.sh` | Slurm 下数字 `CUDA_VISIBLE_DEVICES` 直接返回 | 作业内的 nvidia-smi 仍列全卡，自由选卡会越界（invalid device ordinal） | 代码审查 + `test_resource_scope` 现有断言 |
| `benchmark/scripts/lib_resource_scope.sh` | 新增 Slurm 分支：上限取 `SLURM_MEM_PER_NODE`，按进程树 RSS 采样、90% 软中止，无 systemd；记录 schema 不变 | 计算节点无 user systemd，集群用 `--mem` | 本地以 `SLURM_JOB_ID` 模拟运行，产出合法 `*.resource.json` |
| `benchmark/scripts/{newdata,oldgaps}_queue.sh` | Slurm 下 `GPU` 默认 0 | 作业内只有一张可见卡 | 语法检查 |
| `benchmark/scripts/newdata_campaign.sh`、`qualification_matrix.py`、`run_new_methods_parity.py` | NTF/DeepSpatial 从 venv 路径改为 conda env（`envs/ntf`、`envs/deepspatial`） | 本机系统 Python 3.9，旧 venv 建不出来；改为同版本 conda 解释器 | 环境 import + 重新生成 spec |
| `benchmark/method_specs/{NTF,DeepSpatial}.yaml` | 由 `qualification_matrix.py --write-specs` 重生成（`kind: conda`） | 同上；生成器是唯一写入者 | 生成器输出 24 行矩阵 |
| `benchmark/slurm/*`（新） | Slurm 提交/包装/自检脚本与 README | 用调度器表达"一卡一任务 / T_max / M_max" | e2e + 自检 |
| `benchmark/envs/{ntf,deepspatial}.yaml` | `kind: venv → conda`、路径更新，保留历史注记 | 与实况一致 | YAML 解析 |
| `agent_control/EXPERIMENT_RESOURCE_POLICY.md` | 新增 §4b：五条原则的 Slurm 等价物 + 新硬件解除的旧墙（不自动撤销 N/A） | 原则不变，执行方式变 | 文档 |

**环境 pin**：全部取自 `benchmark/envs/*.yaml` 与 `*_freeze_20260911.txt`；torch 线为
cu126（覆盖 sm_70/80/86），`ntf`/`deepspatial` 保持 torch 2.6.0+cu126。tiny-cuda-nn 编译
`TCNN_CUDA_ARCHITECTURES=80;86`（覆盖 A100/A40；V100 兜底需重编，见下）。

**已知限制**：
1. `ntf` 的 tiny-cuda-nn 目前只编了 sm_80/sm_86；要用 `v100` 分区跑 NTF 必须重编 sm_70。
2. `catchup_queue.sh` 直接调 `systemd-run`，集群上不可用（其余两个队列可，见 README）。
3. `github.com:443` 经 relay 被重置；tiny-cuda-nn 的 submodule 源码改由 codeload tarball
   提供并入库到 `data/thirdparty/tiny-cuda-nn-src/tarballs/`（不入 git，数据区）。
4. 新硬件的 T_max 锚点未实测；沿用旧机 `expected_wall_sec` 只是**上限来源**，不是性能承诺。

## 2026-10-18 · SpatialZ：lr 参数化 + 冻结 lr=40（**参数变更，登记**）

**改动**：`benchmark/src/adapters/spatialz_adapter.py` 从 config 读取 `lr`（默认 1.0，
保持历史行为）与 `nb_iter_max`（默认 1000）；`rerun_spatialz_stress.py` /
`rerun_spatialz_fixed.py` 新增 `--lr` / `SPATIALZ_LR`，并把 `lr` 写进 `metrics.json`
的 extra。**官方 `methods/SpatialZ/SpatialZ.py` 零改动**，更新规则仍是
`Coor_torch -= Coor_torch.grad * lr`。

**为什么必须改**：2026-09-30 的 α 取向修复只修正了虚拟片的**初始化深度**；
`lr=1.0` 下首步位移仅 span 的 2.46e-4，坐标优化基本没有运行。用户 2026-09-28 的判据是
"首步位移 ≈ 点云尺度的 1%"，故 lr 必须按坐标尺度确定，不能沿用 1.0。

**冻结值**：`lr=40`。在 `A1_gap1_anterior`（修正取向、投影每步重抽、120 步、80 投影、
seed 42）实测：

| lr | 首步/span | 净位移/span | loss 变化 | Chamfer 3D |
|---|---|---|---|---|
| 1 | 2.46e-4 | 0.000 | -1.49% | 1.6301 |
| 10 | 2.46e-3 | 0.001 | -0.89% | 1.6253 |
| 20 | 4.92e-3 | 0.002 | -0.22% | 1.6202 |
| 30 | 7.38e-3 | 0.003 | +0.44% | 1.6151 |
| **40** | **9.84e-3** | **0.004** | **+1.09%** | **1.6103** |

初始化 Chamfer 3D = **1.6306**。lr=40 是扫描中**最大且仍稳定**的值：首步达到 1% 判据、
loss 转正、几何单调改善；lr 继续增大未在本次扫描中测试，故不采用。**净位移仍只有
0.4% span**——这说明 SWD 的最优点就在初始化附近，α 修复才是主要增益；论文须如实写成
"坐标优化仅做局部细化"，不得写成"优化器显著移动了点云"。

**证据**：`benchmark/reports/SPATIALZ_LR_PROBE_20261018.{json,log}`；M2_A1 + T3_A1_s2
的确认探针（`benchmark/reports/lr_corrected_confirm.json`）已完成，lr=40 在
M2_A1 上首步/净位移 = 2.24%/1.1%、Chamfer 3D 1.8109→1.7345（-4.2%）；在 T3_A1_s2 上
2.93%/1.6%、2.1501→2.0338（-5.4%）。lr=20 的净位移只有 0.5–0.8%、几何改善更小；故仍取
**lr=40**。首步比 A1 的 1% 大，是因为局部梯度尺度不同——判据应读作"同量级"，不是严格等式。
**旧结果不删除、不覆盖**；新世代为 `ALPHA20261018`。

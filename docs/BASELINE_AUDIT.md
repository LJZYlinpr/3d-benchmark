# 3D ST Reconstruction Benchmark — 三个自建 Baseline 严格审计

Date: 2026-10-22  
Scope: MERFISH ABCA-3 `.006–.010`, 5 slices, 50 µm bins, 7 scenarios `M1a–M3a`, fixed-coordinate expression prediction.  
Audit root: `/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/audit_3baselines_20261022`

Official predictions/metrics were **not modified**. Audit controls are under `audit_controls/` and `audit_3baselines_20261022/controls/`.

## 0. Executive verdict

- **No target-expression leakage was found in Ours-EncMTP, PASTE-OT-Linear or ReHo.** Target-X poisoning/zeroing leaves all three baselines’ predictions bitwise unchanged.
- **Ours-EncMTP high score is not explained by target-expression leakage.** Remaining issues are target-coordinate oracle input, a relative-offset unit bug, visible-only model selection, and incomplete provenance.
- **PASTE-OT-Linear has a material M2b bracket bug**: M2b uses 1→5 and ignores visible section 3; its predictions for the 49,432 target rows are bitwise identical to the corresponding M3a rows. Its PASTE/OT configuration is also not the reference config (PPA, 6,000 anchors, no expansion, 264 non-convergence warnings).
- **ReHo is leakage-clean but is nearest-visible-slice holding, not upstream preceding-slice ReHo.** 5/12 target-section source assignments differ from upstream. It is reproducible and otherwise correctly implemented.
- The current final table is **not ready for paper submission** until the P0 items below are fixed/rerun.

## A. 总审计表

| 检查项 | Ours-EncMTP | PASTE-OT-Linear | ReHo |
|---|---|---|---|
| Target expression leakage | **PASS** — target X never indexed; target-X poison/zero control max diff 0.0 | **PASS** — only `bins["X"][lo_rows/hi_rows]` visible; target-X zero control 0.0 | **PASS** — only `bins["X"][sb]` source; target-X poison 7/7 identical |
| Upstream preprocessing leakage | **PASS (caveat)** — `bins_all` contains target X, but all model X accesses use `vr` visible; `target_gt` whitelist excludes `gt` | **PASS (caveat)** — same canonical file; no target expression in KL/OT; target `tg` geometry only | **PASS (caveat)** — same canonical file; source section always visible; target X never read |
| Target coordinate compliance | **PASS (oracle disclosure required)** — uses target x/y/z and bin geometry only; fixed-coordinate allows it | **PASS** — target x/y/z query geometry only; no target expression | **PASS** — target `(ix,iy)`/x/y geometry; no target expression |
| Algorithm implementation correctness | **PARTIAL/FAIL (model defect)** — contract, params, kNN/MTP, loss and steps match; relative-offset units are wrong (nm offsets divided by 50 µm), degenerating xy Fourier features | **FAIL (partial)** — KL/FGW and z-linear formula are correct, but solver is PPA not PGD, 6,000 vs 20,000 anchors, no nearest-anchor expansion, 264 Sinkhorn non-convergence warnings; **M2b bracket bug** | **PASS vs its own spec; FAIL vs upstream ReHo** — nearest-visible rule, not preceding; same-bin/xy fallback correct |
| Input fairness | **PASS (qualified)** — same visible/target/gene/preprocessing; target coordinates shared; 2.3M params and MTP are native design | **PASS (qualified, under-budget)** — same inputs; 6,000 anchors/no expansion makes it a weak PASTE proxy | **PASS** — same inputs; no training/tuning; CPU-only; label is the issue |
| Tuning fairness | **UNCERTAIN** — 5% visible-only validation, MSE-only checkpoint selection, 3,000 steps per scenario; other methods’ comparable budget not audited | **PASS** — no trainable parameters; fixed deterministic config; anchor budget is a quality handicap, not target tuning | **PASS** — no training/tuning/RNG |
| Reproducibility | **FAIL (provenance)** — only M3a has 3,000-step metrics; M1a–M2c came from a concurrent multi-run; no checkpoint, env lock or per-scenario seed; 4.6 old-vs-rerun diff is explained by a pre-fix inference-coordinate bug | **FAIL** — final files overwritten by multiple runs; no OT plan/error archive; `niter=None`; non-convergence warnings only in stderr; stale `RUNNER_B_STATUS` NameError | **PASS** — deterministic in-process/cross-process; current code reproduces official `.npz` exactly; 7/7 contract pass |
| 是否需要重跑 | **YES** — all 7 in one isolated process; fix unit bug if intended | **YES** — M2b mandatory; full faithful PASTE config if the label is kept | **No rerun if renamed to ReHo-nearest; rerun (~92 s CPU) if upstream ReHo is required** |

## B. 必须解决的问题

### P0 — 会影响结果有效性

1. **PASTE M2b bracket bug（P0）**
   - Code: `run_paste_heart.py:184-186`
   - `lo=max(visible < target[0])`, `hi=min(visible > target[-1])`; M2b visible `{1,3,5}`, target `{2,4}` → code uses 1→5 and ignores visible 3.
   - Correct: `{2}: 1→3`, `{4}: 3→5`.
   - Evidence: all 49,432 M2b target rows are bitwise identical to the corresponding M3a rows (`max_abs_diff=0.0`); same OT metadata. M2b is not an independent k=2 measurement for PASTE.
   - Action: rerun M2b with per-block brackets; regenerate export/metrics/table.

2. **PASTE-OT-Linear is not a faithful PASTE reproduction（P0 for method claim）**
   - Code uses entropic FGW + **PPA** (`run_paste_heart.py:62-67`), not official PASTE’s non-entropic Frank–Wolfe/`ot.optim.cg`.
   - `max_iter=100` (reference 1000), `tol=1e-7` (reference 1e-9), **6,000 anchors** (reference 20,000), and **no nearest-anchor expansion**.
   - Official 7 solves produced **264 “Sinkhorn did not converge” warnings**; `niter=None`, column marginal error 5e-7–1e-6 > inner tolerance.
   - MERFISH reference PGD at 20k anchors produced nonconstant predictions (map median 29.9 µm); the “PGD collapses on 1122 genes” claim is supported only by heart 238-gene evidence, not this dataset.
   - Action: either restore reference config (PGD/full anchors/expansion) or relabel as **PASTE-inspired OT-linear** and state the deviation.

3. **Ours-EncMTP final provenance is mixed（P0 for paper）**
   - `compute_metrics.json` covers only `M3a` (`n_scenarios=1`); M1a–M2c have no 3,000-step per-scenario metrics.
   - M1a–M2c were written during concurrent runs racing on the fixed `.tmp` filename; M3a was a standalone run.
   - No checkpoint/state_dict, environment lock, per-scenario seed, or deterministic flags.
   - Action: rerun all 7 scenarios in one isolated process (or explicit per-scenario seeds), save per-scenario metrics, state_dict, code/env hashes, and prediction hashes.

### P1 — 会影响论文可信度

4. **Ours relative-offset unit defect**
   - Coordinates are µm, multiplied by 1000 to nm at `run_encmtp_heart.py:182,218`, but relative offsets are divided by `bs_um=50` at `:196,209`.
   - On exact 50 µm lattice offsets, 3-level Fourier features collapse to sin≈0/cos≈1; xy relative geometry is effectively degenerate.
   - Not leakage; it weakens the intended model. Fix `bs_um` to 50,000 nm or remove the `*1000` for relative offsets; rerun if reporting the intended relative-geometry model.

5. **ReHo name/rule mismatch**
   - Code `run_reho_heart.py:58-59` selects **nearest visible z**, tie → lower z.
   - Upstream `Workspace/merfish_test/ReHo/run_reho.py:4-12,282` defines **preceding/lower-z visible slice**.
   - 5/12 target sections differ from upstream; 7/12 differ from preceding-label.
   - Action: rename current method **ReHo-nearest / nearest-slice holding** and disclose 5/12 divergence, or change source selection to preceding and rerun (~92 s CPU).

6. **Ours model-selection asymmetry**
   - Ours uses 5% visible bins for MSE-only early stopping; PASTE/ReHo have no learned parameters. This is not leakage, but the paper must state it.
   - Other compared methods’ validation/tuning budgets were not audited; do not claim fully equal tuning.

7. **PASTE under-budget / non-convergence**
   - 6,000 anchors (24–26%) vs reference 20,000 (79–86%); no anchor expansion means ~75% of each visible slice is unused.
   - 264 Sinkhorn warnings; no abort/convergence record.
   - This makes PASTE a weak straw man; restore budget or label clearly.

8. **Target-coordinate oracle disclosure**
   - All three baselines use target x/y/z (and target bin geometry) while excluding target expression.
   - This is allowed by the fixed-coordinate protocol, but must be stated explicitly in the paper; it is not a deployable missing-slice setting.

### P2 — 建议完善

9. Make target-expression isolation structural: feed visible-only X or NaN target rows into runners, rather than relying on row indexing in a file that contains target X.
10. Use a unique temp filename in `save_pred` (include PID) to prevent concurrent-run races.
11. Regenerate `input_audit.csv`/`audit_summary.json` for all 10 methods; current audit is stale NTF/Spateo-only.
12. Fix stale `RUNNER_B_STATUS.json` PASTE `NameError`; save code/env hashes and per-run IDs.
13. Preserve superseded prediction files or at least hashes when a run is overwritten.

## C. 关键证据

### C.1 Common data-path facts

- `_rb_common.py:149-161` `load_target_gt()` default keys are only:
  `("bin_id","section","x","y","scenario","row_index")`; `gt` is **not** loaded.
- `canonical/target_gt.npz` does contain a `gt` key, but no baseline runner loads it.
- `bins_all.h5ad` does contain target-section X (122,669 × 1,122), but all three runners index expression only through visible-section rows.
- Scenarios are disjoint: no `visible ∩ target` overlap in any of M1a–M3a.
- `target_gt` keys: `gt,row_index,scenario,section,bin_id,x,y,z,genes`; no `X` key.

### C.2 Ours-EncMTP

- Code: `missing_combos_50um/all_methods/methods/Ours-EncMTP/run_encmtp_heart.py`.
- Visible-only training/normalization:
  - `:151` `X = bins["X"][vr]`
  - `:179-181` mean/std/Z from that X
  - `:260` `yb = Z[bi]`
  - `:267` aux target `Z[ci[ar,pos]]` (visible pool)
  - `:172-176` target context uses `slice_pts` from visible only.
- Target geometry:
  - `:153` target x/y/z for token coordinates
  - `:218,286-287` target token coords for inference
  - target `bin_id` only echoed for alignment.
- Model: 2,335,530 parameters (M1a), 2,335,498 (M3a); 3,000 steps × batch 1,024; 3-NN per visible slice; 2 cross-attention blocks; K=4 MTP; 5% visible validation, MSE-only selection.
- Target-X control: `audit_3baselines_20261022/controls/encmtp_targetX_poison_m1a.py` and `.result.json` → target-section X poisoned with NaN, 2 CPU steps, same seed: `max_abs_diff=0.0`, bitwise equal.
- Known zero control also `max_abs_diff=0.0`.
- Reproducibility:
  - Code mtime 19:32; M1a–M2c predictions 19:42–19:46; M3a 20:03.
  - `compute_metrics.json`: `n_scenarios=1`, only M3a.
  - No checkpoint; one global seed before scenario loop; fixed `.tmp` race; no env lock/deterministic flags.
  - Old-vs-rerun 4.6 is consistent with pre-fix inference coordinate bug (old line 284 lacked `coords_tok`; current `:286-287` uses `coords_tok=g_q_coords`); training/val MSE identical (M1a 0.23796). Not target leakage; exact old file overwritten.
- Model defect: nm/µm relative scale (`:182`, `:196`, `:209`).

### C.3 PASTE-OT-Linear

- Code: `missing_combos_50um/all_methods/methods/PASTE-OT-Linear/run_paste_heart.py`.
- Leakage:
  - `:195-196` only `bins["X"][lo_rows]`, `bins["X"][hi_rows]`; `lo/hi` from `vis`.
  - `load_target_gt` excludes `gt`; target geometry only.
  - Target-X zero control: `missing_combos_50um/audit_controls/paste_zero` → max_abs_diff 0.0.
- OT/math:
  - KL cost `:34-42`; FGW `:66-67` with `square_loss`, `alpha=0.1`, `epsilon=0.05`, uniform marginals.
  - Solver PPA, `max_iter=100`, `tol=1e-7`; 6,000 anchors; no expansion.
  - Sinkhorn warnings: 264 total (38,15,72,15,30,53,41 per solve); `niter=None`; colsum_err 5e-7–1e-6.
- z-linear:
  - `:110-122`; `tt=(z[s]-z[lo])/(z[hi]-z[lo])`, `wl=1-t`, `wu=t`; positions and expressions symmetric; physical z µm; PASS.
- M2b:
  - `:184-186` one bracket per scenario; visible 3 ignored.
  - M2b 49,432 rows bitwise identical to corresponding M3a rows; OT meta identical.
- Fairness:
  - Same target rows/gene set; 6,000 anchors (24–26% of each visible slice), no expansion, max_iter 100; under-budget vs reference 20,000/1000.
- Repro:
  - Code mtime 19:01; final predictions 20:15–20:42; multiple runs (OOM, single, full, final); earlier predictions overwritten; no plan/error archive; TF32 enabled.
  - `RUNNER_B_STATUS` stale `NameError` for PASTE.

### C.4 ReHo

- Code: `missing_combos_50um/all_methods/methods/ReHo/run_reho_heart.py`.
- Rule:
  - `:58-59` `order = sorted(vis, key=(abs(z_src-z_tgt), z_src))`; source = order[0] → **nearest visible**, tie lower z.
  - Upstream `Workspace/merfish_test/ReHo/run_reho.py:4-12,282` is **preceding/lower-z**, not nearest.
  - 5/12 target-section assignments differ from upstream (M1b→3, M2a→2, M2c→3, M3a→2/3).
- Leakage:
  - only `bins["X"][sb]` (`:68,77`), `sb` source visible; `tg` only section/bin + echo.
  - target-X NaN poison: 7/7 bitwise identical; all 298,876 predicted rows are byte-exact copies of a source row.
- Mapping:
  - same `(ix,iy)` on global lattice `x=50ix−1025`, `y=50iy+675`; unique; residual <0.0004 µm.
  - fallback xy-nearest in same source for 9.51% rows; mean 83–171 µm, max 791 µm; no z interpolation.
- Repro:
  - deterministic in-process/cross-process; code 18:47, predictions 18:48–18:49; 7/7 contract pass; CPU-only; no RNG.

## D. 最终建议

1. **三个 Baseline 当前结果是否可信？**
   - **Leakage 层面可信**：三者均未读取 target expression；target-X 置零/NaN 对照全部为 0 差异。
   - **结果表层面尚未完全可信**：PASTE M2b 是 M3a 的复制；PASTE 配置不是 reference；Ours provenance 混合；ReHo 名称与 upstream 不一致。

2. **Ours-EncMTP 高分能否排除明显 target-expression leakage？**
   - **可以排除**：代码路径、`load_target_gt` whitelist、可见-only normalization/context/MTP、target-X poison 对照均支持无表达泄漏。
   - **不能排除的 confound**：target coordinates oracle、relative-offset unit defect、5% visible validation、per-scenario training、provenance 混合。

3. **PASTE-OT-Linear 和 ReHo 是否被正确实现？**
   - PASTE：z-linear 与 KL/FGW 公式正确，但 **不是 faithful PASTE**；PPA/6k anchors/no expansion/non-convergence；**M2b 实现错误**。
   - ReHo：同一 `(ix,iy)` 复制与 xy fallback 正确；**规则是 nearest，不是 upstream preceding**；当前结果只能作为 ReHo-nearest 变体。

4. **三个 Baseline 与其他方法比较是否公平？**
   - 输入数据、visible/target、gene set、target coordinates 基本一致；target coordinates 是 fixed-coordinate 共同信息，不是作弊。
   - **不完全公平**：PASTE 被 6k anchors/no expansion 明显削弱；Ours 有 5% visible validation/early stopping；ReHo 规则与 upstream 不同；其他方法的 tuning budget 尚未审计。
   - 这些是 **native design / budget / label** 差异，不是 target-expression leakage。

5. **哪些实验必须重跑？**
   - **Ours**：7/7 一次隔离运行，修复 relative-offset 单位，保存 per-scenario metrics/checkpoint/env/hash。
   - **PASTE**：M2b 必须重跑；若保留 PASTE 名称，重跑 faithful config（PGD/20k anchors/expansion）或改名为 PASTE-inspired。
   - **ReHo**：当前结果可保留但必须改名；若表格声称 upstream ReHo，按 preceding-z 规则重跑（约 92 s CPU）。
   - 重跑后重新导出 `predictions_all` 并重算 `metrics_v1_all`。

6. **正式论文还缺什么最关键的可复现性证据？**
   - 冻结代码 commit/hash、environment lock（torch/CUDA/cuDNN/numpy/scipy/POT）、per-scenario seed。
   - 每个 scenario 的完整 config、metrics、checkpoint/state_dict（Ours）。
   - 预测文件 SHA256、OT plan/`niter`/`err`/convergence 记录（PASTE）。
   - 完整的 10 方法 input audit（当前 `input_audit.csv` 仍是 NTF/Spateo-only）。
   - target-expression isolation guard（visible-only X 或 NaN target rows）和 target-X negative control 作为固定审计项。

## Artifacts

- `audit_3baselines_20261022/sections/ours.md`
- `audit_3baselines_20261022/sections/paste.md`
- `audit_3baselines_20261022/sections/reho.md`
- `audit_3baselines_20261022/controls/encmtp_targetX_poison_m1a.py` + `.result.json`
- `audit_3baselines_20261022/controls/paste_checks.py` + `.json`
- `audit_3baselines_20261022/scripts/reho_audit.py`
- `missing_combos_50um/audit_controls/paste_zero/` (PASTE target-X zero control)
- Official predictions: `missing_combos_50um/all_methods/predictions/<M1a..M3a>/<method>.npz`
- Official metrics: `missing_combos_50um/metrics_v1_all/` and `.../metrics_v1_all_final/`

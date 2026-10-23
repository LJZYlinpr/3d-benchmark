#!/usr/bin/env python
"""Phase 3-4: core 10 metrics + spatial P/R + diagnostics for 7 combos x 2 methods x 18 modules.

Reads frozen thresholds from metrics_v1/metric_config.yaml.  Never reads target
expression for anything except scoring.

Writes:
  metrics_core.csv         (combo x method x module x scope, 10 core metrics)
  metrics_module.csv       (module x method summary across combos)
  metrics_spatial_pr.csv   (hotspot/boundary TP-FP-FN, AP/PR-AUC, threshold sensitivity)
  metrics_diagnostics.csv  (long format: mean ratio, zero fractions, reliability, bootstrap, ...)
  core_summary.json        (for status.json)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlib as M

CORE_COLS = [
    "gene_pcc_median", "rmse", "module_pcc", "ms_ssim", "var_ratio_log2",
    "abs_delta_moran_i", "hotspot_recall", "hotspot_precision",
    "boundary_recall_d0", "boundary_precision_d0",
]


def load_cfg():
    with open(os.path.join(HERE, "metric_config.yaml")) as f:
        return yaml.safe_load(f)


def safe_pcc(a, b):
    return M.pcc_vec(a, b)


def compute_threshold_maps(cfg, modules):
    hs = {m["name"]: cfg["modules"]["hotspot_thresholds_reference"][m["name"]]
          for m in modules}
    bd = {m["name"]: cfg["modules"]["boundary_thresholds_reference"][m["name"]]
          for m in modules}
    return hs, bd


def split_half_reliability(Xg, sel, gene_idx, ref, n_splits=5, seed=0):
    """GT-only split-half reliability of a module map (gene split)."""
    idx = np.asarray(gene_idx, dtype=np.int64)
    good = idx[ref["valid"][idx]]
    if good.size < 4:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    out_r, out_s = [], []
    for _ in range(n_splits):
        perm = rng.permutation(good.size)
        h1 = good[perm[: good.size // 2]]
        h2 = good[perm[good.size // 2:]]
        s1, _ = M.module_scores(Xg[sel][:, h1], np.arange(h1.size), ref["mean"][h1],
                                ref["std"][h1], np.ones(h1.size, bool))
        s2, _ = M.module_scores(Xg[sel][:, h2], np.arange(h2.size), ref["mean"][h2],
                                ref["std"][h2], np.ones(h2.size, bool))
        r = safe_pcc(s1, s2)
        out_r.append(r)
    return float(np.nanmean(out_r)), float("nan"), len(good)


def main():
    t0 = time.time()
    cfg = load_cfg()
    modules = M.load_module_defs()["modules"]
    hs_cfg, bd_cfg = compute_threshold_maps(cfg, modules)
    import anndata as ad
    obs = ad.read_h5ad(M.OBS, backed="r")
    ref = M.reference_stats(obs)
    # sanity: config reference stats equal the recomputed ones
    for m in modules:
        n_cfg = cfg["modules"]["hotspot_thresholds_reference"][m["name"]]
        idx = np.asarray(m["gene_idx"], dtype=np.int64)
        assert int(ref["valid"][idx].sum()) == n_cfg["n_valid_genes"], m["name"]
    combos = M.load_combos()

    core_rows = []
    pr_rows = []
    diag_rows = []
    reliability_rows = []
    per_slice_pcc = []  # for bootstrap

    def drow(combo, method, scope, metric, value, n=np.nan, ci_lo=np.nan,
             ci_hi=np.nan, note=""):
        diag_rows.append({"combo": combo, "method": method, "scope": scope,
                          "metric": metric, "value": value, "n": n,
                          "ci_lo": ci_lo, "ci_hi": ci_hi, "note": note})

    for combo in combos:
        tg = M.load_target(combo)
        Xg = np.asarray(tg["X"], dtype=np.float64)
        xy_all = np.asarray(tg["coords"], dtype=np.float64)[:, :2]
        slice_all = tg["slice"].astype(str)
        slices = sorted(set(slice_all))
        n_bins = Xg.shape[0]
        gt_mean = float(Xg.mean())
        gt_zero = float((Xg == 0).mean())
        pcc_all_gt = None
        rng = np.random.default_rng(12345)

        for method in M.METHODS:
            t_start = time.time()
            p = M.load_pred(combo, method)
            Xp = np.asarray(p["X"], dtype=np.float64)
            assert Xp.shape == Xg.shape
            pred_mean = float(Xp.mean())
            pred_zero = float((Xp == 0).mean())
            mean_ratio = pred_mean / gt_mean if gt_mean > 0 else float("nan")
            # per-gene PCC (all 1122 genes; zero-variance genes are NaN)
            r_all = M.pcc_cols(Xp, Xg)
            valid_gene_pcc = r_all[np.isfinite(r_all)]
            drow(combo, method, "all_genes", "gene_pcc_p10",
                 float(np.percentile(valid_gene_pcc, 10)), n=int(valid_gene_pcc.size))
            drow(combo, method, "all_genes", "gene_pcc_frac_gt_0.5",
                 float((valid_gene_pcc > 0.5).mean()), n=int(valid_gene_pcc.size))
            drow(combo, method, "all_genes", "gene_pcc_median",
                 float(np.median(valid_gene_pcc)), n=int(valid_gene_pcc.size))
            drow(combo, method, "expression", "mean_ratio", mean_ratio,
                 note="mean(pred)/mean(GT) over all bins and genes")
            drow(combo, method, "expression", "pred_mean", pred_mean)
            drow(combo, method, "expression", "gt_mean", gt_mean)
            drow(combo, method, "expression", "zero_fraction_pred", pred_zero)
            drow(combo, method, "expression", "zero_fraction_gt", gt_zero)
            drow(combo, method, "expression", "zero_fraction_ratio",
                 pred_zero / gt_zero if gt_zero > 0 else float("nan"))
            drow(combo, method, "input", "n_target_bins", n_bins)
            drow(combo, method, "input", "n_target_slices", len(slices))

            # per-slice pooled arrays, used for the 'pooled' scope row
            pooled = {m["name"]: {"gt_A": [], "pred_A": [], "mask": [],
                                  "gt_vals": [], "pred_vals": [],
                                  "hs_tp": 0, "hs_fp": 0, "hs_fn": 0,
                                  "hs_gtpos": 0, "hs_predpos": 0,
                                  "bd0_tp": 0, "bd0_fp": 0, "bd0_fn": 0,
                                  "bd1_matched_gt": 0, "bd1_matched_pred": 0,
                                  "bd1_n_gt": 0, "bd1_n_pred": 0,
                                  "ssim": [], "moran_gt": [], "moran_pred": [],
                                  "geary_gt": [], "geary_pred": []}
                      for m in modules}

            module_expr = {}
            for m in modules:
                name = m["name"]
                idx = np.asarray(m["gene_idx"], dtype=np.int64)
                good = idx[ref["valid"][idx]]
                r = r_all[idx]
                rv = r[np.isfinite(r)]
                expr = {
                    "n_genes": int(len(idx)), "n_valid": int(good.size),
                    "gene_pcc_median": float(np.median(rv)) if rv.size else np.nan,
                    "gene_pcc_p10": float(np.percentile(rv, 10)) if rv.size else np.nan,
                    "gene_pcc_frac_gt_0.5": float((rv > 0.5).mean()) if rv.size else np.nan,
                }
                if good.size:
                    err = Xp[:, good] - Xg[:, good]
                    expr["rmse"] = float(np.sqrt(np.mean(err ** 2)))
                    rmse_g = np.sqrt(np.mean(err ** 2, axis=0))
                    sd_g = Xg[:, good].std(0)
                    srmse = rmse_g / sd_g
                    expr["rmse_std"] = float(np.nanmean(srmse))
                else:
                    expr["rmse"] = np.nan
                    expr["rmse_std"] = np.nan
                module_expr[name] = expr

            # per-slice spatial metrics
            for s in slices:
                sel = slice_all == s
                Xgs = Xg[sel]
                Xps = Xp[sel]
                xs = xy_all[sel]
                r_s = M.pcc_cols(Xps, Xgs)
                for m in modules:
                    name = m["name"]
                    idx = np.asarray(m["gene_idx"], dtype=np.int64)
                    sgt, nvalid = M.module_scores(Xgs, idx, ref["mean"], ref["std"], ref["valid"])
                    spd, _ = M.module_scores(Xps, idx, ref["mean"], ref["std"], ref["valid"])
                    A_gt, mask, rdiag = M.rasterize(xs, sgt)
                    A_pd, _, _ = M.rasterize(xs, spd)
                    gtv = sgt
                    pdv = spd
                    mp = M.pcc_vec(spd, sgt)
                    vr = M.variance_ratio_log2(spd, sgt)
                    mi_gt, gc_gt = M.morans_i(xs, sgt)
                    mi_pd, gc_pd = M.morans_i(xs, spd)
                    L = max(float(np.percentile(A_gt[mask], 99.5)), 1e-6)
                    mss = M.ms_ssim(A_pd, A_gt, mask, L=L)
                    # hotspot
                    thr = hs_cfg[name]["hotspot_threshold_p90"]
                    gt_h = M.hotspot_mask(A_gt, mask, thr)
                    pd_h = M.hotspot_mask(A_pd, mask, thr)
                    tp = int(np.logical_and(gt_h, pd_h).sum())
                    fp = int(np.logical_and(~gt_h, pd_h).sum())
                    fn = int(np.logical_and(gt_h, ~pd_h).sum())
                    n_gtpos = int(gt_h.sum())
                    n_predpos = int(pd_h.sum())
                    rec_h, pre_h, f1_h = M.prf(tp, fp, fn)
                    # boundary
                    gthr = bd_cfg[name]["gradient_threshold_p90"]
                    b_gt, grad_gt, inner = M.boundary_mask(A_gt, mask, gthr)
                    b_pd, grad_pd, _ = M.boundary_mask(A_pd, mask, gthr)
                    tp0 = int(np.logical_and(b_gt, b_pd).sum())
                    fp0 = int(np.logical_and(~b_gt, b_pd).sum())
                    fn0 = int(np.logical_and(b_gt, ~b_pd).sum())
                    rec0, pre0, f10 = M.prf(tp0, fp0, fn0)
                    bm1 = M.boundary_match(b_gt, b_pd, 1)
                    rec1, pre1, f11 = bm1["recall"], bm1["precision"], bm1["f1"]

                    # per-slice core row
                    expr = module_expr[name]
                    idx_good = idx[ref["valid"][idx]]
                    rv_s = r_s[idx]
                    rv_s = rv_s[np.isfinite(rv_s)]
                    if idx_good.size:
                        err_s = Xps[:, idx_good] - Xgs[:, idx_good]
                        rmse_s = float(np.sqrt(np.mean(err_s ** 2)))
                        sd_g = Xgs[:, idx_good].std(0)
                        rmse_std_s = float(np.nanmean(
                            np.sqrt(np.mean(err_s ** 2, axis=0)) / sd_g))
                    else:
                        rmse_s = rmse_std_s = np.nan
                    status, na = [], []
                    if nvalid == 0:
                        status.append("NO_VALID_GENES")
                    if not np.isfinite(mp):
                        na.append("module_pcc")
                    if not np.isfinite(mss):
                        na.append("ms_ssim")
                    if not np.isfinite(vr):
                        na.append("var_ratio")
                    if not np.isfinite(mi_gt) or not np.isfinite(mi_pd):
                        na.append("moran")
                    if n_gtpos == 0:
                        na.append("hotspot(no_GT_positive)")
                    if n_predpos == 0:
                        na.append("hotspot_precision(no_pred_positive)")
                    n_bd_gt = int(b_gt.sum())
                    if n_bd_gt == 0:
                        na.append("boundary_recall(no_GT_boundary)")
                    if int(b_pd.sum()) == 0:
                        na.append("boundary_precision(no_pred_boundary)")
                    core_rows.append({
                        "combo": combo, "method": method, "module": name,
                        "scope": s, "n_bins": int(sel.sum()),
                        "n_genes_module": expr["n_genes"],
                        "n_genes_valid": expr["n_valid"],
                        "gene_pcc_median": float(np.median(rv_s)) if rv_s.size else np.nan,
                        "gene_pcc_p10": float(np.percentile(rv_s, 10)) if rv_s.size else np.nan,
                        "gene_pcc_frac_gt_0.5": float((rv_s > 0.5).mean()) if rv_s.size else np.nan,
                        "rmse": rmse_s,
                        "rmse_std": rmse_std_s,
                        "module_pcc": mp, "ms_ssim": mss,
                        "var_ratio_log2": vr,
                        "abs_delta_moran_i": abs(mi_pd - mi_gt)
                        if np.isfinite(mi_gt) and np.isfinite(mi_pd) else np.nan,
                        "moran_i_gt": mi_gt, "moran_i_pred": mi_pd,
                        "geary_gt": gc_gt, "geary_pred": gc_pd,
                        "hotspot_recall": rec_h,
                        "hotspot_precision": pre_h,
                        "boundary_recall_d0": rec0,
                        "boundary_precision_d0": pre0,
                        "boundary_recall_d1": rec1,
                        "boundary_precision_d1": pre1,
                        "status": ";".join(status) if status else "OK",
                        "na_reason": ";".join(na),
                    })

                    # hotspot AP / PR-AUC / threshold sensitivity
                    score_vals = A_pd[mask]
                    lab_vals = (A_gt[mask] > thr)
                    if lab_vals.sum() == 0:
                        ap = auc = np.nan
                        na_ap = "no_GT_positive"
                    elif np.nanstd(score_vals) == 0:
                        ap = auc = np.nan
                        na_ap = "constant_prediction"
                    else:
                        ap, auc, _ = M.ap_pr_auc(score_vals, lab_vals)
                        na_ap = ""
                    sens = {}
                    for q in (80, 90, 95):
                        tq = hs_cfg[name][f"hotspot_threshold_p{q}"]
                        gq = M.hotspot_mask(A_gt, mask, tq)
                        pq = M.hotspot_mask(A_pd, mask, tq)
                        tpq = int(np.logical_and(gq, pq).sum())
                        fpq = int(np.logical_and(~gq, pq).sum())
                        fnq = int(np.logical_and(gq, ~pq).sum())
                        sens[f"p{q}_recall"], sens[f"p{q}_precision"], _ = M.prf(
                            tpq, fpq, fnq)
                    pr_rows.append({
                        "combo": combo, "method": method, "module": name,
                        "scope": s, "n_bins": int(sel.sum()),
                        "hotspot_threshold": thr,
                        "n_gt_positive": n_gtpos, "n_pred_positive": n_predpos,
                        "hotspot_tp": tp, "hotspot_fp": fp, "hotspot_fn": fn,
                        "hotspot_recall": rec_h,
                        "hotspot_precision": pre_h,
                        "hotspot_f1": f1_h,
                        "hotspot_AP": ap, "hotspot_PR_AUC": auc,
                        "hotspot_AP_na": na_ap,
                        "sens_p80_recall": sens["p80_recall"],
                        "sens_p80_precision": sens["p80_precision"],
                        "sens_p90_recall": sens["p90_recall"],
                        "sens_p90_precision": sens["p90_precision"],
                        "sens_p95_recall": sens["p95_recall"],
                        "sens_p95_precision": sens["p95_precision"],
                        "boundary_gradient_threshold": gthr,
                        "n_gt_boundary": int(b_gt.sum()),
                        "n_pred_boundary": int(b_pd.sum()),
                        "boundary_tp_d0": tp0, "boundary_fp_d0": fp0,
                        "boundary_fn_d0": fn0, "boundary_recall_d0": rec0,
                        "boundary_precision_d0": pre0, "boundary_f1_d0": f10,
                        "boundary_matched_gt_d1": bm1["matched_gt"],
                        "boundary_matched_pred_d1": bm1["matched_pred"],
                        "boundary_recall_d1": rec1,
                        "boundary_precision_d1": pre1, "boundary_f1_d1": f11,
                        "n_gt_boundary_components": int(M.component_centroids(b_gt).shape[0]),
                        "n_pred_boundary_components": int(M.component_centroids(b_pd).shape[0]),
                        "boundary_centroid_dist_bins": M.centroid_distances(b_gt, b_pd),
                    })

                    # accumulate pooled
                    acc = pooled[name]
                    acc["gt_A"].append(A_gt)
                    acc["pred_A"].append(A_pd)
                    acc["mask"].append(mask)
                    acc["gt_vals"].append(gtv)
                    acc["pred_vals"].append(pdv)
                    acc["hs_tp"] += tp
                    acc["hs_fp"] += fp
                    acc["hs_fn"] += fn
                    acc["hs_gtpos"] += n_gtpos
                    acc["hs_predpos"] += n_predpos
                    acc["bd0_tp"] += tp0
                    acc["bd0_fp"] += fp0
                    acc["bd0_fn"] += fn0
                    acc["bd1_matched_gt"] += bm1["matched_gt"]
                    acc["bd1_matched_pred"] += bm1["matched_pred"]
                    acc["bd1_n_gt"] += bm1["n_gt"]
                    acc["bd1_n_pred"] += bm1["n_pred"]
                    acc["ssim"].append(mss)
                    acc["moran_gt"].append(mi_gt)
                    acc["moran_pred"].append(mi_pd)
                    acc["geary_gt"].append(gc_gt)
                    acc["geary_pred"].append(gc_pd)

                    per_slice_pcc.append({
                        "combo": combo, "method": method, "module": name,
                        "slice": s, "module_pcc": mp})

            # pooled rows
            for m in modules:
                name = m["name"]
                acc = pooled[name]
                expr = module_expr[name]
                gtv = np.concatenate(acc["gt_vals"])
                pdv = np.concatenate(acc["pred_vals"])
                mp = M.pcc_vec(pdv, gtv)
                vr = M.variance_ratio_log2(pdv, gtv)
                mis = [abs(a - b) for a, b in zip(acc["moran_pred"], acc["moran_gt"])
                       if np.isfinite(a) and np.isfinite(b)]
                rec_h, pre_h, f1_h = M.prf(acc["hs_tp"], acc["hs_fp"], acc["hs_fn"])
                rec0, pre0, _ = M.prf(acc["bd0_tp"], acc["bd0_fp"], acc["bd0_fn"])
                rec1 = acc["bd1_matched_gt"] / acc["bd1_n_gt"] if acc["bd1_n_gt"] else float("nan")
                pre1 = acc["bd1_matched_pred"] / acc["bd1_n_pred"] if acc["bd1_n_pred"] else float("nan")
                status, na = [], []
                if expr["n_valid"] == 0:
                    status.append("NO_VALID_GENES")
                if not np.isfinite(mp):
                    na.append("module_pcc")
                if not np.isfinite(vr):
                    na.append("var_ratio")
                if not mis:
                    na.append("moran")
                if acc["hs_gtpos"] == 0:
                    na.append("hotspot(no_GT_positive)")
                if acc["hs_predpos"] == 0:
                    na.append("hotspot_precision(no_pred_positive)")
                if acc["bd0_tp"] + acc["bd0_fn"] == 0:
                    na.append("boundary_recall(no_GT_boundary)")
                if acc["bd0_tp"] + acc["bd0_fp"] == 0:
                    na.append("boundary_precision(no_pred_boundary)")
                core_rows.append({
                    "combo": combo, "method": method, "module": name,
                    "scope": "pooled", "n_bins": n_bins,
                    "n_genes_module": expr["n_genes"],
                    "n_genes_valid": expr["n_valid"],
                    "gene_pcc_median": expr["gene_pcc_median"],
                    "gene_pcc_p10": expr["gene_pcc_p10"],
                    "gene_pcc_frac_gt_0.5": expr["gene_pcc_frac_gt_0.5"],
                    "rmse": expr["rmse"], "rmse_std": expr["rmse_std"],
                    "module_pcc": mp,
                    "ms_ssim": float(np.nanmean(acc["ssim"])) if acc["ssim"] else np.nan,
                    "var_ratio_log2": vr,
                    "abs_delta_moran_i": float(np.nanmean(mis)) if mis else np.nan,
                    "moran_i_gt": float(np.nanmean(acc["moran_gt"])),
                    "moran_i_pred": float(np.nanmean(acc["moran_pred"])),
                    "geary_gt": float(np.nanmean(acc["geary_gt"])),
                    "geary_pred": float(np.nanmean(acc["geary_pred"])),
                    "hotspot_recall": rec_h, "hotspot_precision": pre_h,
                    "boundary_recall_d0": rec0, "boundary_precision_d0": pre0,
                    "boundary_recall_d1": rec1, "boundary_precision_d1": pre1,
                    "status": ";".join(status) if status else "OK",
                    "na_reason": ";".join(na),
                })
                # diagnostics: AP and boundary F1 aggregated over slices (micro)
                aps = []
                for r in pr_rows:
                    if (r["combo"] == combo and r["method"] == method
                            and r["module"] == name):
                        if np.isfinite(r["hotspot_AP"]):
                            aps.append(r["hotspot_AP"])
                drow(combo, method, name, "hotspot_AP_mean",
                     float(np.mean(aps)) if aps else np.nan, n=len(aps))
                drow(combo, method, name, "boundary_F1_d0",
                     2 * rec0 * pre0 / (rec0 + pre0)
                     if np.isfinite(rec0) and np.isfinite(pre0) and (rec0 + pre0) > 0
                     else np.nan)
                drow(combo, method, name, "boundary_F1_d1",
                     2 * rec1 * pre1 / (rec1 + pre1)
                     if np.isfinite(rec1) and np.isfinite(pre1) and (rec1 + pre1) > 0
                     else np.nan)

            # combo-level coverage / no-positive diagnostics
            core_m = [r for r in core_rows if r["combo"] == combo
                      and r["method"] == method and r["scope"] == "pooled"]
            n_ok = sum(1 for r in core_m if all(
                np.isfinite(r[c]) for c in CORE_COLS))
            drow(combo, method, "modules", "coverage_all_10_core",
                 float(n_ok / len(core_m)), n=len(core_m))
            drow(combo, method, "modules", "n_modules_no_hotspot_positive",
                 int(sum(1 for r in core_m if "no_GT_positive" in r["na_reason"])))
            drow(combo, method, "modules", "n_modules_no_boundary",
                 int(sum(1 for r in core_m if "no_GT_boundary" in r["na_reason"])))
            drow(combo, method, "runtime", "metrics_sec",
                 float(time.time() - t_start))

        # ---- GT split-half reliability (method independent), per module/slice
        for m in modules:
            name = m["name"]
            idx = np.asarray(m["gene_idx"], dtype=np.int64)
            vals = []
            for s in slices:
                sel = slice_all == s
                r, _, nvalid = split_half_reliability(Xg, sel, idx, ref)
                if np.isfinite(r):
                    vals.append(r)
                    reliability_rows.append({
                        "combo": combo, "method": "GT", "module": name,
                        "slice": s, "split_half_pcc": r, "n_valid_genes": nvalid})
            if vals:
                drow(combo, "GT", name, "split_half_pcc_mean", float(np.mean(vals)),
                     n=len(vals), note="gene split-half of GT module map, 5 random splits")
        allrel = [r["split_half_pcc"] for r in reliability_rows if r["combo"] == combo]
        if allrel:
            drow(combo, "GT", "modules", "split_half_pcc_mean_all_modules",
                 float(np.mean(allrel)), n=len(allrel))

    # ---------------- bootstrap by slice (5 slices, paired) ----------------
    ps = pd.DataFrame(per_slice_pcc)
    rng = np.random.default_rng(20261022)
    slice_ids = np.array(sorted(ps["slice"].unique()))
    n_boot = 2000
    boot_idx = rng.integers(0, len(slice_ids), size=(n_boot, len(slice_ids)))

    def boot_mean(vals_by_slice, resample):
        out = []
        for b in range(resample.shape[0]):
            sel = np.concatenate([np.where(vals_by_slice["slice"].values ==
                                          slice_ids[j])[0]
                                  for j in resample[b]])
            if sel.size:
                out.append(float(np.nanmean(vals_by_slice["module_pcc"].values[sel])))
        return np.array(out)

    method_boot = {}
    for method in M.METHODS:
        sub = ps[ps["method"] == method]
        point = float(np.nanmean(sub["module_pcc"]))
        b = boot_mean(sub, boot_idx)
        method_boot[method] = b
        drow("ALL", method, "modules", "module_pcc_mean_bootstrap_by_slice",
             point, n=int(len(sub)), ci_lo=float(np.nanpercentile(b, 2.5)),
             ci_hi=float(np.nanpercentile(b, 97.5)),
             note=f"{n_boot} resamples over 5 slice ids; rows = combo x slice x module")
    if "Spateo" in method_boot and "NTF" in method_boot:
        diff = method_boot["Spateo"] - method_boot["NTF"]
        pt = float(np.nanmean(ps[ps["method"] == "Spateo"]["module_pcc"]) -
                   np.nanmean(ps[ps["method"] == "NTF"]["module_pcc"]))
        drow("ALL", "Spateo-NTF", "modules", "paired_module_pcc_diff_bootstrap",
             pt, n=int(len(ps)), ci_lo=float(np.nanpercentile(diff, 2.5)),
             ci_hi=float(np.nanpercentile(diff, 97.5)),
             note="paired by slice; 2000 resamples; 5 slices only, CIs are wide")

    # ---------------- write outputs ----------------
    core = pd.DataFrame(core_rows)
    core.to_csv(os.path.join(HERE, "metrics_core.csv"), index=False)
    pr = pd.DataFrame(pr_rows)
    pr.to_csv(os.path.join(HERE, "metrics_spatial_pr.csv"), index=False)
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(os.path.join(HERE, "metrics_diagnostics.csv"), index=False)
    rel = pd.DataFrame(reliability_rows)
    rel.to_csv(os.path.join(HERE, "gt_split_half_reliability.csv"), index=False)

    # module summary: mean/median across the 7 combos for pooled rows
    pool = core[core["scope"] == "pooled"].copy()
    agg_rows = []
    for (method, name), g in pool.groupby(["method", "module"], sort=False):
        row = {"method": method, "module": name,
               "n_combos": int(len(g)),
               "n_combos_all_10_core": int(sum(
                   all(np.isfinite(r[c]) for c in CORE_COLS) for _, r in g.iterrows()))}
        for c in CORE_COLS:
            v = g[c].astype(float)
            row[c + "_mean"] = float(v.mean())
            row[c + "_median"] = float(v.median())
            row[c + "_min"] = float(v.min())
            row[c + "_max"] = float(v.max())
            row[c + "_n"] = int(v.notna().sum())
        agg_rows.append(row)
    pd.DataFrame(agg_rows).to_csv(os.path.join(HERE, "metrics_module.csv"), index=False)

    # summary for status.json
    summ = {
        "n_core_rows": int(len(core)),
        "n_pr_rows": int(len(pr)),
        "n_diag_rows": int(len(diag)),
        "n_reliability_rows": int(len(rel)),
        "coverage_ok_mean": float(diag[diag["metric"] == "coverage_all_10_core"]["value"].mean()),
        "bootstrap": {r["method"]: {"point": r["value"], "ci_lo": r["ci_lo"],
                                    "ci_hi": r["ci_hi"]}
                      for _, r in diag[diag["metric"] ==
                                       "module_pcc_mean_bootstrap_by_slice"].iterrows()},
        "bootstrap_diff": {
            "point": float(diag[diag["metric"] ==
                                "paired_module_pcc_diff_bootstrap"]["value"].iloc[0]),
            "ci_lo": float(diag[diag["metric"] ==
                                "paired_module_pcc_diff_bootstrap"]["ci_lo"].iloc[0]),
            "ci_hi": float(diag[diag["metric"] ==
                                "paired_module_pcc_diff_bootstrap"]["ci_hi"].iloc[0])},
        "core_highlights": {},
        "seconds": float(time.time() - t0),
        "gpu": {},
    }
    import torch
    if torch.cuda.is_available():
        summ["gpu"] = {"device": torch.cuda.get_device_name(0),
                       "max_memory_allocated_mb": float(
                           torch.cuda.max_memory_allocated() / 2 ** 20),
                       "max_memory_reserved_mb": float(
                           torch.cuda.max_memory_reserved() / 2 ** 20)}
    with open(os.path.join(HERE, "core_summary.json"), "w") as f:
        json.dump(summ, f, indent=2, default=M.jsonable)
    print(json.dumps(summ, indent=2, default=M.jsonable))
    print("elapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()

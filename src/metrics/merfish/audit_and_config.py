#!/usr/bin/env python
"""Phase 1-2: input audit + frozen metric configuration (metric protocol v1).

Writes:
  metrics_v1/input_audit.csv
  metrics_v1/metric_config.yaml
  metrics_v1/audit_summary.json   (machine-readable summary for status.json)
"""
import csv
import json
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlib as M

OUT = HERE
AUDIT_ROWS = []


def add(section, check, combo="", method="", scope="", value=None,
        expected=None, ok=None, note=""):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{v:.8g}"
        return "" if v is None else str(v)
    AUDIT_ROWS.append({
        "section": section, "check": check, "combo": combo, "method": method,
        "scope": scope, "value": fmt(value), "expected": fmt(expected),
        "pass": "" if ok is None else bool(ok), "note": note,
    })


def main():
    t0 = time.time()
    import anndata as ad

    combos = M.load_combos()
    mdef = M.load_module_defs()
    modules = mdef["modules"]
    genes = M.load_genes()

    obs = ad.read_h5ad(M.OBS, backed="r")
    var_names = obs.var_names.astype(str)
    n_genes = len(var_names)
    add("global", "n_genes", value=n_genes, expected=1122, ok=n_genes == 1122)
    add("global", "gene_order_md5", value=M.md5_array(var_names.astype("U")),
        note="md5 of var_names; X columns follow this order")
    add("global", "n_combos", value=len(combos), expected=7, ok=len(combos) == 7)
    add("global", "combos", value=",".join(combos))
    add("global", "methods", value=",".join(M.METHODS))

    # --- gene order / module / reference-stat integrity -------------------
    ref = M.reference_stats(obs)
    sd_zero = ~ref["valid"]
    add("genes", "gene_order_equals_reference_stats",
        value=np.array_equal(ref["genes"], genes), expected=True,
        ok=np.array_equal(ref["genes"], genes))
    add("genes", "reference_stats_genes_equals_var_names",
        value=np.array_equal(ref["genes"], var_names), expected=True,
        ok=np.array_equal(ref["genes"], var_names))
    stored = np.load(os.path.join(M.ROOT, "modules", "reference_stats.npz"),
                     allow_pickle=True)
    d_mean = float(np.max(np.abs(ref["mean"] - stored["mean"].astype(np.float64))))
    st_std = stored["std"].astype(np.float64)
    nz = ref["valid"]
    d_std = float(np.max(np.abs(ref["std"][nz] - st_std[nz]))) if nz.any() else 0.0
    floor_ok = bool(np.all(st_std[~nz] == 1.0))
    add("leakage", "reference_stats_recomputed_from_S1S5_mean_maxdiff",
        value=d_mean, expected=0.0, ok=d_mean < 1e-5,
        note="recomputed from observed_50um rows of S1/S5")
    add("leakage", "reference_stats_recomputed_from_S1S5_std_maxdiff",
        value=d_std, expected=0.0, ok=d_std < 1e-3,
        note="non-zero-std genes; tolerance 1e-3 for float32 storage; zero-std genes stored as 1.0")
    add("leakage", "reference_stats_zero_variance_floor", value=floor_ok,
        expected=True, ok=floor_ok, note="zero-std genes stored as 1.0 but excluded from metrics")
    add("genes", "zero_reference_variance_genes", value=int(sd_zero.sum()),
        expected=6, ok=int(sd_zero.sum()) == 6,
        note=",".join(genes[sd_zero].tolist()))
    add("leakage", "module_reference_slices", value=",".join(mdef["reference_slices"]),
        expected=",".join(M.REF_SLICES),
        ok=list(mdef["reference_slices"]) == list(M.REF_SLICES))
    add("modules", "n_modules", value=len(modules), expected=18, ok=len(modules) == 18)

    mod_names = [m["name"] for m in modules]
    add("modules", "module_names_unique", value=len(set(mod_names)),
        expected=len(mod_names), ok=len(set(mod_names)) == len(mod_names),
        note=";".join(mod_names))

    gene_to_idx = {}
    for i, g in enumerate(genes):
        gene_to_idx.setdefault(g, []).append(i)
    for m in modules:
        idx = np.asarray(m["gene_idx"], dtype=np.int64)
        ok = True
        note = []
        if np.any(idx < 0) or np.any(idx >= n_genes):
            ok = False
            note.append("index out of range")
        else:
            if [genes[i] for i in idx] != list(m["genes"]):
                ok = False
                note.append("gene_idx/name mismatch")
            if gene_to_idx.get(m["genes"][0], []) and \
                    gene_to_idx[m["genes"][0]][0] != int(idx[0]):
                ok = False
                note.append("name not at gene_idx")
        valid_genes = int(ref["valid"][idx].sum())
        add("modules", "module_gene_coverage", combo="", method="", scope=m["name"],
            value=f"{len(idx)} genes / {valid_genes} valid",
            expected="20 / 20 unless zero-variance gene present", ok=ok,
            note=";".join(note) + ("" if valid_genes == len(idx) else
                                   " excluded=" + ",".join(
                                       [genes[i] for i in idx if not ref["valid"][i]])))

    # --- reference module thresholds (S1/S5 only) ------------------------
    X_ref, meta_ref = M.load_reference_expression(obs)
    ref_slices = meta_ref["slice"].astype(str).values
    ref_thresholds = {}
    ref_boundary = {}
    for m in modules:
        sc, n_valid = M.module_scores(X_ref, m["gene_idx"], ref["mean"],
                                      ref["std"], ref["valid"])
        vals = []
        for s in M.REF_SLICES:
            sel = ref_slices == s
            A, mask, _ = M.rasterize(meta_ref.loc[sel, ["x_mm", "y_mm"]].values,
                                     sc[sel])
            vals.append(A[mask])
        vals = np.concatenate(vals)
        qs = {q: float(np.percentile(vals, q)) for q in (80, 90, 95)}
        ref_thresholds[m["name"]] = {
            "n_valid_genes": int(n_valid), "n_reference_bins": int(vals.size),
            "reference_mean": float(vals.mean()), "reference_std": float(vals.std()),
            "hotspot_threshold_p80": qs[80], "hotspot_threshold_p90": qs[90],
            "hotspot_threshold_p95": qs[95],
        }
        grads = []
        for s in M.REF_SLICES:
            sel = ref_slices == s
            A, mask, _ = M.rasterize(meta_ref.loc[sel, ["x_mm", "y_mm"]].values,
                                     sc[sel])
            g = M.gradient_map(A, mask, sigma=1.0)
            inner = M.erode_mask(mask, M.CONTOUR_BINS)
            grads.append(g[inner])
        grads = np.concatenate(grads)
        ref_boundary[m["name"]] = {
            "n_reference_bins": int(grads.size),
            "gradient_threshold_p80": float(np.percentile(grads, 80)),
            "gradient_threshold_p90": float(np.percentile(grads, 90)),
            "gradient_threshold_p95": float(np.percentile(grads, 95)),
        }

    # --- load target GT once (S2/S3/S4 only) -----------------------------
    all_names = []
    for c in combos:
        t = M.load_target(c)
        all_names.append(t["obs_names"].astype(str))
    all_names = np.concatenate(all_names)
    obs_names = obs.obs_names.astype(str)
    name_to_obs = {n: i for i, n in enumerate(obs_names)}
    missing = [n for n in all_names if n not in name_to_obs]
    add("masks", "target_obs_names_found_in_observed_h5ad",
        value=len(all_names) - len(missing), expected=len(all_names),
        ok=len(missing) == 0, note=f"missing={len(missing)}")
    obs_idx = np.array([name_to_obs[n] for n in all_names], dtype=np.int64)
    uniq_idx, inv = np.unique(obs_idx, return_inverse=True)
    X_unique = np.asarray(obs.X[uniq_idx], dtype=np.float32)
    X_obs_rows = X_unique[inv]
    obs_meta = obs.obs.iloc[uniq_idx].iloc[inv]
    obs_names_rows = obs_names[uniq_idx][inv]
    add("masks", "target_slices_disjoint_from_reference",
        value=",".join(sorted(set(obs_meta["slice"].astype(str)) - set(M.REF_SLICES))),
        expected="S2,S3,S4", ok=set(obs_meta["slice"].astype(str)).isdisjoint(M.REF_SLICES),
        note="target slices never used for reference stats/modules/thresholds")
    full_slice_counts = obs.obs["slice"].astype(str).value_counts()
    for s in M.REF_SLICES:
        add("masks", "reference_slice_present", scope=s,
            value=int(full_slice_counts.get(s, 0)), expected=">0",
            ok=int(full_slice_counts.get(s, 0)) > 0)

    # --- per-combo / per-method audit ------------------------------------
    import pandas as pd
    prior = {}
    if os.path.exists(os.path.join(M.ROOT, "metrics.csv")):
        prior = pd.read_csv(os.path.join(M.ROOT, "metrics.csv")).set_index(
            ["combo", "method"]).to_dict("index")

    ptr = 0
    audit_summary = {"combos": {}, "thresholds": ref_thresholds,
                     "boundary_thresholds": ref_boundary,
                     "n_genes": n_genes,
                     "zero_variance_genes": genes[sd_zero].tolist()}
    for combo in combos:
        t = M.load_target(combo)
        tn = t["obs_names"].astype(str)
        Xg = np.asarray(t["X"], dtype=np.float64)
        # row selection in the concatenated observed block
        sl = slice(ptr, ptr + len(tn))
        ptr += len(tn)
        obs_block = X_obs_rows[sl]
        om = obs_meta.iloc[sl]
        d_gt = float(np.max(np.abs(Xg - obs_block.astype(np.float64))))
        coord_obs = om[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=np.float64)
        d_coord = float(np.max(np.abs(np.asarray(t["coords"], dtype=np.float64) - coord_obs)))
        sl_names = t["slice"].astype(str)
        add("target", "target_rows", combo=combo, value=int(Xg.shape[0]),
            expected=int(Xg.shape[0]), ok=True)
        add("target", "target_gene_columns", combo=combo, value=int(Xg.shape[1]),
            expected=n_genes, ok=Xg.shape[1] == n_genes)
        tgt_slices = sorted(set(sl_names))
        add("target", "target_slices", combo=combo, value=",".join(tgt_slices),
            expected=",".join(combos[combo]["missing_slice_ids"]),
            ok=tgt_slices == sorted(combos[combo]["missing_slice_ids"]))
        add("target", "target_X_matches_observed_h5ad", combo=combo, value=d_gt,
            expected=0.0, ok=d_gt < 1e-6, note="max|target.X - observed.X| on same obs_names")
        add("target", "target_coords_match_observed", combo=combo, value=d_coord,
            expected=0.0, ok=d_coord < 1e-5, note="max|coords - obs x/y/z_mm|")
        add("target", "target_finite_fraction", combo=combo,
            value=float(np.isfinite(Xg).mean()), expected=1.0,
            ok=bool(np.isfinite(Xg).all()))
        add("target", "target_zero_fraction", combo=combo,
            value=float((Xg == 0).mean()), expected=">0")
        add("target", "target_constant_rows", combo=combo,
            value=int((Xg.std(1) == 0).sum()), expected=0,
            ok=int((Xg.std(1) == 0).sum()) == 0)
        add("target", "target_mean", combo=combo, value=float(Xg.mean()))
        dup_max = 0
        for s in tgt_slices:
            xy_s = np.asarray(t["coords"], dtype=np.float64)[sl_names == s, :2]
            dup_max = max(dup_max, int(len(xy_s) - len(np.unique(xy_s, axis=0))))
        add("target", "target_duplicate_2d_bins_within_slice", combo=combo,
            value=dup_max, expected=0, ok=dup_max == 0,
            note="cross-slice (x,y) repeats are expected and audited separately")

        n_valid_modules = 0
        valid_gene_counts = []
        for m in modules:
            ok_n = int(ref["valid"][np.asarray(m["gene_idx"], dtype=np.int64)].sum())
            valid_gene_counts.append(ok_n)
            n_valid_modules += int(ok_n > 0)
        add("modules", "modules_with_valid_genes", combo=combo,
            value=n_valid_modules, expected=18, ok=n_valid_modules == 18,
            note=f"valid genes per module min={min(valid_gene_counts)}")

        audit_summary["combos"][combo] = {
            "n_target": int(Xg.shape[0]), "target_slices": tgt_slices,
            "target_mean": float(Xg.mean()),
            "target_zero_frac": float((Xg == 0).mean()), "methods": {}}

        for method in M.METHODS:
            p = M.load_pred(combo, method)
            Xp = np.asarray(p["X"], dtype=np.float64)
            pn = p["obs_names"].astype(str)
            row = {
                "n_rows": int(Xp.shape[0]),
                "shape_ok": Xp.shape == Xg.shape,
                "names_equal": bool(np.array_equal(pn, tn)),
                "coords_maxdiff": float(np.max(np.abs(
                    np.asarray(p["coords"], dtype=np.float64) -
                    np.asarray(t["coords"], dtype=np.float64)))),
                "slice_equal": bool(np.array_equal(p["slice"].astype(str), sl_names)),
                "finite": bool(np.isfinite(Xp).all()),
                "const_rows": int((Xp.std(1) == 0).sum()),
                "pred_mean": float(Xp.mean()),
                "zero_frac": float((Xp == 0).mean()),
                "eq_frac_all": float((Xp == Xg).mean()),
                "eq_frac_gt_positive": float((Xp[Xg > 0] == Xg[Xg > 0]).mean())
                if int((Xg > 0).sum()) else float("nan"),
                "method_meta": str(p.get("method", "")),
                "combo_meta": str(p.get("combo", "")),
            }
            # pooled PCC sanity vs the earlier metrics.csv (cross-check)
            a = Xp.ravel() - Xp.mean()
            b = Xg.ravel() - Xg.mean()
            row["pcc_pooled"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            if prior.get((combo, method)):
                row["pcc_prior"] = float(prior[(combo, method)]["pcc_pooled"])
            row["pcc_matches_prior"] = bool(
                "pcc_prior" not in row or
                abs(row["pcc_pooled"] - row["pcc_prior"]) < 1e-5)

            add("alignment", "pred_rows", combo=combo, method=method,
                value=row["n_rows"], expected=int(Xg.shape[0]),
                ok=row["n_rows"] == Xg.shape[0])
            add("alignment", "pred_shape", combo=combo, method=method,
                value=str(Xp.shape), expected=str(Xg.shape), ok=row["shape_ok"])
            add("alignment", "obs_names_exact", combo=combo, method=method,
                value=row["names_equal"], expected=True, ok=row["names_equal"])
            add("alignment", "coords_exact", combo=combo, method=method,
                value=row["coords_maxdiff"], expected=0.0,
                ok=row["coords_maxdiff"] < 1e-6)
            add("alignment", "slice_labels_exact", combo=combo, method=method,
                value=row["slice_equal"], expected=True, ok=row["slice_equal"])
            add("alignment", "finite_fraction", combo=combo, method=method,
                value=float(np.isfinite(Xp).mean()), expected=1.0, ok=row["finite"])
            add("alignment", "constant_rows", combo=combo, method=method,
                value=row["const_rows"], expected=0, ok=row["const_rows"] == 0)
            add("alignment", "pred_metadata", combo=combo, method=method,
                value=f"{row['method_meta']}/{row['combo_meta']}",
                expected=f"{method}/{combo}",
                ok=(row["method_meta"] == method and row["combo_meta"] == combo))
            add("leakage", "pred_equals_target_elementwise_all", combo=combo,
                method=method, value=row["eq_frac_all"], expected="n/a (zeros coincide)",
                ok=None,
                note="informational: zero-inflated matrices coincide at zeros; the GT-positive equality check is the leakage test")
            add("leakage", "pred_equals_target_elementwise_gt_positive",
                combo=combo, method=method, value=row["eq_frac_gt_positive"],
                expected="~0 (<1e-3)", ok=row["eq_frac_gt_positive"] < 1e-3)
            add("leakage", "pcc_matches_prior_compute_metrics", combo=combo,
                method=method, value=row.get("pcc_prior", ""),
                expected=row["pcc_pooled"], ok=row["pcc_matches_prior"])
            add("expression", "pred_mean", combo=combo, method=method,
                value=row["pred_mean"], expected=f"GT {Xg.mean():.6g}")
            add("expression", "pred_zero_fraction", combo=combo, method=method,
                value=row["zero_frac"], expected=f"GT {(Xg == 0).mean():.6g}")
            add("expression", "pred_pooled_pcc", combo=combo, method=method,
                value=row["pcc_pooled"])
            audit_summary["combos"][combo]["methods"][method] = row

    # --- write audit CSV -------------------------------------------------
    with open(os.path.join(OUT, "input_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "check", "combo", "method",
                                          "scope", "value", "expected", "pass", "note"])
        w.writeheader()
        for r in AUDIT_ROWS:
            w.writerow(r)

    n_pass = sum(1 for r in AUDIT_ROWS if r["pass"] is True)
    n_fail = sum(1 for r in AUDIT_ROWS if r["pass"] is False)
    n_na = sum(1 for r in AUDIT_ROWS if r["pass"] == "")
    audit_summary.update({"n_checks": len(AUDIT_ROWS), "n_pass": n_pass,
                          "n_fail": n_fail, "n_info": n_na,
                          "failures": [r for r in AUDIT_ROWS if r["pass"] is False]})

    # --- frozen config ---------------------------------------------------
    cfg = {
        "protocol": "MERFISH 5-slice metric protocol v1",
        "status": "FROZEN before any method metric was computed",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": M.ROOT,
        "inputs": {
            "observed_50um_h5ad": M.OBS,
            "targets": "targets/<combo>/target.npz (keys X, coords, obs_names, slice)",
            "predictions": "predictions/<combo>/<method>.npz (same keys)",
            "modules": "modules/modules_celltype.json",
            "reference_stats": "modules/reference_stats.npz (recomputed; audited)",
        },
        "data": {
            "bin_um": 50, "n_genes": int(n_genes),
            "gene_order_md5": M.md5_array(var_names.astype("U")),
            "expression_scale": "npz X used as-is; no renormalisation",
            "alignment": "exact obs_names + coords required (audited)",
            "target_slices": ["C57BL6J-3.007", "C57BL6J-3.008", "C57BL6J-3.009"],
            "reference_slices": M.REF_SLICES,
            "n_slices_total": 5,
        },
        "modules": {
            "source": "modules/modules_celltype.json",
            "n_modules": len(modules),
            "reference_slices_for_modules": list(mdef["reference_slices"]),
            "module_map": "M_k(bin) = mean_g z_g(bin) over valid genes",
            "zscore": "per gene: (X - mean_g) / std_g, mean/std from observed S1/S5 bins only",
            "zero_variance_rule": "genes with S1/S5 std == 0 excluded from module means and from gene PCC; listed in audit",
            "zero_variance_genes": genes[sd_zero].tolist(),
            "names": mod_names,
            "module_genes": {m["name"]: m["genes"] for m in modules},
            "module_gene_idx": {m["name"]: [int(i) for i in m["gene_idx"]] for m in modules},
            "n_valid_genes": {m["name"]: ref_thresholds[m["name"]]["n_valid_genes"]
                              for m in modules},
            "hotspot_thresholds_reference": ref_thresholds,
            "boundary_thresholds_reference": ref_boundary,
        },
        "metrics": {
            "expression": {
                "gene_pcc": "Pearson per gene across target bins; median over module genes (core); p10 and frac>0.5 (diagnostics)",
                "rmse": "over module genes and target bins on npz X scale",
                "standardised_rmse": "per (gene,bin) error / target std_g, then RMSE (diagnostics)",
                "zero_variance": "N/A",
            },
            "module": {
                "module_pcc": "Pearson between predicted and GT module maps across target bins",
                "ms_ssim": {"dim": "2D per target section", "window": M.SSIM_WIN,
                            "scales": 3, "weights_raw": [0.0448, 0.2856, 0.3001],
                            "weights_normalised": [float(x) for x in M.SSIM_WEIGHTS],
                            "range": "masked tissue, invalid border windows dropped; geometric mean"},
            },
            "spatial": {
                "variance_ratio": "log2(var(pred module map) / var(GT module map))",
                "morans_i": {"knn_k": M.KNN_K, "weights": "symmetrised row-normalised kNN",
                             "primary": "|I_pred - I_GT|", "secondary": "Geary's C"},
            },
            "hotspot": {
                "threshold": "per-module p90 of module score on visible reference slices S1/S5",
                "same_numeric_threshold_for_gt_and_pred": True,
                "sensitivity_percentiles": [80, 90, 95],
                "threshold_free": "average precision (AP) and trapezoidal PR-AUC",
                "no_positive_rule": "recall N/A if GT has no positives; precision N/A if pred has no positives; AP/PR-AUC N/A if GT has no positives",
            },
            "boundary": {
                "smoothing_sigma_bins": 1.0,
                "gradient": "np.gradient magnitude of Gaussian-smoothed module map",
                "threshold": "per-module p90 of reference S1/S5 gradient (same numeric threshold for GT and pred)",
                "outer_contour": f"erode tissue mask by {M.CONTOUR_BINS} bins, then drop components < {M.MIN_COMPONENT_BINS} bins",
                "tolerance_bins": [0, 1],
                "primary_core": "delta=0; delta=1 in metrics_spatial_pr.csv",
                "f1": "2PR/(P+R) (diagnostics)",
                "object_level": "component counts and GT->pred centroid distance",
            },
        },
        "core_metrics": [
            "gene_pcc_median", "rmse", "module_pcc", "ms_ssim",
            "var_ratio_log2", "abs_delta_moran_i",
            "hotspot_recall", "hotspot_precision",
            "boundary_recall_d0", "boundary_precision_d0"],
        "diagnostics": [
            "mean_ratio", "zero_fraction", "gene_pcc_p10", "gene_pcc_frac_gt_0.5",
            "hotspot_AP", "hotspot_PR_AUC", "boundary_F1", "coverage",
            "no_positive", "split_half_GT_reliability", "paired_bootstrap_by_slice",
            "runtime", "GPU/RAM"],
        "aggregation": {
            "per_slice": "all spatial metrics computed per target section",
            "pooled": "expression metrics over all target bins; MS-SSIM/Moran mean over sections; hotspot/boundary micro-summed over sections",
            "module_summary": "mean/median across the 7 combos in metrics_module.csv",
            "no_composite_score": True,
        },
        "perturbation": {
            "input": "held-out GT module maps only (no predictions, no retraining)",
            "conditions": {
                "identity": [], "blur_sigma_bins": [1, 2, 3],
                "add_hotspot_radius_bins": [2, 3],
                "add_boundary_band": [1],
                "shift_bins": [1, 2],
                "correlated_noise_std_factor": [0.5, 1.0, 2.0],
                "merge_components_footprint_bins": [2],
                "split_components_footprint_bins": [2],
            },
            "correlated_noise": "gaussian_filter(white noise, sigma=1.5 bins), unit variance, scaled to factor*std(GT)",
            "merge_split": "morphological closing/opening of the p90 hotspot mask; map values inside changed regions set above/below threshold",
        },
        "leakage_guards": {
            "target_expression_never_used_in": [
                "module gene lists", "z-score mean/std", "hotspot threshold",
                "boundary threshold", "MS-SSIM constants", "any calibration"],
            "audit": "target X equals observed_50um rows for the held-out slices (verified); reference stats recomputed from S1/S5 equals stored",
            "predictions_used_as_is": True,
        },
        "statistics": {
            "slices": 5,
            "combos_reuse_slices": True,
            "independent_replicates": False,
            "bootstrap": "2000 resamples over 5 slice ids, paired by slice, module PCC",
            "multiple_comparison": "none; descriptive per-combo/module effects only",
        },
        "runtime": {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))},
    }
    cfg_path = os.path.join(OUT, "metric_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    audit_summary["config_sha256"] = M.md5_file(cfg_path)
    audit_summary["config_written_utc"] = cfg["written_utc"]
    with open(os.path.join(OUT, "audit_summary.json"), "w") as f:
        json.dump(audit_summary, f, indent=2, default=M.jsonable)
    print(f"audit checks: {len(AUDIT_ROWS)} pass={n_pass} fail={n_fail} info={n_na}")
    for r in AUDIT_ROWS:
        if r["pass"] is False:
            print("  FAIL", r["section"], r["check"], r["combo"], r["method"],
                  r["value"], r["expected"], r["note"])
    print("wrote", cfg_path)
    print("elapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()

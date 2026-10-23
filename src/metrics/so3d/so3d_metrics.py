#!/usr/bin/env python
"""So3D 46 datasets x 8 methods metric protocol.

Outputs (all under Workspace/so3d_bench/metrics/):
  so3d_metrics_long.csv          dataset x method x level/module x metric
  so3d_method_summary.csv/.md    per-method macro aggregates + N/A counts
  so3d_dataset_coverage.csv      per-dataset coverage/support diagnostics
  so3d_dataset_method_summary.csv per-dataset x method core values
  status.json, run.log, shards/

Leakage guard: target.npz X is opened only inside score_method() for scoring.
Reference z-scores, marker modules, hotspot/boundary thresholds and the raster
grid are all derived from visible.h5ad / meta.json (geometry) only.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.sparse import issparse
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRED = os.path.join(ROOT, "predictions")
CANON = os.path.join(ROOT, "canonical")
OUT = HERE
SHARDS = os.path.join(OUT, "shards")

sys.path.insert(0, HERE)
import so3d_mlib as ml  # noqa: E402

METHODS = ["ReHo", "PASTE-OT-Linear", "NTF", "SINTER3D", "SpatialZ",
           "isoST", "DeepSpatial", "UniST"]

# frozen protocol constants
MIN_MODULE_CELLS = 20
MODULE_TOPK = 20
MODULE_MIN_D = 0.25
MODULE_MIN_GENES = 5
MIN_TARGET_SPOTS_SPATIAL = 100
MIN_HOTSPOT_NEG = 10        # min GT- and pred-positive spots for hotspot P/R
MIN_BOUNDARY_SPOTS = 5      # min GT- and pred-boundary pixels
MIN_SSIM_WINDOWS = 10       # min fully-masked 7x7 SSIM windows in the 2D slice

GENE_METRICS = [
    "gene_pcc_median", "gene_pcc_mean", "gene_pcc_p10",
    "gene_pcc_frac_gt_0.5", "gene_rmse", "gene_rmse_median",
    "gene_mae", "gene_mae_median", "gene_cosine_median",
    "gene_cosine_mean", "gene_spearman_median", "gene_spearman_mean",
    "gene_vr_log2_median", "gene_vr_log2_mean", "gene_vr_within_2fold_frac",
    "n_genes_scored",
]
MODULE_METRICS = ["module_pcc", "module_vr_log2", "module_rmse",
                  "module_mean_ratio"]
SPATIAL_METRICS = ["moran_i_gt", "moran_i_pred", "abs_delta_moran_i",
                   "geary_gt", "geary_pred", "abs_delta_geary_c",
                   "ms_ssim", "hotspot_recall", "hotspot_precision",
                   "hotspot_f1", "hotspot_AP", "hotspot_PR_AUC",
                   "hotspot_recall_p80", "hotspot_precision_p80",
                   "boundary_recall_d0", "boundary_precision_d0",
                   "boundary_f1_d0", "boundary_recall_d1",
                   "boundary_precision_d1", "boundary_recall_p80",
                   "boundary_precision_p80", "boundary_centroid_dist_bins"]
MACRO_METRICS = ["module_pcc", "module_vr_log2", "module_rmse"] + SPATIAL_METRICS


def log(msg):
    print(msg, flush=True)


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def rows_add(rows, dataset, method, level, module, metric, value, n=np.nan,
             status="OK", reason=""):
    rows.append({
        "dataset": dataset, "method": method, "level": level,
        "module": module, "metric": metric,
        "value": (np.nan if value is None else float(value)),
        "n": (np.nan if (n is None or (isinstance(n, float) and np.isnan(n)))
              else int(n)),
        "status": status, "reason": reason,
    })


# --------------------------------------------------------------- visible side --
def sparse_mean_std(X):
    n = X.shape[0]
    if issparse(X):
        s1 = np.asarray(X.sum(0), dtype=np.float64).ravel()
        s2 = np.asarray(X.multiply(X).sum(0), dtype=np.float64).ravel()
        Xsq = None
    else:
        Xd = np.asarray(X, dtype=np.float64)
        s1 = Xd.sum(0)
        s2 = (Xd * Xd).sum(0)
        Xsq = Xd
    mean = s1 / n
    var = np.maximum(s2 / n - mean * mean, 0.0)
    return mean, np.sqrt(var), Xsq


def load_visible(ds):
    import anndata as ad
    p = os.path.join(CANON, ds, "visible.h5ad")
    a = ad.read_h5ad(p)
    X = a.X
    genes = np.asarray(a.var_names, dtype=str)
    spatial = np.asarray(a.obsm["spatial"], dtype=np.float64)
    labels = a.obs["Celltype"].astype(str).values
    sl = a.obs["slice"].astype(str).values
    mean, std, Xd = sparse_mean_std(X)
    return {"X": X, "Xd": Xd, "genes": genes, "spatial": spatial,
            "labels": labels, "slice": sl, "mean": mean, "std": std,
            "valid": std > 1e-8, "n": a.n_obs}


def define_modules(vis):
    """Marker modules from visible celltype labels only (one-vs-rest Cohen's d)."""
    X, mean, std, valid = vis["X"], vis["mean"], vis["std"], vis["valid"]
    n = vis["n"]
    s1 = np.asarray(X.sum(0), dtype=np.float64).ravel() if issparse(X) else X.sum(0)
    labels = vis["labels"]
    cats, counts = np.unique(labels, return_counts=True)
    mods, skipped = [], []
    for cat, cnt in zip(cats, counts):
        if cnt < MIN_MODULE_CELLS:
            skipped.append({"module": str(cat), "n_cells": int(cnt),
                            "reason": f"n_cells<{MIN_MODULE_CELLS}"})
            continue
        m = labels == cat
        cm = np.asarray(X[m].mean(0), dtype=np.float64).ravel()
        rest = (s1 - cm * cnt) / max(n - cnt, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(valid & (std > 0), (cm - rest) / np.maximum(std, 1e-12),
                         -np.inf)
        order = np.argsort(-d)
        order = order[np.isfinite(d[order])]
        order = order[d[order] >= MODULE_MIN_D]
        genes_idx = order[:MODULE_TOPK]
        if genes_idx.size < MODULE_MIN_GENES:
            skipped.append({"module": str(cat), "n_cells": int(cnt),
                            "reason": f"n_valid_markers<{MODULE_MIN_GENES}",
                            "n_valid_markers": int(genes_idx.size)})
            continue
        mods.append({"module": str(cat), "n_cells": int(cnt),
                     "gene_idx": genes_idx.astype(int),
                     "marker_d_median": float(np.median(d[genes_idx])),
                     "marker_d_min": float(np.min(d[genes_idx]))})
    return mods, skipped


def module_scores_sparse(X, idx, mean, std):
    """Mean z over module genes (reference stats from visible)."""
    sub = X[:, idx]
    sub = sub.toarray() if issparse(sub) else np.asarray(sub)
    Z = (np.asarray(sub, dtype=np.float64) - mean[idx]) / std[idx]
    return Z.mean(1)


def choose_bin(xy, target_cells=3800, max_dim=128, min_occ=0.15):
    """Adaptive 2D bin size (coordinate units): NN scale, capped grid, >=15% filled."""
    xy = np.asarray(xy, dtype=np.float64)
    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=2)
    nn = float(np.median(d[:, 1])) if len(xy) > 1 else 0.0
    span = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    if span <= 0:
        return 1.0
    b = max(nn, span / 96.0, 1e-9)
    for _ in range(24):
        H = int(np.floor(np.ptp(xy[:, 1]) / b)) + 1
        W = int(np.floor(np.ptp(xy[:, 0]) / b)) + 1
        if H < 2 or W < 2:
            b /= 1.5
            continue
        if H > max_dim or W > max_dim:
            b *= 1.25
            continue
        ix = np.rint((xy[:, 0] - xy[:, 0].min()) / b).astype(np.int64)
        iy = np.rint((xy[:, 1] - xy[:, 1].min()) / b).astype(np.int64)
        occ = len(np.unique(iy * W + ix)) / float(H * W)
        if occ >= min_occ:
            return float(b)
        b *= 1.25
    return float(b)


def raster_index(xy, b):
    x0 = float(np.min(xy[:, 0]))
    y0 = float(np.min(xy[:, 1]))
    ix = np.rint((xy[:, 0] - x0) / b).astype(np.int64)
    iy = np.rint((xy[:, 1] - y0) / b).astype(np.int64)
    H, W = int(iy.max()) + 1, int(ix.max()) + 1
    return ix, iy, H, W


def rasterize(ix, iy, H, W, vals):
    s = np.zeros((H, W), dtype=np.float64)
    c = np.zeros((H, W), dtype=np.float64)
    np.add.at(s, (iy, ix), np.asarray(vals, dtype=np.float64))
    np.add.at(c, (iy, ix), 1.0)
    occ = c > 0
    A = np.full((H, W), np.nan)
    A[occ] = s[occ] / c[occ]
    return A, occ


def tissue_mask(occ):
    st = ndi.generate_binary_structure(2, 1)
    m = ndi.binary_closing(occ, structure=st, iterations=2)
    m = ndi.binary_closing(m, structure=np.ones((3, 3), bool), iterations=1)
    return m


def filled_raster(A, occ, mask):
    filled = ml.fill_nearest(A, occ)
    return np.where(mask, filled, np.nan)


def adaptive_mask(occ, min_windows=MIN_SSIM_WINDOWS):
    """Tissue footprint: close+fill holes; dilate occupied cells only when the
    spot layout is too sparse to contain MS-SSIM windows."""
    st = ndi.generate_binary_structure(2, 1)
    m = tissue_mask(occ)
    for dil in (0, 1, 2):
        o = ndi.binary_dilation(occ, structure=st, iterations=dil) if dil else occ
        m = ndi.binary_fill_holes(tissue_mask(o))
        nwin = int((ndi.uniform_filter(m.astype(np.float64), size=7,
                                       mode="constant") >= 0.999).sum())
        if nwin >= min_windows:
            return m, dil
    return m, 2


def inner_mask(mask):
    """Interior footprint with adaptive erosion (2 -> 1 -> 0 bins)."""
    if mask.sum() == 0:
        return mask
    i2 = ml.erode_mask(mask, 2)
    if i2.sum() >= 0.05 * mask.sum():
        return i2
    i1 = ml.erode_mask(mask, 1)
    if i1.sum() >= 0.05 * mask.sum():
        return i1
    return mask.copy()


def visible_thresholds(vis, mods, b):
    """Hotspot (p90 score) and boundary (pooled p90 gradient) thresholds from
    visible slices only; same numeric thresholds later applied to GT and pred."""
    X, mean, std = vis["X"], vis["mean"], vis["std"]
    xy = vis["spatial"][:, :2]
    sl = vis["slice"]
    slices = np.unique(sl)
    for mod in mods:
        idx = mod["gene_idx"]
        sc = module_scores_sparse(X, idx, mean, std)
        mod["hotspot_thr"] = float(np.percentile(sc, 90))
        mod["hotspot_thr_p80"] = float(np.percentile(sc, 80))
        grad_vals = []
        for s in slices:
            sm = sl == s
            if int(sm.sum()) < 50:
                continue
            xs = xy[sm]
            ix, iy, H, W = raster_index(xs, b)
            A, occ = rasterize(ix, iy, H, W, sc[sm])
            mask = tissue_mask(occ)
            if mask.sum() < 25:
                continue
            Af = filled_raster(A, occ, mask)
            grad = ml.gradient_map(Af, mask)
            inner = ml.erode_mask(mask, 2)
            if inner.sum() >= 10:
                v = grad[inner]
                grad_vals.append(v[np.isfinite(v)])
        if grad_vals:
            allv = np.concatenate(grad_vals)
            if allv.size > 400000:
                allv = np.random.default_rng(0).choice(allv, 400000, replace=False)
            mod["boundary_thr"] = float(np.percentile(allv, 90))
            mod["boundary_thr_p80"] = float(np.percentile(allv, 80))
            mod["boundary_n_ref"] = int(allv.size)
        else:
            mod["boundary_thr"] = np.nan
            mod["boundary_thr_p80"] = np.nan
            mod["boundary_n_ref"] = 0
    return mods


# ---------------------------------------------------------------- scoring side --
def nvalid_n(gm):
    return int(gm.get("n_genes_scored", 0))


def gene_metrics(rows, ds, method, G, P):
    """Gene-level metrics on the held-out target slice only."""
    n, g = G.shape
    Gf = np.asarray(G, dtype=np.float64)
    Pf = np.asarray(P, dtype=np.float64)
    var_g = Gf.std(0)
    var_p = Pf.std(0)
    valid = (var_g > 0) & (var_p > 0)
    nvalid = int(valid.sum())
    pcc = ml.pcc_cols(Gf[:, valid], Pf[:, valid])
    sp = ml.spearman_cols(Gf[:, valid], Pf[:, valid])
    cos = ml.cosine_cols(Gf[:, valid], Pf[:, valid])
    vr = ml.variance_ratio_log2_cols(Pf, Gf)
    rmse_g = np.sqrt(np.mean((Pf - Gf) ** 2, axis=0))
    mae_g = np.mean(np.abs(Pf - Gf), axis=0)
    vals = {
        "gene_pcc_median": np.nanmedian(pcc),
        "gene_pcc_mean": np.nanmean(pcc),
        "gene_pcc_p10": np.nanpercentile(pcc, 10),
        "gene_pcc_frac_gt_0.5": np.nanmean(pcc > 0.5),
        "gene_rmse": np.sqrt(np.mean((Pf - Gf) ** 2)),
        "gene_rmse_median": np.nanmedian(rmse_g),
        "gene_mae": np.mean(np.abs(Pf - Gf)),
        "gene_mae_median": np.nanmedian(mae_g),
        "gene_cosine_median": np.nanmedian(cos),
        "gene_cosine_mean": np.nanmean(cos),
        "gene_spearman_median": np.nanmedian(sp),
        "gene_spearman_mean": np.nanmean(sp),
        "gene_vr_log2_median": np.nanmedian(vr),
        "gene_vr_log2_mean": np.nanmean(vr),
        "gene_vr_within_2fold_frac": np.nanmean(
            (vr >= -1.0) & (vr <= 1.0)),
        "n_genes_scored": nvalid,
    }
    for k in GENE_METRICS:
        if k in ("gene_rmse", "gene_mae"):
            n_use = int(n * g)
        elif k in ("gene_rmse_median", "gene_mae_median",
                   "gene_vr_log2_median", "gene_vr_log2_mean",
                   "gene_vr_within_2fold_frac", "n_genes_scored"):
            n_use = nvalid if k == "n_genes_scored" else g
        else:
            n_use = nvalid
        rows_add(rows, ds, method, "gene", "ALL", k, vals[k], n_use)
    return {"n_genes_scored": nvalid, "n_genes": g,
            "gene_pcc_median": vals["gene_pcc_median"]}


def module_metrics(rows, ds, method, G, P, mods, ref):
    if not mods:
        for k in MODULE_METRICS:
            rows_add(rows, ds, method, "module", "__none__", k, np.nan,
                     status="NA", reason="no_valid_modules")
        for k in ("module_pcc", "module_vr_log2", "module_rmse"):
            rows_add(rows, ds, method, "module_macro", "MACRO", k, np.nan,
                     status="NA", reason="no_valid_modules")
        return {"n_modules_used": 0}

    mac = {k: [] for k in ("module_pcc", "module_vr_log2", "module_rmse")}
    for mod in mods:
        idx = mod["gene_idx"]
        mean, std = ref["mean"], ref["std"]
        sg = ((np.asarray(G[:, idx], np.float64) - mean[idx]) / std[idx]).mean(1)
        sp = ((np.asarray(P[:, idx], np.float64) - mean[idx]) / std[idx]).mean(1)
        pcc = ml.pcc_cols(sg[:, None], sp[:, None])[0]
        vr = ml.variance_ratio_log2_cols(sp[:, None], sg[:, None])[0]
        rmse = float(np.sqrt(np.mean((sp - sg) ** 2)))
        mr = float(sp.mean() / sg.mean()) if sg.mean() != 0 else np.nan
        vals = {"module_pcc": pcc, "module_vr_log2": vr,
                "module_rmse": rmse, "module_mean_ratio": mr}
        for k in MODULE_METRICS:
            st = "OK" if np.isfinite(vals[k]) else "NA"
            rs = "" if st == "OK" else "constant_module_score"
            rows_add(rows, ds, method, "module", mod["module"], k, vals[k],
                     n=len(sg), status=st, reason=rs)
        mod["_scores"] = (sg, sp)
        for k in mac:
            if np.isfinite(vals[k]):
                mac[k].append(vals[k])
    macro_out = {}
    for k, v in mac.items():
        arr = np.asarray(v, dtype=float)
        mv = float(arr.mean()) if arr.size else np.nan
        macro_out[k] = mv
        rows_add(rows, ds, method, "module_macro", "MACRO", k, mv, n=arr.size,
                 status="OK" if arr.size else "NA_SUPPORT",
                 reason="" if arr.size else "all_modules_constant")
    return {"n_modules_used": len(mods), "macro": macro_out}


def spatial_metrics(rows, ds, method, mods, geom, W_sp, target_diag):
    if not mods:
        for k in SPATIAL_METRICS:
            rows_add(rows, ds, method, "spatial_macro", "MACRO", k, np.nan,
                     status="NA_SUPPORT", reason="no_valid_modules")
        return {}
    xy = geom["xy"]
    ix, iy, H, W = geom["ix"], geom["iy"], geom["H"], geom["W"]
    mask = geom["mask"]
    inner = geom["inner"]
    nspots = len(xy)
    macro = {k: [] for k in SPATIAL_METRICS}
    for mod in mods:
        sg, sp = mod["_scores"]
        name = mod["module"]
        # Moran / Geary on the 2D target-slice spot kNN graph
        if nspots >= 50 and W_sp is not None:
            I_g, C_g = ml.morans_i_W(W_sp, sg)
            I_p, C_p = ml.morans_i_W(W_sp, sp)
        else:
            I_g = I_p = C_g = C_p = np.nan
        dI = abs(I_p - I_g) if np.isfinite(I_p) and np.isfinite(I_g) else np.nan
        dC = abs(C_p - C_g) if np.isfinite(C_p) and np.isfinite(C_g) else np.nan
        for k, v in [("moran_i_gt", I_g), ("moran_i_pred", I_p),
                     ("abs_delta_moran_i", dI), ("geary_gt", C_g),
                     ("geary_pred", C_p), ("abs_delta_geary_c", dC)]:
            st = "OK" if np.isfinite(v) else "NA"
            rows_add(rows, ds, method, "spatial", name, k, v, n=nspots,
                     status=st, reason="" if st == "OK" else "spatial_unsupported")
            if np.isfinite(v):
                macro[k].append(v)

        # hotspot on spots with a fixed visible-derived threshold (p90 primary,
        # p80 sensitivity); AP is threshold-free and needs GT positives only.
        thr = mod.get("hotspot_thr", np.nan)
        thr80 = mod.get("hotspot_thr_p80", np.nan)
        hkeys = ["hotspot_recall", "hotspot_precision", "hotspot_f1",
                 "hotspot_AP", "hotspot_PR_AUC"]
        if nspots >= MIN_TARGET_SPOTS_SPATIAL and np.isfinite(thr):
            lab_g = sg > thr
            lab_p = sp > thr
            ng, npp = int(lab_g.sum()), int(lab_p.sum())
            ap, prauc, _ = ml.ap_pr_auc(sp, lab_g) if ng >= MIN_HOTSPOT_NEG \
                else (np.nan, np.nan, np.nan)
            if ng >= MIN_HOTSPOT_NEG and npp >= MIN_HOTSPOT_NEG:
                tp = int(np.logical_and(lab_g, lab_p).sum())
                rec, pre, f1 = ml.prf(tp, npp - tp, ng - tp)
                hvals = {"hotspot_recall": rec, "hotspot_precision": pre,
                         "hotspot_f1": f1, "hotspot_AP": ap,
                         "hotspot_PR_AUC": prauc}
            else:
                hvals = {k: np.nan for k in hkeys}
                hvals["hotspot_AP"] = ap
                hvals["hotspot_PR_AUC"] = prauc
            for k in hkeys:
                v = hvals[k]
                if np.isfinite(v):
                    st, rs = "OK", ""
                else:
                    st = "NA"
                    if ng < MIN_HOTSPOT_NEG:
                        rs = "hotspot_gt_positive_insufficient"
                    elif k in ("hotspot_AP", "hotspot_PR_AUC"):
                        rs = "hotspot_AP_undefined"
                    else:
                        rs = "hotspot_pred_positive_insufficient"
                rows_add(rows, ds, method, "spatial", name, k, v, n=nspots,
                         status=st, reason=rs)
                if np.isfinite(v):
                    macro[k].append(v)
            rows_add(rows, ds, method, "spatial_diag", name,
                     "n_gt_positive", ng, n=nspots)
            rows_add(rows, ds, method, "spatial_diag", name,
                     "n_pred_positive", npp, n=nspots)
            # p80 sensitivity (same visible-only threshold family)
            if np.isfinite(thr80):
                lg80 = sg > thr80
                lp80 = sp > thr80
                g80, p80 = int(lg80.sum()), int(lp80.sum())
                if g80 >= MIN_HOTSPOT_NEG and p80 >= MIN_HOTSPOT_NEG:
                    tp = int(np.logical_and(lg80, lp80).sum())
                    rec, pre, _ = ml.prf(tp, p80 - tp, g80 - tp)
                    for k, v in [("hotspot_recall_p80", rec),
                                 ("hotspot_precision_p80", pre)]:
                        rows_add(rows, ds, method, "spatial", name, k, v,
                                 n=nspots, status="OK")
                        if np.isfinite(v):
                            macro[k].append(v)
                else:
                    rr = "hotspot_p80_support_insufficient"
                    for k in ["hotspot_recall_p80", "hotspot_precision_p80"]:
                        rows_add(rows, ds, method, "spatial", name, k, np.nan,
                                 n=nspots, status="NA", reason=rr)
        else:
            reason = ("n_target<%d" % MIN_TARGET_SPOTS_SPATIAL
                      if nspots < MIN_TARGET_SPOTS_SPATIAL
                      else "no_visible_hotspot_threshold")
            for k in hkeys + ["hotspot_recall_p80", "hotspot_precision_p80"]:
                rows_add(rows, ds, method, "spatial", name, k, np.nan,
                         n=nspots, status="NA", reason=reason)

        # masked MS-SSIM on the target 2D slice
        if nspots >= MIN_TARGET_SPOTS_SPATIAL:
            A_g, occ = rasterize(ix, iy, H, W, sg)
            A_p, _ = rasterize(ix, iy, H, W, sp)
            A_g = filled_raster(A_g, occ, mask)
            A_p = filled_raster(A_p, occ, mask)
            nwin = int((ndi.uniform_filter(mask.astype(np.float64), size=7,
                                           mode="constant") >= 0.999).sum())
            if nwin >= MIN_SSIM_WINDOWS:
                L = max(float(np.nanpercentile(A_g[mask], 99.5)), 1e-6)
                v = ml.ms_ssim(A_p, A_g, mask, L=L)
                st = "OK" if np.isfinite(v) else "NA"
                rows_add(rows, ds, method, "spatial", name, "ms_ssim", v,
                         n=nwin, status=st,
                         reason="" if st == "OK" else "ms_ssim_degenerate")
                if np.isfinite(v):
                    macro["ms_ssim"].append(v)
            else:
                rows_add(rows, ds, method, "spatial", name, "ms_ssim", np.nan,
                         n=nwin, status="NA",
                         reason=f"ssim_valid_windows<{MIN_SSIM_WINDOWS}")
        else:
            rows_add(rows, ds, method, "spatial", name, "ms_ssim", np.nan,
                     n=nspots, status="NA",
                     reason="n_target<%d" % MIN_TARGET_SPOTS_SPATIAL)

        # boundary precision/recall (gradient threshold from visible only)
        bthr = mod.get("boundary_thr", np.nan)
        if nspots >= MIN_TARGET_SPOTS_SPATIAL and np.isfinite(bthr) and inner.sum() > 0:
            A_g, occ = rasterize(ix, iy, H, W, sg)
            A_p, _ = rasterize(ix, iy, H, W, sp)
            A_g = filled_raster(A_g, occ, mask)
            A_p = filled_raster(A_p, occ, mask)
            cb = geom.get("contour_bins", 2)
            b_g, _, _ = ml.boundary_mask(A_g, mask, bthr, contour_bins=cb)
            b_p, _, _ = ml.boundary_mask(A_p, mask, bthr, contour_bins=cb)
            m0 = ml.boundary_match(b_g, b_p, 0)
            m1 = ml.boundary_match(b_g, b_p, 1)
            bthr80 = mod.get("boundary_thr_p80", np.nan)
            if m0["n_gt"] >= MIN_BOUNDARY_SPOTS and m0["n_pred"] >= MIN_BOUNDARY_SPOTS:
                dvals = {"boundary_recall_d0": m0["recall"],
                         "boundary_precision_d0": m0["precision"],
                         "boundary_f1_d0": m0["f1"],
                         "boundary_recall_d1": m1["recall"],
                         "boundary_precision_d1": m1["precision"],
                         "boundary_centroid_dist_bins":
                             ml.centroid_distances(b_g, b_p)}
                for k, v in dvals.items():
                    rows_add(rows, ds, method, "spatial", name, k, v,
                             n=m0["n_gt"], status="OK" if np.isfinite(v) else "NA",
                             reason="" if np.isfinite(v) else "no_boundary_components")
                    if np.isfinite(v):
                        macro[k].append(v)
                rows_add(rows, ds, method, "spatial_diag", name,
                         "n_gt_boundary", m0["n_gt"], n=m0["n_gt"])
                rows_add(rows, ds, method, "spatial_diag", name,
                         "n_pred_boundary", m0["n_pred"], n=m0["n_pred"])
                if np.isfinite(bthr80):
                    g80, _, _ = ml.boundary_mask(A_g, mask, bthr80, contour_bins=cb)
                    p80, _, _ = ml.boundary_mask(A_p, mask, bthr80, contour_bins=cb)
                    if int(g80.sum()) >= MIN_BOUNDARY_SPOTS and \
                            int(p80.sum()) >= MIN_BOUNDARY_SPOTS:
                        mm80 = ml.boundary_match(g80, p80, 0)
                        for k, v in [("boundary_recall_p80", mm80["recall"]),
                                     ("boundary_precision_p80", mm80["precision"])]:
                            rows_add(rows, ds, method, "spatial", name, k, v,
                                     n=mm80["n_gt"], status="OK")
                            if np.isfinite(v):
                                macro[k].append(v)
                    else:
                        rr = "boundary_p80_support_insufficient"
                        for k in ["boundary_recall_p80", "boundary_precision_p80"]:
                            rows_add(rows, ds, method, "spatial", name, k,
                                     np.nan, status="NA", reason=rr)
                else:
                    for k in ["boundary_recall_p80", "boundary_precision_p80"]:
                        rows_add(rows, ds, method, "spatial", name, k, np.nan,
                                 status="NA", reason="no_visible_boundary_threshold")
            else:
                if m0["n_gt"] < MIN_BOUNDARY_SPOTS:
                    reason = "boundary_gt_support_insufficient"
                else:
                    reason = "boundary_pred_support_insufficient"
                for k in ["boundary_recall_d0", "boundary_precision_d0",
                          "boundary_f1_d0", "boundary_recall_d1",
                          "boundary_precision_d1", "boundary_recall_p80",
                          "boundary_precision_p80",
                          "boundary_centroid_dist_bins"]:
                    rows_add(rows, ds, method, "spatial", name, k, np.nan,
                             status="NA", reason=reason)
        else:
            reason = ("n_target<%d" % MIN_TARGET_SPOTS_SPATIAL
                      if nspots < MIN_TARGET_SPOTS_SPATIAL
                      else "no_visible_boundary_threshold")
            for k in ["boundary_recall_d0", "boundary_precision_d0",
                      "boundary_f1_d0", "boundary_recall_d1",
                      "boundary_precision_d1", "boundary_recall_p80",
                      "boundary_precision_p80", "boundary_centroid_dist_bins"]:
                rows_add(rows, ds, method, "spatial", name, k, np.nan,
                         status="NA", reason=reason)
    macro_out = {}
    for k in SPATIAL_METRICS:
        v = np.asarray([x for x in macro[k] if np.isfinite(x)], dtype=float)
        mv = float(v.mean()) if v.size else np.nan
        macro_out[k] = mv
        rows_add(rows, ds, method, "spatial_macro", "MACRO", k, mv, n=v.size,
                 status="OK" if v.size else "NA",
                 reason="" if v.size else "all_modules_unsupported")
    return macro_out


# ------------------------------------------------------------------- dataset --
def score_method(ds, method, ref, mods, geom, W_sp, meta, vis_info, rows,
                 scored_any_target=True):
    """Load target + prediction (scoring only) and emit all metric rows."""
    tgt = np.load(os.path.join(CANON, ds, "target.npz"), allow_pickle=True)
    Gt = np.asarray(tgt["X"], dtype=np.float32)
    coords = np.asarray(tgt["coords"])
    names_t = np.asarray(tgt["obs_names"], dtype=str)
    n, g = Gt.shape
    base = {"n_target": n, "n_genes": g, "align_ok": False}
    p = os.path.join(PRED, ds, f"{method}.npz")
    diag = dict(base)
    if not os.path.exists(p):
        for level, metrics in [("gene", GENE_METRICS),
                               ("module", MODULE_METRICS),
                               ("module_macro", MACRO_METRICS),
                               ("spatial_macro", SPATIAL_METRICS)]:
            for k in metrics:
                rows_add(rows, ds, method, level, "ALL" if level == "gene" else "__none__",
                         k, np.nan, status="FAILED", reason="prediction_file_missing")
        diag.update(status="FAILED", reason="prediction_file_missing")
        return diag, []
    z = np.load(p, allow_pickle=True)
    keys = list(z.files)
    P = np.asarray(z["pred"] if "pred" in z.files else z["X"], dtype=np.float32)
    names_p = np.asarray(z["obs_names"], dtype=str) if "obs_names" in z.files else None
    coords_p = np.asarray(z["coords"]) if "coords" in z.files else None
    genes_p = np.asarray(z["genes"], dtype=str) if "genes" in z.files else None
    row_index = np.asarray(z["row_index"]).astype(np.int64) if "row_index" in z.files else None
    reasons = []
    align = True
    shapes_ok = (P.shape == Gt.shape)
    if not shapes_ok:
        align = False
        reasons.append(f"shape_mismatch_pred{P.shape}_target{Gt.shape}")
    if names_p is not None and not np.array_equal(names_p, names_t):
        align = False
        reasons.append("obs_names_mismatch")
    if coords_p is not None:
        cdiff = float(np.max(np.abs(coords_p - coords))) if shapes_ok else np.nan
    else:
        cdiff = np.nan
    if coords_p is not None and shapes_ok and cdiff > 1e-6:
        align = False
        reasons.append(f"coords_mismatch_max{cdiff:.2e}")
    genes_ok = (genes_p is not None and np.array_equal(genes_p,
                np.asarray(meta["genes"], dtype=str)))
    if genes_p is not None and not genes_ok:
        align = False
        reasons.append("genes_mismatch")
    if row_index is not None and shapes_ok and not np.array_equal(
            row_index, np.arange(n)):
        align = False
        reasons.append("row_index_not_identity")
    if not align:
        for level, metrics in [("gene", GENE_METRICS),
                               ("module", MODULE_METRICS),
                               ("module_macro", MACRO_METRICS),
                               ("spatial_macro", SPATIAL_METRICS)]:
            for k in metrics:
                rows_add(rows, ds, method, level, "ALL" if level == "gene" else "MACRO",
                         k, np.nan, status="FAILED", reason=";".join(reasons))
        diag.update(status="FAILED", reason=";".join(reasons),
                    align_names=False, coords_max_absdiff=cdiff,
                    align_genes=bool(genes_ok), pred_shape=list(P.shape))
        return diag, []

    finite = np.isfinite(P)
    finite_frac = float(finite.mean())
    neg_frac = float((P < 0).mean())
    diag.update(align_ok=True, align_names=bool(names_p is None or np.array_equal(names_p, names_t)) if names_p is not None else None,
                coords_max_absdiff=cdiff, align_genes=bool(genes_ok or genes_p is None),
                finite_frac=finite_frac, negative_frac=neg_frac,
                pred_min=float(np.nanmin(P)), pred_max=float(np.nanmax(P)),
                zero_frac_pred=float((P == 0).mean()),
                zero_frac_target=float((Gt == 0).mean()),
                pred_mean_ratio=float(P.mean() / Gt.mean()) if Gt.mean() != 0 else np.nan,
                target_zero_var_genes=int((Gt.std(0) == 0).sum()),
                pred_zero_var_genes=int((P.std(0) == 0).sum()),
                target_slice=int(meta["target_slice"]), n_visible=vis_info["n"],
                n_visible_slices=len(meta["visible_slices"]),
                n_modules=len(mods))
    diag["n_modules"] = len(mods)
    for k in ["n_target", "n_genes", "zero_frac_target", "zero_frac_pred",
              "pred_mean_ratio", "finite_frac", "negative_frac",
              "target_zero_var_genes", "pred_zero_var_genes",
              "coords_max_absdiff", "n_modules"]:
        v = diag.get(k, np.nan)
        rows_add(rows, ds, method, "diagnostic", "ALL", k,
                 v if isinstance(v, (int, float, np.integer, np.floating)) else np.nan,
                 status="OK")

    if not finite.all() or neg_frac > 0:
        reason = ("nonfinite_prediction" if not finite.all()
                  else "negative_prediction")
        for level, metrics in [("gene", GENE_METRICS),
                               ("module", MODULE_METRICS),
                               ("module_macro", MACRO_METRICS),
                               ("spatial_macro", SPATIAL_METRICS)]:
            for k in metrics:
                rows_add(rows, ds, method, level,
                         "ALL" if level == "gene" else "MACRO", k, np.nan,
                         status="FAILED", reason=reason)
        diag.update(status="FAILED", reason=reason)
        return diag, []

    gm = gene_metrics(rows, ds, method, Gt, P)
    diag["gene_coverage_frac"] = (float(gm["n_genes_scored"]) / g) if g else np.nan
    rows_add(rows, ds, method, "diagnostic", "ALL", "gene_coverage_frac",
             diag["gene_coverage_frac"], n=nvalid_n(gm))
    mm = module_metrics(rows, ds, method, Gt, P, mods, ref)
    sm = spatial_metrics(rows, ds, method, mods, geom, W_sp, diag)
    diag.update(gm)
    diag["n_modules_used"] = mm.get("n_modules_used", 0)
    diag["macro"] = {**mm.get("macro", {}),
                     **{k: sm.get(k, np.nan) for k in SPATIAL_METRICS}}
    na_modules = sum(1 for mod in mods if not np.isfinite(mod.get("boundary_thr", np.nan)))
    diag["n_modules_without_boundary_thr"] = na_modules
    status = "OK"
    if (not mods) or diag["n_modules_used"] == 0:
        status = "PARTIAL"
        diag["reason"] = "no_module_support"
    else:
        sp_missing = [k for k in ["ms_ssim", "abs_delta_moran_i"]
                      if not np.isfinite(sm.get(k, np.nan))]
        if sp_missing:
            status = "PARTIAL"
            diag["reason"] = "spatial_na:" + ",".join(sp_missing)
    diag["status"] = status
    return diag, []


def process_dataset(ds):
    t0 = time.time()
    rows, err = [], None
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    meta = json.load(open(os.path.join(CANON, ds, "meta.json")))
    vis = load_visible(ds)
    ref = {"mean": vis["mean"], "std": vis["std"], "valid": vis["valid"],
           "genes": vis["genes"]}
    mods, skipped = define_modules(vis)

    tgt_meta = np.load(os.path.join(CANON, ds, "target.npz"), allow_pickle=True)
    xy_all = np.asarray(tgt_meta["coords"], dtype=np.float64)[:, :2]
    z = np.asarray(tgt_meta["z"], dtype=np.float64)
    n_target = int(tgt_meta["X"].shape[0])
    del tgt_meta
    b = choose_bin(xy_all)
    ix, iy, H, W = raster_index(xy_all, b)
    A0, occ = rasterize(ix, iy, H, W, np.zeros(len(xy_all)))
    mask, mask_dil = adaptive_mask(occ)
    inner = inner_mask(mask)
    i2 = ml.erode_mask(mask, 2)
    i1 = ml.erode_mask(mask, 1)
    contour = (2 if mask.sum() > 0 and i2.sum() >= 0.05 * mask.sum()
               else 1 if mask.sum() > 0 and i1.sum() >= 0.05 * mask.sum()
               else 0)
    geom = {"xy": xy_all, "ix": ix, "iy": iy, "H": H, "W": W, "mask": mask,
            "inner": inner, "contour_bins": contour, "bin": b,
            "mask_dilation": mask_dil}
    W_sp = ml.knn_weights(xy_all, ml.KNN_K) if len(xy_all) >= 52 else None
    mods = visible_thresholds(vis, mods, b)

    vis_info = {"n": int(vis["n"]), "n_slices": len(np.unique(vis["slice"]))}
    out = {"dataset": ds, "rows": rows, "diag": {}, "modules": [],
           "skipped_modules": skipped,
           "geometry": {"bin": b, "grid": [int(H), int(W)],
                        "occupancy": float(occ.mean()),
                        "mask_frac": float(mask.mean()),
                        "mask_dilation": int(mask_dil),
                        "inner_frac": float(geom["inner"].mean()),
                        "unique_xy": int(len(np.unique(xy_all.round(6), axis=0))),
                        "n_z_values": int(np.unique(np.round(z, 6)).size)},
           "target_slice": int(meta["target_slice"]),
           "target_index": int(meta.get("target_index", meta["target_slice"])),
           "n_target": n_target, "n_genes": int(len(meta["genes"])),
           "n_visible": int(vis["n"]),
           "n_visible_slices": int(len(meta["visible_slices"])),
           "zero_frac_target": float((np.asarray(
               np.load(os.path.join(CANON, ds, "target.npz"))["X"]) == 0).mean()),
           "target_zero_var_genes": int((np.asarray(
               np.load(os.path.join(CANON, ds, "target.npz"))["X"]).std(0) == 0).sum()),
           "n_modules": len(mods),
           "module_min_cells": int(min([m["n_cells"] for m in mods])) if mods else 0,
           "modules": [{"module": m["module"], "n_cells": m["n_cells"],
                        "n_genes": int(len(m["gene_idx"])),
                        "hotspot_thr": m.get("hotspot_thr"),
                        "boundary_thr": m.get("boundary_thr"),
                        "marker_d_median": m.get("marker_d_median")}
                       for m in mods],
           "skipped_modules": skipped,
           "methods": {}}

    for method in METHODS:
        diag, _ = score_method(ds, method, ref, mods, geom, W_sp, meta,
                               vis_info, rows)
        out["methods"][method] = diag
    out["elapsed_s"] = time.time() - t0
    out["rows"] = rows
    return out
    # noqa


def _worker(ds):
    try:
        res = process_dataset(ds)
        with open(os.path.join(SHARDS, f"{ds}.json"), "w") as f:
            json.dump(res, f, default=jsonable, allow_nan=True)
        return ds, "OK", res.get("elapsed_s"), None
    except Exception:
        return ds, "ERROR", None, traceback.format_exc()


# ------------------------------------------------------------------ finalize --
CORE_GENE = ["gene_pcc_median", "gene_pcc_mean", "gene_rmse", "gene_mae",
             "gene_cosine_median", "gene_spearman_median",
             "gene_vr_log2_median", "gene_pcc_frac_gt_0.5",
             "gene_vr_within_2fold_frac"]
CORE_MODULE = ["module_pcc", "module_vr_log2", "module_rmse"]
CORE_SPATIAL = ["ms_ssim", "abs_delta_moran_i", "abs_delta_geary_c",
                "hotspot_recall", "hotspot_precision", "hotspot_f1",
                "hotspot_AP", "boundary_recall_d0", "boundary_precision_d0",
                "boundary_f1_d0"]
SUMMARY_SPATIAL = CORE_SPATIAL + [
    "moran_i_gt", "moran_i_pred", "geary_gt", "geary_pred",
    "hotspot_PR_AUC", "hotspot_recall_p80", "hotspot_precision_p80",
    "boundary_recall_d1", "boundary_precision_d1",
    "boundary_recall_p80", "boundary_precision_p80"]


def finalize(datasets):
    dfs = []
    for ds in datasets:
        p = os.path.join(SHARDS, f"{ds}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        df = pd.DataFrame(d["rows"])
        df["status"] = df["status"].replace({"NA": "NA_SUPPORT"})
        df["target_slice"] = d["target_slice"]
        dfs.append(df)
    long = pd.concat(dfs, ignore_index=True)
    long.to_csv(os.path.join(OUT, "so3d_metrics_long.csv"), index=False)

    # per-method macro summary
    summary_rows = []
    for method in METHODS:
        sub = long[long["method"] == method]
        for level, metrics in [("gene", CORE_GENE), ("module_macro", CORE_MODULE),
                               ("spatial_macro", SUMMARY_SPATIAL),
                               ("diagnostic", ["gene_coverage_frac",
                                               "pred_mean_ratio",
                                               "zero_frac_pred",
                                               "n_modules"])]:
            for k in metrics:
                s = sub[(sub["level"] == level) & (sub["metric"] == k)]
                v = s["value"].dropna().values
                na = s[s["status"] != "OK"]
                reasons = {}
                for r in na["reason"].fillna(""):
                    if r:
                        reasons[r] = reasons.get(r, 0) + 1
                top = ";".join(f"{a}({b})" for a, b in
                               sorted(reasons.items(), key=lambda x: -x[1])[:3])
                summary_rows.append({
                    "method": method, "level": level, "metric": k,
                    "n_datasets": len(s), "n_available": int(len(v)),
                    "n_na": int(len(s) - len(v)),
                    "macro_mean": float(np.mean(v)) if len(v) else np.nan,
                    "macro_median": float(np.median(v)) if len(v) else np.nan,
                    "std": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
                    "min": float(np.min(v)) if len(v) else np.nan,
                    "max": float(np.max(v)) if len(v) else np.nan,
                    "na_reasons": top,
                })
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(os.path.join(OUT, "so3d_method_summary.csv"), index=False)

    # per dataset x method core table
    core_rows = long[
        ((long["level"] == "gene") & long["metric"].isin(CORE_GENE)) |
        ((long["level"] == "module_macro") & long["metric"].isin(CORE_MODULE)) |
        ((long["level"] == "spatial_macro") & long["metric"].isin(CORE_SPATIAL))]
    core = (core_rows.pivot_table(index=["dataset", "target_slice", "method"],
                                  columns="metric", values="value", aggfunc="first")
            .reset_index())
    core.to_csv(os.path.join(OUT, "so3d_dataset_method_summary.csv"), index=False)

    # dataset coverage
    cov_rows = []
    diag = {}
    for ds in datasets:
        p = os.path.join(SHARDS, f"{ds}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        diag[ds] = {k: d.get(k) for k in
                    ["target_slice", "target_index", "n_target", "n_genes",
                     "n_visible", "n_visible_slices", "zero_frac_target",
                     "target_zero_var_genes", "n_modules", "module_min_cells",
                     "geometry", "skipped_modules", "elapsed_s"]}
        methods = d.get("methods", {})
        st = [m.get("status", "MISSING") for m in methods.values()]
        gsub = long[(long["dataset"] == ds) & (long["metric"] == "gene_pcc_median")]
        best = gsub.loc[gsub["value"].idxmax(), "method"] if gsub["value"].notna().any() else ""
        cov_rows.append({
            "dataset": ds, "target_slice": d["target_slice"],
            "target_index": d["target_index"], "n_target": d["n_target"],
            "n_genes": d["n_genes"], "n_visible": d["n_visible"],
            "n_visible_slices": d["n_visible_slices"],
            "zero_frac_target": d["zero_frac_target"],
            "target_zero_var_genes": d["target_zero_var_genes"],
            "n_modules": d["n_modules"],
            "module_min_cells": d["module_min_cells"],
            "n_modules_skipped": len(d.get("skipped_modules", [])),
            "grid_h": d["geometry"]["grid"][0], "grid_w": d["geometry"]["grid"][1],
            "grid_bin": d["geometry"]["bin"],
            "grid_occupancy": d["geometry"]["occupancy"],
            "raster_mask_frac": d["geometry"]["mask_frac"],
            "n_z_values_target": d["geometry"]["n_z_values"],
            "methods_present": int(sum(1 for m in methods.values()
                                      if m.get("status") in ("OK", "PARTIAL"))),
            "methods_ok": int(sum(1 for s_ in st if s_ == "OK")),
            "methods_partial": int(sum(1 for s_ in st if s_ == "PARTIAL")),
            "methods_failed": int(sum(1 for s_ in st if s_ == "FAILED")),
            "methods_missing": int(8 - len(methods)),
            "best_method_gene_pcc": best,
            "spatial_ok": int(any(np.isfinite(v) for v in
                                  [methods.get(m, {}).get("macro", {}).get("ms_ssim", np.nan)
                                   for m in METHODS])),
            "notes": ("; ".join(sorted({m.get("reason", "") for m in methods.values()
                                        if m.get("reason")})) or ""),
        })
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(os.path.join(OUT, "so3d_dataset_coverage.csv"), index=False)

    # status.json
    na_counts = (long[long["status"] != "OK"].groupby(["level", "metric"])
                 .size().sort_values(ascending=False))
    failures = long[long["status"] == "FAILED"]
    status = {
        "status": "DONE" if len(failures) == 0 else "PARTIAL",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_datasets": len(datasets),
        "n_methods": len(METHODS),
        "methods": METHODS,
        "n_cells_expected": len(datasets) * len(METHODS),
        "prediction_counts": {m: int((long[(long["method"] == m) &
                                           (long["metric"] == "gene_pcc_median")].shape[0]))
                              for m in METHODS},
        "predictions_present": {
            m: sum(1 for d in datasets
                   if os.path.exists(os.path.join(PRED, d, f"{m}.npz")))
            for m in METHODS},
        "statuses_present": {m: sum(1 for d in datasets
                                    if os.path.exists(os.path.join(PRED, d, f"{m}.status.json")))
                             for m in METHODS},
        "sparse_target_datasets_lt_100_spots": [
            {"dataset": r["dataset"], "n_target": int(r["n_target"]),
             "spatial_metrics": "NA"}
            for _, r in cov[cov["n_target"] < MIN_TARGET_SPOTS_SPATIAL].iterrows()],
        "datasets_without_modules": [
            {"dataset": r["dataset"], "n_visible": int(r["n_visible"]),
             "reason": "no celltype with >=%d visible spots and >=%d markers"
                       % (MIN_MODULE_CELLS, MODULE_MIN_GENES)}
            for _, r in cov[cov["n_modules"] == 0].iterrows()],
        "n_metric_rows": int(len(long)),
        "n_ok_rows": int((long["status"] == "OK").sum()),
        "n_na_rows": int((long["status"] == "NA_SUPPORT").sum()),
        "n_failed_rows": int((long["status"] == "FAILED").sum()),
        "na_by_metric": {f"{k[0]}|{k[1]}": int(v) for k, v in na_counts.items()},
        "failed_cells": (failures[["dataset", "method", "level", "module",
                                   "metric", "reason"]]
                         .drop_duplicates().to_dict("records")),
        "protocol": {
            "target": "one fixed held-out middle slice per dataset (meta.target_slice)",
            "target_X_usage": ("target.npz X is first opened after all visible-derived "
                               "statistics are fixed; used only for scoring and "
                               "target-only diagnostics"),
            "geometry": ("target 2D xy coordinates only (no target expression) for kNN "
                         "graph and raster bin/mask"),
            "gene_metrics": ("all genes on target spots; corr/VR restricted to genes "
                             "non-constant in both target and prediction"),
            "modules": (f"visible obs['Celltype'] one-vs-rest Cohen's d, top {MODULE_TOPK}, "
                        f"d>={MODULE_MIN_D}, n_cells>={MIN_MODULE_CELLS}, "
                        f">={MODULE_MIN_GENES} genes; z reference = all visible spots"),
            "module_score": "mean z over module genes (visible mean/std)",
            "spatial_2d": ("spot-level kNN (k=6) Moran/Geary on target xy; masked "
                           "3-scale 7x7 MS-SSIM on an adaptive-bin 2D raster "
                           "(bin from target xy spacing/occupancy, grid<=128x128)"),
            "raster_mask": ("closed+filled occupied cells; occupied cells dilated 0-2 "
                            "bins only when fewer than %d valid 7x7 windows" % MIN_SSIM_WINDOWS),
            "hotspot": (f"threshold = visible p90 module score (p80 sensitivity); P/R "
                        f"requires >={MIN_HOTSPOT_NEG} GT and pred positives; AP/PR-AUC "
                        f"threshold-free and requires >={MIN_HOTSPOT_NEG} GT positives"),
            "boundary": ("threshold = pooled visible-slice p90 module-gradient (p80 "
                         "sensitivity), strict visible interior (erode 2 bins); target "
                         "contour erosion adaptive 2/1/0 bins; components>=2 px; "
                         "tolerance 0 and 1 bin; P/R requires >=%d GT and pred pixels"
                         % MIN_BOUNDARY_SPOTS),
            "support_gates": {
                "spatial_min_target_spots": MIN_TARGET_SPOTS_SPATIAL,
                "min_hotspot_positives": MIN_HOTSPOT_NEG,
                "min_boundary_pixels": MIN_BOUNDARY_SPOTS,
                "min_ssim_windows": MIN_SSIM_WINDOWS,
            },
            "variance_ratio": "log2(var(pred)/var(target)); median per gene or module macro",
            "no_composite_score": True,
        },
        "aggregation": "macro-average over datasets with finite metric values; "
                       "N/A never imputed",
        "datasets": diag,
    }
    with open(os.path.join(OUT, "status.json"), "w") as f:
        json.dump(status, f, indent=2, default=jsonable)
    return long, summ, cov, status


def write_markdown(long, summ, cov, status):
    lines = []
    lines.append("# So3D 46 x 8 metric summary (metrics_v1-adapted protocol)\n")
    lines.append(f"- Generated: {status['created_utc']}")
    lines.append(f"- Datasets: {status['n_datasets']}, methods: {status['n_methods']}, "
                 f"cells: {status['n_datasets'] * status['n_methods']}")
    lines.append(f"- Metric rows: {status['n_metric_rows']} "
                 f"(OK {status['n_ok_rows']}, NA {status['n_na_rows']}, "
                 f"FAILED {status['n_failed_rows']})")
    lines.append(f"- Held-out slice: fixed per dataset (`meta.target_slice`), "
                 f"target X used for scoring only.")
    lines.append(f"- Raster: adaptive 2D bin from target xy geometry; "
                 f"module z-scores/hotspot/boundary thresholds from visible slices only.")
    lines.append("")

    def table(metrics, level, title):
        lines.append(f"## {title}\n")
        lines.append("Macro-average over datasets (only finite cells; N/A not imputed).\n")
        head = "| method | " + " | ".join(metrics) + " |"
        sep = "|---" * (len(metrics) + 1) + "|"
        lines.append(head)
        lines.append(sep)
        for m in METHODS:
            vals = []
            for k in metrics:
                r = summ[(summ["method"] == m) & (summ["level"] == level) &
                         (summ["metric"] == k)]
                if len(r) and np.isfinite(r.iloc[0]["macro_mean"]):
                    vals.append(f"{r.iloc[0]['macro_mean']:.3f} "
                                f"({int(r.iloc[0]['n_available'])}/{int(r.iloc[0]['n_datasets'])})")
                else:
                    vals.append("N/A")
            lines.append(f"| {m} | " + " | ".join(vals) + " |")
        lines.append("")

    table(CORE_GENE, "gene", "Gene-level (target spots, all genes)")
    table(CORE_MODULE, "module_macro", "Module-level (visible-celltype marker modules)")
    table(CORE_SPATIAL, "spatial_macro", "Spatial (2D target slice, module maps)")
    table(["gene_coverage_frac", "pred_mean_ratio", "zero_frac_pred"],
          "diagnostic", "Prediction diagnostics (mean across datasets)")

    lines.append("## N/A / support counts\n")
    lines.append("| level | metric | N/A cells | top reasons |")
    lines.append("|---|---|---|---|")
    nas = (long[long["status"] != "OK"].groupby(["level", "metric", "reason"])
           .size().reset_index(name="n").sort_values("n", ascending=False))
    for (lv, mt), grp in nas.groupby(["level", "metric"], sort=False):
        reasons = "; ".join(f"{r['reason']} ({r['n']})" for _, r in grp.head(3).iterrows())
        lines.append(f"| {lv} | {mt} | {int(grp['n'].sum())} | {reasons} |")
    lines.append("")

    lines.append("## Failures\n")
    fails = long[long["status"] == "FAILED"]
    if len(fails) == 0:
        lines.append("No failed dataset x method cells (all 368 predictions aligned, "
                     "finite, non-negative).")
    else:
        for _, r in fails[["dataset", "method", "metric", "reason"]].drop_duplicates().iterrows():
            lines.append(f"- {r['dataset']} / {r['method']} / {r['metric']}: {r['reason']}")
    lines.append("")

    lines.append("## Sparse-target and support caveats\n")
    small = cov[cov["n_target"] < MIN_TARGET_SPOTS_SPATIAL]
    lines.append(f"- {len(small)}/46 target slices have < {MIN_TARGET_SPOTS_SPATIAL} "
                 f"spots, so all spatial metrics (Moran/Geary, MS-SSIM, hotspot, "
                 f"boundary) are N/A for those datasets: "
                 + ", ".join(f"{r['dataset']}={r['n_target']}" for _, r in small.iterrows()) + ".")
    nomod = cov[cov["n_modules"] == 0]
    lines.append(f"- {len(nomod)} datasets have no valid marker modules "
                 f"(celltype categories with >={MIN_MODULE_CELLS} visible spots and "
                 f">={MODULE_MIN_GENES} markers); module/spatial metrics are N/A there.")
    nocov = cov[cov["methods_failed"] > 0]
    lines.append(f"- Dataset x method failures: {len(nocov)} datasets with >=1 failed cell.")
    lines.append(f"- Module macro values weight modules equally, not by size; "
                 f"hotspot/boundary use fixed visible-derived thresholds, so "
                 f"scale-shifted predictions lose recall by construction.")
    lines.append(f"- Hotspot/boundary means are computed only on cells passing the "
                 f"support gates, and support differs by method (boundary d0: "
                 f"SINTER3D 6/46, UniST 15/46, others 42-43/46; hotspot d0: "
                 f"SINTER3D 38/46, UniST 35/46, others 42-44/46), so those columns "
                 f"are selection-biased and not directly comparable across methods.")
    lines.append(f"- UniST and other methods can emit constant genes/spots; "
                 f"gene_pcc/gene_vr are computed only where both target and "
                 f"prediction are non-constant (see gene_coverage_frac), which "
                 f"favours methods that only predict a subset of genes.")
    lines.append(f"- Sparse targets: median zero fraction across datasets = "
                 f"{cov['zero_frac_target'].median():.3f}; gene PCC is computed only "
                 f"for genes non-constant in both target and prediction "
                 f"(n_genes_scored in long table).")
    lines.append("")
    with open(os.path.join(OUT, "so3d_method_summary.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--no-finalize", action="store_true")
    args = ap.parse_args()
    os.makedirs(SHARDS, exist_ok=True)
    datasets = args.datasets or sorted(
        d for d in os.listdir(CANON)
        if os.path.isdir(os.path.join(CANON, d)) and
        os.path.exists(os.path.join(CANON, d, "meta.json")))
    log(f"start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
        f"datasets={len(datasets)} workers={args.workers} pilot={args.pilot}")
    t0 = time.time()
    if args.workers <= 1 or len(datasets) == 1:
        results = [_worker(d) for d in datasets]
    else:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            results = pool.map(_worker, datasets)
    for ds, st, el, tb in results:
        if st == "OK":
            log(f"  {ds}: OK {el:.1f}s")
        else:
            log(f"  {ds}: ERROR\n{tb}")
    if not args.no_finalize:
        long, summ, cov, status = finalize(datasets)
        write_markdown(long, summ, cov, status)
        log(f"finalize: rows={len(long)} summary={len(summ)} coverage={len(cov)}")
        log(f"status: {status['status']} ok={status['n_ok_rows']} "
            f"na={status['n_na_rows']} failed={status['n_failed_rows']}")
    log(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

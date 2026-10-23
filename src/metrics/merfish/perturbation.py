#!/usr/bin/env python
"""Phase 5: perturbation validation of the metric protocol on held-out GT module maps.

Only GT module maps are perturbed; no prediction and no retraining is involved.
For each perturbation the perturbed map is scored against the unperturbed GT map
with the same 10-metric machinery.

Writes:
  perturbation_curves.csv   (condition x level x metric: mean/median/sd over samples)
  perturbation_samples.csv  (per combo x slice x module x condition sample record)
  perturbation_summary.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlib as M

NOISE_SIGMA_BINS = 1.5
NOISE_REALISATIONS = 3


def add_bump(A, mask, center, radius, amplitude):
    H, W = A.shape
    yy, xx = np.mgrid[0:H, 0:W]
    sig = max(radius / 2.0, 0.5)
    g = np.exp(-(((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * sig * sig)))
    out = A.copy()
    out[mask] = out[mask] + amplitude * g[mask]
    return out


def count_components(m):
    lab, n = ndi.label(m, structure=ndi.generate_binary_structure(2, 1))
    return int(n)


def perturb_maps(A_gt, mask, thr, name, seed=0):
    """Yield (condition, level, perturbed map, meta)."""
    filled = M.fill_nearest(A_gt, mask)
    sd = float(np.std(A_gt[mask]))
    inner = M.erode_mask(mask, 2)
    gy, gx = np.mgrid[0:A_gt.shape[0], 0:A_gt.shape[1]]
    yield "identity", 0.0, A_gt.copy(), {}

    for sigma in (1, 2, 3):
        B = ndi.gaussian_filter(filled, sigma=sigma, mode="nearest")
        yield "blur", float(sigma), B, {}

    # add a hotspot of radius 2/3 bins at the lowest-scoring interior bin
    smooth_low = ndi.gaussian_filter(filled, sigma=2.0, mode="nearest")
    score = np.where(inner & np.isfinite(smooth_low), smooth_low, np.inf)
    cy, cx = np.unravel_index(int(np.argmin(score)), score.shape)
    for radius in (2, 3):
        amp = max(thr - float(A_gt[cy, cx]) + sd, 0.5 * sd)
        B = add_bump(filled, mask, (cy, cx), radius, amp)
        yield "add_hotspot", float(radius), B, {
            "center_row": int(cy), "center_col": int(cx), "amplitude": float(amp),
            "components_gt": count_components(M.hotspot_mask(A_gt, mask, thr)),
            "components_pert": count_components(
                M.hotspot_mask(B, mask, thr))}

    # add a spurious vertical boundary band (a high stripe -> two gradient ridges)
    cols = np.where(inner.any(0))[0]
    c = int(np.median(cols))
    B = filled.copy()
    band = np.zeros_like(mask)
    band[:, max(c - 1, 0):c + 1] = True
    band &= mask
    B[band] = B[band] + 3.0 * sd
    yield "add_boundary_band", 2.0, B, {"center_col": c, "width_bins": 2,
                                        "amplitude_sd": 3.0}

    for sh in (1, 2):
        B = ndi.shift(filled, shift=(0, sh), order=1, mode="nearest")
        yield "shift", float(sh), B, {}

    rng = np.random.default_rng(seed)
    for factor in (0.5, 1.0, 2.0):
        acc = np.zeros_like(A_gt, dtype=np.float64)
        for _ in range(NOISE_REALISATIONS):
            w = rng.standard_normal(A_gt.shape)
            sm = ndi.gaussian_filter(w, sigma=NOISE_SIGMA_BINS, mode="nearest")
            sm = sm / max(float(np.std(sm[mask])), 1e-12)
            acc += sm
        acc /= NOISE_REALISATIONS
        B = filled + factor * sd * acc
        yield "correlated_noise", float(factor), B, {
            "noise_sigma_bins": NOISE_SIGMA_BINS, "realisations": NOISE_REALISATIONS}

    st = ndi.generate_binary_structure(2, 1)
    high = M.hotspot_mask(A_gt, mask, thr)
    closed = ndi.binary_closing(high, structure=st, iterations=2)
    newpix = closed & ~high & mask
    B = A_gt.copy()
    B[newpix] = np.maximum(B[newpix], thr + 0.5 * sd)
    yield "merge_components", 2.0, B, {
        "components_gt": count_components(high),
        "components_pert": count_components(M.hotspot_mask(B, mask, thr)),
        "n_pixels_changed": int(newpix.sum())}

    # split: cut each sizeable hotspot component along its median column
    lab, nlab = ndi.label(high, structure=st)
    B = A_gt.copy()
    n_cut = 0
    for k in range(1, nlab + 1):
        ys, xs = np.where(lab == k)
        if len(ys) < 8 or (xs.max() - xs.min()) < 3:
            continue
        c = int(np.median(xs))
        sel = (lab == k) & (np.arange(B.shape[1])[None, :] == c)
        if sel.sum() == 0 or sel.sum() == len(ys):
            continue
        B[sel] = np.minimum(B[sel], thr - 0.5 * sd)
        n_cut += 1
    yield "split_components", 2.0, B, {
        "components_gt": count_components(high),
        "components_pert": count_components(M.hotspot_mask(B, mask, thr)),
        "n_pixels_changed": int((M.hotspot_mask(B, mask, thr) != high).sum()),
        "n_components_cut": int(n_cut)}


def score_map(A_p, A_gt, mask, W, thr, gthr):
    rec = {}
    gt_vals = A_gt[mask]
    p_vals = A_p[mask]
    rec["module_pcc"] = M.pcc_vec(p_vals, gt_vals)
    L = max(float(np.percentile(gt_vals, 99.5)), 1e-6)
    rec["ms_ssim"] = M.ms_ssim(A_p, A_gt, mask, L=L)
    rec["var_ratio_log2"] = M.variance_ratio_log2(p_vals, gt_vals)
    mi_g, _ = M.morans_i_W(W, gt_vals)
    mi_p, _ = M.morans_i_W(W, p_vals)
    rec["abs_delta_moran_i"] = abs(mi_p - mi_g) if np.isfinite(mi_g) and np.isfinite(mi_p) else np.nan
    gt_h = M.hotspot_mask(A_gt, mask, thr)
    p_h = M.hotspot_mask(A_p, mask, thr)
    tp = int(np.logical_and(gt_h, p_h).sum())
    fp = int(np.logical_and(~gt_h, p_h).sum())
    fn = int(np.logical_and(gt_h, ~p_h).sum())
    rec["hotspot_recall"], rec["hotspot_precision"], rec["hotspot_f1"] = M.prf(tp, fp, fn)
    if gt_h.sum() > 0 and np.nanstd(p_vals) > 0:
        rec["hotspot_AP"], rec["hotspot_PR_AUC"], _ = M.ap_pr_auc(p_vals, gt_h[mask])
    else:
        rec["hotspot_AP"] = np.nan
        rec["hotspot_PR_AUC"] = np.nan
    b_gt, _, inner = M.boundary_mask(A_gt, mask, gthr)
    b_p, _, _ = M.boundary_mask(A_p, mask, gthr)
    bm0 = M.boundary_match(b_gt, b_p, 0)
    rec["boundary_recall_d0"] = bm0["recall"]
    rec["boundary_precision_d0"] = bm0["precision"]
    rec["boundary_f1_d0"] = bm0["f1"]
    bm1 = M.boundary_match(b_gt, b_p, 1)
    rec["boundary_recall_d1"] = bm1["recall"]
    rec["boundary_precision_d1"] = bm1["precision"]
    rec["boundary_f1_d1"] = bm1["f1"]
    rec["n_gt_boundary"] = int(b_gt.sum())
    rec["n_pred_boundary"] = int(b_p.sum())
    return rec


def main():
    t0 = time.time()
    with open(os.path.join(HERE, "metric_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    modules = M.load_module_defs()["modules"]
    import anndata as ad
    obs = ad.read_h5ad(M.OBS, backed="r")
    ref = M.reference_stats(obs)
    combos = M.load_combos()
    samples = []
    meta_rows = []

    for combo in combos:
        tg = M.load_target(combo)
        Xg = np.asarray(tg["X"], dtype=np.float64)
        xy_all = np.asarray(tg["coords"], dtype=np.float64)[:, :2]
        slice_all = tg["slice"].astype(str)
        for s in sorted(set(slice_all)):
            sel = slice_all == s
            xs = xy_all[sel]
            W = M.knn_weights(xs)
            Xgs = Xg[sel]
            for m in modules:
                name = m["name"]
                idx = np.asarray(m["gene_idx"], dtype=np.int64)
                sc, nvalid = M.module_scores(Xgs, idx, ref["mean"], ref["std"],
                                             ref["valid"])
                if nvalid == 0:
                    continue
                A_gt, mask, _ = M.rasterize(xs, sc)
                if float(np.nanstd(A_gt[mask])) == 0:
                    continue
                thr = cfg["modules"]["hotspot_thresholds_reference"][name][
                    "hotspot_threshold_p90"]
                gthr = cfg["modules"]["boundary_thresholds_reference"][name][
                    "gradient_threshold_p90"]
                for cond, level, A_p, meta in perturb_maps(
                        A_gt, mask, thr, name,
                        seed=hash((combo, s, name)) % (2 ** 31)):
                    rec = score_map(A_p, A_gt, mask, W, thr, gthr)
                    rec.update({"combo": combo, "slice": s, "module": name,
                                "condition": cond, "level": float(level),
                                "n_bins": int(mask.sum()),
                                "n_valid_genes": int(nvalid)})
                    samples.append(rec)
                    meta_rows.append({"combo": combo, "slice": s, "module": name,
                                      "condition": cond, "level": float(level),
                                      **meta})

    df = pd.DataFrame(samples)
    df.to_csv(os.path.join(HERE, "perturbation_samples.csv"), index=False)
    metrics = ["module_pcc", "ms_ssim", "var_ratio_log2", "abs_delta_moran_i",
               "hotspot_recall", "hotspot_precision", "hotspot_f1",
               "hotspot_AP", "boundary_recall_d0", "boundary_precision_d0",
               "boundary_f1_d0", "boundary_recall_d1", "boundary_precision_d1",
               "boundary_f1_d1", "n_gt_boundary", "n_pred_boundary"]
    rows = []
    for (cond, level), g in df.groupby(["condition", "level"], sort=False):
        for metric in metrics:
            v = g[metric].astype(float).dropna()
            rows.append({"condition": cond, "level": float(level),
                         "combo": "ALL", "metric": metric,
                         "mean": float(v.mean()) if len(v) else np.nan,
                         "median": float(v.median()) if len(v) else np.nan,
                         "sd": float(v.std()) if len(v) > 1 else np.nan,
                         "n": int(len(v))})
        for combo, gc in g.groupby("combo"):
            for metric in metrics:
                v = gc[metric].astype(float).dropna()
                rows.append({"condition": cond, "level": float(level),
                             "combo": combo, "metric": metric,
                             "mean": float(v.mean()) if len(v) else np.nan,
                             "median": float(v.median()) if len(v) else np.nan,
                             "sd": float(v.std()) if len(v) > 1 else np.nan,
                             "n": int(len(v))})
    curves = pd.DataFrame(rows)
    curves.to_csv(os.path.join(HERE, "perturbation_curves.csv"), index=False)

    meta = pd.DataFrame(meta_rows)
    comp_cols = [c for c in ("components_gt", "components_pert", "n_pixels_changed")
                 if c in meta.columns]
    comp_summary = {}
    for cond in ("add_hotspot", "merge_components", "split_components"):
        sub = meta[meta["condition"] == cond]
        if len(sub) and comp_cols:
            comp_summary[cond] = {
                c: {"mean": float(sub[c].dropna().mean()),
                    "median": float(sub[c].dropna().median())}
                for c in comp_cols if c in sub and sub[c].notna().any()}
    identity = curves[(curves["condition"] == "identity") &
                      (curves["combo"] == "ALL")].set_index("metric")
    summ = {
        "n_samples": int(len(df)),
        "n_maps": int(len(df[df["condition"] == "identity"])),
        "conditions": sorted(df["condition"].unique().tolist()),
        "levels": {c: sorted(df[df["condition"] == c]["level"].unique().tolist())
                   for c in df["condition"].unique()},
        "identity_sanity": {m: float(identity.loc[m, "mean"])
                            for m in metrics if m in identity.index},
        "component_changes": comp_summary,
        "noise": {"sigma_bins": NOISE_SIGMA_BINS,
                  "realisations": NOISE_REALISATIONS},
        "seconds": float(time.time() - t0),
    }
    import torch
    if torch.cuda.is_available():
        summ["gpu"] = {"device": torch.cuda.get_device_name(0),
                       "max_memory_allocated_mb": float(
                           torch.cuda.max_memory_allocated() / 2 ** 20)}
    with open(os.path.join(HERE, "perturbation_summary.json"), "w") as f:
        json.dump(summ, f, indent=2, default=M.jsonable)
    print(json.dumps(summ, indent=2, default=M.jsonable))
    print("elapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()

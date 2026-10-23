#!/usr/bin/env python
"""Shared definitions for the held-out missing-slice experiment (50 um MERFISH).

Protocol
--------
* For every combo, models are trained ONLY on observed bins of the visible
  slices.  Target (missing) slices contribute only their bin coordinates; their
  expression is read only by ``compute_metrics.py`` after predictions are on
  disk.
* Predictions are written as ``npz`` (coords + X + gene names) so metrics can
  be recomputed without retraining.
"""
import json
import os

import numpy as np

ROOT = os.environ.get("MERFISH5_ROOT", os.getcwd())
OBS = os.environ.get("MERFISH5_OBSERVED", os.path.join(ROOT, "observed_50um.h5ad"))
GRID = os.environ.get("MERFISH5_GRID", os.path.join(ROOT, "grid_50um.npz"))
N_GENES = 1122

SLICE_IDS = {
    "S1": "C57BL6J-3.006",
    "S2": "C57BL6J-3.007",
    "S3": "C57BL6J-3.008",
    "S4": "C57BL6J-3.009",
    "S5": "C57BL6J-3.010",
}

COMBOS = {
    "M1a": {"missing": ["S2"], "visible": ["S1", "S3", "S4", "S5"]},
    "M1b": {"missing": ["S3"], "visible": ["S1", "S2", "S4", "S5"]},
    "M1c": {"missing": ["S4"], "visible": ["S1", "S2", "S3", "S5"]},
    "M2a": {"missing": ["S2", "S3"], "visible": ["S1", "S4", "S5"]},
    "M2b": {"missing": ["S2", "S4"], "visible": ["S1", "S3", "S5"]},
    "M2c": {"missing": ["S3", "S4"], "visible": ["S1", "S2", "S5"]},
    "M3a": {"missing": ["S2", "S3", "S4"], "visible": ["S1", "S5"]},
}

METHODS = ["NTF", "Spateo"]


def combo_dir(combo):
    return os.path.join(ROOT, combo)


def visible_names(combo):
    return [SLICE_IDS[s] for s in COMBOS[combo]["visible"]]


def missing_names(combo):
    return [SLICE_IDS[s] for s in COMBOS[combo]["missing"]]


def write_combos_json(adata=None):
    """Write combos.json with slice lists and bin counts (counts need the h5ad)."""
    out = {}
    for combo, spec in COMBOS.items():
        d = {
            "missing": spec["missing"],
            "visible": spec["visible"],
            "missing_slice_ids": missing_names(combo),
            "visible_slice_ids": visible_names(combo),
        }
        if adata is not None:
            counts = adata.obs["slice"].value_counts()
            d["n_target"] = int(sum(counts[s] for s in d["missing_slice_ids"]))
            d["n_visible"] = int(sum(counts[s] for s in d["visible_slice_ids"]))
            d["n_target_per_slice"] = {
                s: int(counts[s]) for s in d["missing_slice_ids"]}
            d["n_visible_per_slice"] = {
                s: int(counts[s]) for s in d["visible_slice_ids"]}
        out[combo] = d
    payload = {
        "dataset": OBS,
        "voxel_um": 50,
        "n_total_observed": int(adata.n_obs) if adata is not None else None,
        "n_genes": N_GENES,
        "slice_order": ["S1", "S2", "S3", "S4", "S5"],
        "slice_ids": SLICE_IDS,
        "protocol": (
            "train only on observed bins of `visible` slices; target coordinates are the "
            "observed 50um bins of `missing` slices; target expression is used for scoring only"
        ),
        "combos": out,
    }
    path = os.path.join(ROOT, "combos.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def pcc_rows(a, b):
    """Per-row Pearson r, NaN where a row is constant in either matrix."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    sa = np.sqrt((a * a).sum(1))
    sb = np.sqrt((b * b).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b).sum(1) / (sa * sb)
    r[(sa == 0) | (sb == 0)] = np.nan
    return r


def pcc_cols(a, b):
    return pcc_rows(a.T, b.T)


def core_metrics(pred, truth):
    """Core metrics for a (n_spots, n_genes) prediction against ground truth."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    assert pred.shape == truth.shape, (pred.shape, truth.shape)
    a = pred.ravel() - pred.mean()
    b = truth.ravel() - truth.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    pooled = float(a.dot(b) / denom) if denom > 0 else float("nan")

    rg = pcc_cols(pred, truth)
    rs = pcc_rows(pred, truth)
    err = pred - truth
    abs_err = np.abs(err)
    row_std = pred.std(1)

    def _f(x):
        return None if x is None or not np.isfinite(x) else float(x)

    return {
        "pcc_pooled": _f(pooled),
        "pcc_gene_mean": _f(np.nanmean(rg)),
        "pcc_gene_median": _f(np.nanmedian(rg)),
        "pcc_gene_p10": _f(np.nanpercentile(rg, 10)),
        "n_genes_scored": int(np.isfinite(rg).sum()),
        "pcc_spot_mean": _f(np.nanmean(rs)),
        "pcc_spot_median": _f(np.nanmedian(rs)),
        "n_spots_scored": int(np.isfinite(rs).sum()),
        "rmse": _f(np.sqrt((err ** 2).mean())),
        "mae": _f(abs_err.mean()),
        "mse": _f((err ** 2).mean()),
        "n_spots": int(pred.shape[0]),
        "n_genes": int(pred.shape[1]),
        "pred_mean": _f(pred.mean()),
        "gt_mean": _f(truth.mean()),
        "pred_min": _f(pred.min()),
        "pred_max": _f(pred.max()),
        "finite_frac": _f(np.isfinite(pred).mean()),
        "n_constant_rows": int((row_std == 0).sum()),
        "row_std_mean": _f(row_std.mean()),
    }


def save_prediction(combo, method, coords, X, obs_names, slice_names, extra=None):
    d = os.path.join(ROOT, "predictions", combo)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{method}.npz")
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32),
        coords=np.asarray(coords, dtype=np.float32),
        obs_names=np.asarray(obs_names, dtype=str),
        slice=np.asarray(slice_names, dtype=str),
        method=np.asarray(method),
        combo=np.asarray(combo),
    )
    if extra:
        with open(os.path.join(d, f"{method}.json"), "w") as f:
            json.dump(extra, f, indent=2, ensure_ascii=False)
    return path


def save_target(combo, coords, X, obs_names, slice_names):
    d = os.path.join(ROOT, "targets", combo)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "target.npz")
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32),
        coords=np.asarray(coords, dtype=np.float32),
        obs_names=np.asarray(obs_names, dtype=str),
        slice=np.asarray(slice_names, dtype=str),
    )
    return path

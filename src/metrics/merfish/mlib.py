#!/usr/bin/env python
"""Shared library for MERFISH 5-slice metric protocol v1.

Frozen protocol (no target expression in any fitting/calibration step):
  * expression scale: target/prediction npz ``X`` used as-is (already normalised);
  * z-score: per-gene mean/std computed on observed bins of S1/S5 only;
    genes with zero reference std are excluded from module maps (invalid);
  * module map: mean of z over the module's valid genes, per 50 um bin;
  * all thresholds (hotspot, boundary gradient) come from S1/S5 only;
  * sparse 2D per-section rasters only; no dense/3D volume is written.
"""
import hashlib
import json
import os
import re

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # missing_combos_50um
DATA_ROOT = ROOT
OBS = "/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/Workspace/merfish_test/observed_50um.h5ad"
RES_MM = 0.05
REF_SLICES = ["C57BL6J-3.006", "C57BL6J-3.010"]  # S1, S5
COMBOS = ["M1a", "M1b", "M1c", "M2a", "M2b", "M2c", "M3a"]
METHODS = [
    "NTF", "Spateo", "PASTE-OT-Linear", "ReHo", "Ours-EncMTP",
    "SINTER3D", "SpatialZ", "isoST", "DeepSpatial", "UniST",
]
KNN_K = 6
SSIM_WIN = 7
SSIM_SCALES = 3
SSIM_WEIGHTS = np.array([0.0448, 0.2856, 0.3001], dtype=np.float64)
SSIM_WEIGHTS = SSIM_WEIGHTS / SSIM_WEIGHTS.sum()
CONTOUR_BINS = 2
MIN_COMPONENT_BINS = 2


# ------------------------------------------------------------------ loading --
def load_combos():
    with open(os.path.join(ROOT, "combos.json")) as f:
        return json.load(f)["combos"]


def load_module_defs():
    with open(os.path.join(ROOT, "modules", "modules_celltype.json")) as f:
        d = json.load(f)
    return d


def load_genes():
    """Gene order of the shared expression matrix (observed_50um var order)."""
    ref = np.load(os.path.join(ROOT, "modules", "reference_stats.npz"), allow_pickle=True)
    return ref["genes"].astype(str)


def load_target(combo):
    p = os.path.join(ROOT, "targets", combo, "target.npz")
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_pred(combo, method):
    # NTF/Spateo live in the original predictions/; the other 8 were run by
    # the all-method remediation into all_methods/predictions/.
    p = os.path.join(ROOT, "predictions", combo, f"{method}.npz")
    if not os.path.exists(p):
        p = os.path.join(ROOT, "all_methods", "predictions", combo, f"{method}.npz")
    z = np.load(p, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    # Newer target-only runners use `pred`; older NTF/Spateo use `X`.
    if "X" not in d and "pred" in d:
        d["X"] = d["pred"]
    return d


def reference_stats(obs=None):
    """Recompute S1/S5 per-gene mean/std (population std) from the observed h5ad.

    Returns dict(mean, std, valid, genes, n_rows).  ``valid`` is False where the
    reference std is exactly zero; those genes are excluded from module maps.
    """
    import anndata as ad

    if obs is None:
        obs = ad.read_h5ad(OBS, backed="r")
    mask = obs.obs["slice"].astype(str).isin(REF_SLICES).values
    X = np.asarray(obs[mask].X, dtype=np.float64)
    mean = X.mean(0)
    std = X.std(0)  # ddof=0
    genes = obs.var_names.astype(str)
    return {"mean": mean, "std": std, "valid": std > 0, "genes": genes,
            "n_rows": int(mask.sum())}


def load_reference_expression(obs):
    mask = obs.obs["slice"].astype(str).isin(REF_SLICES).values
    X = np.asarray(obs[mask].X, dtype=np.float64)
    meta = obs.obs.loc[mask, ["slice", "x_mm", "y_mm"]].copy()
    return X, meta


# ----------------------------------------------------------------- rastering --
def _grid_index(xy, res=RES_MM):
    x = np.asarray(xy[:, 0], dtype=np.float64)
    y = np.asarray(xy[:, 1], dtype=np.float64)
    x0 = float(x.min())
    y0 = float(y.min())
    ix = np.rint((x - x0) / res).astype(np.int64)
    iy = np.rint((y - y0) / res).astype(np.int64)
    rx = np.abs(x - (x0 + ix * res))
    ry = np.abs(y - (y0 + iy * res))
    return ix, iy, float(rx.max()), float(ry.max()), (x0, y0)


def rasterize(xy, values, res=RES_MM):
    """Rasterise per-bin values onto a 2D section grid.

    Returns A (float64, NaN outside tissue), mask (bool), and diagnostics.
    """
    ix, iy, rx, ry, origin = _grid_index(xy, res)
    if len(np.unique(ix * (iy.max() + 1) + iy)) != len(ix):
        raise ValueError("duplicate 2D bin coordinates in section")
    H = int(iy.max()) + 1
    W = int(ix.max()) + 1
    A = np.full((H, W), np.nan, dtype=np.float64)
    mask = np.zeros((H, W), dtype=bool)
    A[iy, ix] = np.asarray(values, dtype=np.float64)
    mask[iy, ix] = True
    return A, mask, {"max_x_residual": rx, "max_y_residual": ry,
                     "origin_mm": origin, "shape": (H, W), "ix": ix, "iy": iy}


def fill_nearest(A, mask):
    if mask.all():
        return A.copy()
    idx = ndi.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return A[tuple(idx)]


def erode_mask(mask, bins=CONTOUR_BINS):
    """Interior tissue mask, excluding the outer contour by ``bins`` bins."""
    return ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(2, 1),
                              iterations=bins, border_value=0)


# -------------------------------------------------------------------- stats --
def pcc_vec(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    sa = np.sqrt((a * a).sum())
    sb = np.sqrt((b * b).sum())
    if sa == 0 or sb == 0:
        return float("nan")
    return float((a * b).sum() / (sa * sb))


def pcc_cols(a, b):
    """Per-column Pearson r (NaN for constant columns)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    sa = np.sqrt((a * a).sum(0))
    sb = np.sqrt((b * b).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b).sum(0) / (sa * sb)
    r[(sa == 0) | (sb == 0)] = np.nan
    return r


def module_scores(X, gene_idx, mean, std, valid, need_z=False):
    """Module map = mean z over valid genes.  Returns (scores, n_valid_genes)."""
    idx = np.asarray(gene_idx, dtype=np.int64)
    good = idx[valid[idx]]
    if good.size == 0:
        return np.full(X.shape[0], np.nan), 0
    Z = (np.asarray(X[:, good], dtype=np.float64) - mean[good]) / std[good]
    return Z.mean(1), int(good.size)


def zscore_matrix(X, mean, std, valid):
    Z = (np.asarray(X, dtype=np.float64) - mean) / std
    Z[:, ~valid] = 0.0
    return Z


def knn_weights(xy, k=KNN_K):
    """Symmetrised, row-normalised kNN spatial weights for one section."""
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    if n < k + 2:
        return None
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k + 1)
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = idx[:, 1:].ravel().astype(np.int64)
    W = csr_matrix((np.ones(n * k), (rows, cols)), shape=(n, n))
    W = W + W.T
    W = W.multiply(1.0 / np.asarray(W.sum(1)).ravel()[:, None])
    return csr_matrix(W)


def morans_i_W(W, vals):
    v = np.asarray(vals, dtype=np.float64)
    n = len(v)
    if W is None or n < 2 or np.nanstd(v) == 0:
        return float("nan"), float("nan")
    z = v - v.mean()
    denom = float(z @ z)
    sumw = float(W.sum())
    if denom <= 0 or sumw <= 0:
        return float("nan"), float("nan")
    Wz = W @ z
    I = (n / sumw) * float(z @ Wz) / denom
    d = np.asarray(W.sum(1)).ravel()
    num = 2.0 * (float((d * z * z).sum()) - float(z @ Wz))
    C = ((n - 1) / (2.0 * sumw)) * num / denom
    return float(I), float(C)


def morans_i(xy, vals, k=KNN_K):
    """Moran's I and Geary's C with symmetrised, row-normalised kNN weights."""
    return morans_i_W(knn_weights(xy, k), vals)


def variance_ratio_log2(pred_vals, gt_vals):
    vp = float(np.var(pred_vals))
    vg = float(np.var(gt_vals))
    if vg <= 0:
        return float("nan")
    if vp <= 0:
        return float("-inf")
    return float(np.log2(vp / vg))


# --------------------------------------------------------------------- SSIM --
_SSIM_TORCH = {"device": None}


def _torch_device():
    import torch
    if _SSIM_TORCH["device"] is None:
        _SSIM_TORCH["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return _SSIM_TORCH["device"]


def _conv_same(x, w):
    import torch.nn.functional as F
    return F.conv2d(x, w, padding=w.shape[-1] // 2)


def _ssim_map_torch(P, G, mask, L, win=SSIM_WIN):
    import torch
    dev = P.device
    kern = torch.ones(1, 1, win, win, dtype=torch.float32, device=dev) / (win * win)
    mx = _conv_same(P, kern)
    my = _conv_same(G, kern)
    sxx = _conv_same(P * P, kern) - mx * mx
    syy = _conv_same(G * G, kern) - my * my
    sxy = _conv_same(P * G, kern) - mx * my
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    ssim = ((2 * mx * my + C1) * (2 * sxy + C2)) / \
           ((mx * mx + my * my + C1) * (sxx + syy + C2) + 1e-12)
    valid = _conv_same(mask, kern) >= 0.999
    if int(valid.sum()) == 0:
        return None
    return ssim[valid]


def ms_ssim(pred_A, gt_A, mask, L=None, scales=SSIM_SCALES, win=SSIM_WIN):
    """Masked multi-scale 2D SSIM (weighted geometric mean, 3 scales, 7x7)."""
    import torch
    import torch.nn.functional as F

    if pred_A is None or np.nanstd(pred_A[mask]) == 0 or np.nanstd(gt_A[mask]) == 0:
        return float("nan")
    if L is None:
        L = max(float(np.percentile(gt_A[mask], 99.5)), 1e-6)
    if L <= 0:
        return float("nan")
    dev = _torch_device()
    P = torch.from_numpy(pred_A[None, None].astype(np.float32)).to(dev)
    G = torch.from_numpy(gt_A[None, None].astype(np.float32)).to(dev)
    M = torch.from_numpy(mask[None, None].astype(np.float32)).to(dev)
    means = []
    for s in range(scales):
        vals = _ssim_map_torch(P, G, M, float(L), win=win)
        if vals is None:
            break
        means.append(float(vals.mean().item()))
        if s < scales - 1 and P.shape[-1] > 2 and P.shape[-2] > 2:
            P = F.avg_pool2d(P, 2, ceil_mode=True)
            G = F.avg_pool2d(G, 2, ceil_mode=True)
            M = F.avg_pool2d(M, 2, ceil_mode=True)
            M = (M >= 0.999).float()
    if not means:
        return float("nan")
    w = SSIM_WEIGHTS[:len(means)] / SSIM_WEIGHTS[:len(means)].sum()
    logs = float(np.sum(w * np.log(np.maximum(np.asarray(means), 1e-6))))
    return float(np.exp(logs))


# ------------------------------------------------------------ binary metrics --
def hotspot_mask(A, mask, thr):
    m = mask & np.isfinite(A) & (A > thr)
    return m


def threshold_set(A, mask, qs=(80, 90, 95)):
    v = A[mask]
    return {q: float(np.percentile(v, q)) for q in qs}


def _pr_curves(scores, labels):
    order = np.argsort(-scores, kind="mergesort")
    lab = labels[order].astype(np.float64)
    tp = np.cumsum(lab)
    fp = np.cumsum(1 - lab)
    P = tp / np.maximum(tp + fp, 1)
    R = tp / max(lab.sum(), 1)
    return P, R


def ap_pr_auc(scores, labels):
    """Average precision and trapezoidal PR-AUC (binary labels)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    npos = int(labels.sum())
    if npos == 0:
        return float("nan"), float("nan"), float("nan")
    if npos == labels.size:
        return 1.0, 1.0, 1.0
    P, R = _pr_curves(scores, labels)
    # AP: sum over unique recall increments (step interpolation)
    dR = np.diff(np.concatenate([[0.0], R]))
    ap = float(np.sum(dR * P))
    # trapezoidal PR-AUC, with recall axis made monotone
    # (precision is non-increasing for a fixed ranking; append endpoints)
    pr_auc = float(np.trapz(np.concatenate([[1.0], P, [0.0]]),
                            np.concatenate([[0.0], R, [1.0]])))
    return ap, pr_auc, float(R.max())


def dilate(mask, delta):
    if delta <= 0:
        return mask.copy()
    st = ndi.generate_binary_structure(2, 1)
    out = mask.copy()
    for _ in range(delta):
        out = ndi.binary_dilation(out, structure=st)
    return out


def boundary_mask(A, mask, gthr, sigma=1.0, contour_bins=CONTOUR_BINS,
                  min_size=MIN_COMPONENT_BINS):
    """Smoothed gradient-magnitude boundary, outer contour excluded."""
    filled = fill_nearest(A, mask)
    sm = ndi.gaussian_filter(filled, sigma=sigma, mode="nearest")
    gy, gx = np.gradient(sm)
    grad = np.hypot(gx, gy)
    inner = erode_mask(mask, contour_bins)
    b = inner & (grad > gthr)
    lab, n = ndi.label(b, structure=ndi.generate_binary_structure(2, 1))
    if n:
        sizes = ndi.sum(b, lab, index=np.arange(1, n + 1))
        keep = np.concatenate([[False], sizes >= min_size])
        b = keep[lab]
    return b, grad, inner


def gradient_map(A, mask, sigma=1.0):
    filled = fill_nearest(A, mask)
    sm = ndi.gaussian_filter(filled, sigma=sigma, mode="nearest")
    gy, gx = np.gradient(sm)
    return np.hypot(gx, gy)


def prf(tp, fp, fn):
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    pre = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    if np.isfinite(rec) and np.isfinite(pre) and (rec + pre) > 0:
        f1 = 2 * rec * pre / (rec + pre)
    elif np.isfinite(rec) and np.isfinite(pre):
        f1 = 0.0
    else:
        f1 = float("nan")
    return float(rec), float(pre), float(f1)


def boundary_match(b_gt, b_p, delta):
    """Boundary recall/precision with symmetric tolerance matching.

    recall    = |GT boundary within delta of pred| / |GT boundary|
    precision = |pred boundary within delta of GT| / |pred boundary|
    (dilating the prediction for precision would count tolerance inflation as
    false positives, so each side is matched against the other's dilation).
    """
    n_gt = int(b_gt.sum())
    n_p = int(b_p.sum())
    matched_gt = int(np.logical_and(b_gt, dilate(b_p, delta)).sum()) if n_gt else 0
    matched_p = int(np.logical_and(b_p, dilate(b_gt, delta)).sum()) if n_p else 0
    rec = matched_gt / n_gt if n_gt else float("nan")
    pre = matched_p / n_p if n_p else float("nan")
    if np.isfinite(rec) and np.isfinite(pre) and (rec + pre) > 0:
        f1 = 2 * rec * pre / (rec + pre)
    elif np.isfinite(rec) and np.isfinite(pre):
        f1 = 0.0
    else:
        f1 = float("nan")
    return {"n_gt": n_gt, "n_pred": n_p, "matched_gt": matched_gt,
            "matched_pred": matched_p, "recall": float(rec),
            "precision": float(pre), "f1": float(f1)}


def component_centroids(mask):
    lab, n = ndi.label(mask, structure=ndi.generate_binary_structure(2, 1))
    if n == 0:
        return np.zeros((0, 2))
    return np.array(ndi.center_of_mass(mask, lab, np.arange(1, n + 1)))


def centroid_distances(gt_b, pred_b):
    cg = component_centroids(gt_b)
    cp = component_centroids(pred_b)
    if len(cg) == 0 or len(cp) == 0:
        return float("nan")
    tree = cKDTree(cp)
    d, _ = tree.query(cg, k=1)
    return float(np.mean(d))


def mask_iou(a, b):
    u = np.logical_or(a, b).sum()
    if u == 0:
        return float("nan")
    return float(np.logical_and(a, b).sum() / u)


# ------------------------------------------------------------------ helpers --
def md5_array(a):
    a = np.ascontiguousarray(a)
    return hashlib.md5(a.tobytes()).hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slug(name):
    s = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()
    return s


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=jsonable, ensure_ascii=False)

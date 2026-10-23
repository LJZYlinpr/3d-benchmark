"""So3D metric helpers, adapted from metrics_v1 (merfish_test/missing_combos_50um):
pcc/rank stats, Moran's I / Geary's C, masked multi-scale SSIM, hotspot/boundary
precision-recall. 2D-only, sparse rasters, no target fitting.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

KNN_K = 6
SSIM_WIN = 7
SSIM_SCALES = 3
SSIM_WEIGHTS = np.array([0.0448, 0.2856, 0.3001], dtype=np.float64)
SSIM_WEIGHTS = SSIM_WEIGHTS / SSIM_WEIGHTS.sum()
CONTOUR_BINS = 2
MIN_COMPONENT_BINS = 2


# ------------------------------------------------------------------ gene stats --
def pcc_cols(a, b):
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


def cosine_cols(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.sqrt((a * a).sum(0))
    nb = np.sqrt((b * b).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b).sum(0) / (na * nb)
    r[(na == 0) | (nb == 0)] = np.nan
    return r


def spearman_cols(a, b):
    from scipy.stats import rankdata
    ra = rankdata(a, axis=0)
    rb = rankdata(b, axis=0)
    return pcc_cols(ra, rb)


def variance_ratio_log2_cols(pred, gt):
    vp = np.var(pred, axis=0, dtype=np.float64)
    vg = np.var(gt, axis=0, dtype=np.float64)
    out = np.full(vp.shape, np.nan)
    ok = vg > 0
    out[ok] = np.log2(np.maximum(vp[ok], 1e-12) / vg[ok])
    return out


# ------------------------------------------------------------------ spatial --
def knn_weights(xy, k=KNN_K):
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
    return morans_i_W(knn_weights(xy, k), vals)


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

    if pred_A is None or np.nanstd(gt_A[mask]) == 0 or np.nanstd(pred_A[mask]) == 0:
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
def _pr_curves(scores, labels):
    order = np.argsort(-scores, kind="mergesort")
    lab = labels[order].astype(np.float64)
    tp = np.cumsum(lab)
    fp = np.cumsum(1 - lab)
    P = tp / np.maximum(tp + fp, 1)
    R = tp / max(lab.sum(), 1)
    return P, R


def ap_pr_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    npos = int(labels.sum())
    if npos == 0:
        return float("nan"), float("nan"), float("nan")
    if npos == labels.size:
        return 1.0, 1.0, 1.0
    P, R = _pr_curves(scores, labels)
    dR = np.diff(np.concatenate([[0.0], R]))
    ap = float(np.sum(dR * P))
    pr_auc = float(np.trapz(np.concatenate([[1.0], P, [0.0]]),
                            np.concatenate([[0.0], R, [1.0]])))
    return ap, pr_auc, float(R.max())


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


def dilate(mask, delta):
    if delta <= 0:
        return mask.copy()
    st = ndi.generate_binary_structure(2, 1)
    out = mask.copy()
    for _ in range(delta):
        out = ndi.binary_dilation(out, structure=st)
    return out


def fill_nearest(A, mask):
    if mask.all():
        return A.copy()
    idx = ndi.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return A[tuple(idx)]


def erode_mask(mask, bins=CONTOUR_BINS):
    return ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(2, 1),
                              iterations=bins, border_value=0)


def boundary_mask(A, mask, gthr, sigma=1.0, contour_bins=CONTOUR_BINS,
                  min_size=MIN_COMPONENT_BINS):
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


def boundary_match(b_gt, b_p, delta):
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
    return {"n_gt": n_gt, "n_pred": n_p, "recall": float(rec),
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

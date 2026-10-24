"""CPU kNN shim for torch_cluster.knn_graph (torch 2.x / pyg 2.8 env).

torch_cluster wheels are unavailable for this torch build (data.pyg.org blocked,
nvcc absent). isoST only uses knn_graph, so this module reproduces its semantics
with scipy cKDTree: exact Euclidean kNN, per-graph via a batch vector, self loops
dropped when loop=False, edges returned (source, target) in torch_cluster order.
"""
import numpy as np
import torch

__version__ = "1.6.3+shim"


def knn_graph(x, k, batch=None, loop=False, flow="source_to_target",
              cosine=False, num_workers=1):
    from scipy.spatial import cKDTree

    assert flow in ["source_to_target", "target_to_source"]
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    pts_all = x.detach().cpu().numpy().astype(np.float64, copy=False)
    if cosine:
        n = np.linalg.norm(pts_all, axis=1, keepdims=True)
        pts_all = pts_all / np.clip(n, 1e-12, None)
    n_nodes = pts_all.shape[0]

    if batch is None:
        batch_np = np.zeros(n_nodes, dtype=np.int64)
    else:
        batch_np = (batch.detach().cpu().numpy() if torch.is_tensor(batch) else np.asarray(batch))
        batch_np = batch_np.astype(np.int64, copy=False)

    src, dst = [], []
    for g in np.unique(batch_np):
        idx = np.flatnonzero(batch_np == g)
        m = idx.size
        if m <= 1:
            continue
        kk = min(int(k) + (0 if loop else 1), m)
        q = cKDTree(pts_all[idx]).query(pts_all[idx], k=kk, workers=num_workers)
        nbr = np.asarray(q[1])
        if nbr.ndim == 1:
            nbr = nbr[:, None]
        tgt = np.repeat(idx[:, None], nbr.shape[1], axis=1)
        if not loop:
            keep = nbr != np.arange(m)[:, None]
            nbr = nbr[keep]
            tgt = tgt[keep]
        src.append(idx[nbr])
        dst.append(tgt)

    if not src:
        return torch.empty((2, 0), dtype=torch.long)
    src = np.concatenate(src)
    dst = np.concatenate(dst)
    if flow == "target_to_source":
        src, dst = dst, src
    return torch.from_numpy(np.stack([src, dst])).long()


def radius_graph(*a, **kw):  # pragma: no cover - not used by isoST
    raise NotImplementedError("radius_graph not implemented in isost shim")


def grid_cluster(*a, **kw):  # pragma: no cover - not used by isoST
    raise NotImplementedError("grid_cluster not implemented in isost shim")

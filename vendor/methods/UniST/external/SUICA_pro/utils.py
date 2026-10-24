import scanpy as sc
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import torch
from rich import print
from omegaconf import OmegaConf
from rich.pretty import pprint

def pprint_config(conf):
    print("[red]Current Configs:[/red]")
    conf_dict = OmegaConf.to_container(conf, resolve=True)
    pprint(conf_dict, expand_all=True)
def construct_subgraph(data, adj,neighbors,idx):
    neighbor_dict = {value: index for index, value in enumerate(neighbors)}
    new_idx = [neighbor_dict[i] for i in idx]
    sub_set_data = data[neighbors]
    sub_set_adj = adj[neighbors][:,neighbors]
    # find the new index of idx in the subgraph
    sub_set_adj = torch.tensor(sub_set_adj.toarray())
    return sub_set_data, sub_set_adj,new_idx


def construct_subgraph_cuda(data_cuda, edge_src, edge_dst, neighbors, idx):
    """GPU twin of construct_subgraph (GPU-acceleration campaign 2026-08-20,
    registered in benchmark/METHOD_CHANGES.md). Bit-identical outputs with GPU
    gathers/scatter instead of scipy sparse slicing + toarray:
      - data_cuda: full expression matrix on GPU (N, D) float32
      - edge_src/edge_dst: COO edge lists of the FULL graph (long, GPU)
      - neighbors: same python list as upstream (order preserved)
      - idx: same idx_np numpy array as upstream
    The dense sub-adjacency is filled with 1.0 at the graph's edges exactly like
    scipy's connectivity toarray (accumulate=True mirrors scipy's duplicate-summing
    semantics; the kneighbors connectivity graphs have no duplicates in practice).
    """
    neighbour_dict = {value: index for index, value in enumerate(neighbors)}
    new_idx = [neighbour_dict[i] for i in idx]
    nb_t = torch.tensor(neighbors, dtype=torch.long, device=data_cuda.device)
    sub_set_data = data_cuda[nb_t]
    n = len(neighbors)
    local_map = torch.full((data_cuda.shape[0],), -1, dtype=torch.long, device=data_cuda.device)
    local_map[nb_t] = torch.arange(n, device=data_cuda.device)
    keep = (local_map[edge_src] >= 0) & (local_map[edge_dst] >= 0)
    sub_set_adj = torch.zeros(n, n, dtype=data_cuda.dtype, device=data_cuda.device)
    if keep.any():
        sub_set_adj.index_put_(
            (local_map[edge_src[keep]], local_map[edge_dst[keep]]),
            torch.ones(int(keep.sum()), dtype=data_cuda.dtype, device=data_cuda.device),
            accumulate=True)
    return sub_set_data, sub_set_adj, new_idx


def construct_subgraph_cuda_fast(data_cuda, edge_src, edge_dst, e_row_start,
                                 idx_cuda, neighbors_cuda, local_map_buf):
    """All-GPU fast subgraph construction for GAE training/validation steps
    (2026-08-30, registered in benchmark/METHOD_CHANGES.md).

    Produces the SAME induced subgraph as construct_subgraph_cuda_edges /
    construct_subgraph_cuda (same node set, same edge multiset, same
    per-source edge order -> scatter_add result bitwise identical) without:
      - Python set/dict dedup of the neighbourhood (per-step D2H syncs)
      - scanning all ~7M graph edges (only the CSR row blocks of the subgraph
        nodes, ~n*k ~ 3e4 candidates)
      - per-step allocation of the full-size local map (caller buffer reused)

    Node ORDER differs (torch.unique sorted vs Python set hash order): a graph
    isomorphism; content_graph_conv scatter_add and BatchNorm1d are
    permutation-equivariant and the loss reads only batch-center rows, so the
    loss is unchanged up to float32 reduction-order noise (registered class).

    All inputs on GPU: data_cuda (N,D) float32; edge_src/edge_dst full COO
    (E,) long; e_row_start = adj.indptr (N+1,) long; idx_cuda (B,);
    neighbors_cuda (B,k); local_map_buf (N,) long pre-allocated (-1).

    Returns (sub_set_data, (src_local, dst_local), idx_local) on GPU.
    """
    B = idx_cuda.numel()
    merged = torch.cat([idx_cuda, neighbors_cuda.flatten()])
    uniq, inv = torch.unique(merged, return_inverse=True)  # sorted dedup
    idx_local = inv[:B]
    n = uniq.numel()
    sub_set_data = data_cuda[uniq]

    # induced edges = CSR row blocks of the subgraph nodes (same multiset as
    # the full-COO keep filter; per-source order identical to the COO).
    # Subgraph nodes are scattered across the CSR -> the within-row offset must
    # restart at 0 at each row block (a global arange offset would overshoot E).
    starts = e_row_start[uniq]
    lens = e_row_start[uniq + 1] - starts
    base = starts.repeat_interleave(lens)               # no sync (metadata sizes)
    blk_start = (torch.cumsum(lens, 0) - lens).repeat_interleave(lens)
    cand = base + torch.arange(base.numel(), device=base.device) - blk_start
    # edge sources are exactly the uniq nodes: local ids via repeat_interleave
    src_local = torch.repeat_interleave(torch.arange(n, device=starts.device), lens)

    buf = local_map_buf
    buf.fill_(-1)
    buf[uniq] = torch.arange(n, device=buf.device)
    dst_local = buf[edge_dst[cand]]
    keep = dst_local >= 0
    src_local = src_local[keep]
    dst_local = dst_local[keep]
    return sub_set_data, (src_local, dst_local), idx_local


def construct_subgraph_cuda_edges(data_cuda, edge_src, edge_dst, neighbors, idx):
    """Sparse-edge twin of construct_subgraph_cuda (2026-08-21 sparse-GCN
    campaign, registered in benchmark/METHOD_CHANGES.md). Same local-map and
    edge-filtering math, but returns the in-batch edge lists (src_local,
    dst_local) instead of the dense [n,n] adjacency. content_graph_conv then
    computes adj @ x via y[src] += x[dst] (scatter_add), which is exactly
    torch.mm(dense_adj, x) up to float accumulation order (~1e-7 relative,
    registered accepted noise class). Measured ~1.9x faster per training step
    at the official 1M-spot scale (A/B loss rel diff 0.0e+00 at float32).
    """
    neighbour_dict = {value: index for index, value in enumerate(neighbors)}
    new_idx = [neighbour_dict[i] for i in idx]
    nb_t = torch.tensor(neighbors, dtype=torch.long, device=data_cuda.device)
    sub_set_data = data_cuda[nb_t]
    n = len(neighbors)
    local_map = torch.full((data_cuda.shape[0],), -1, dtype=torch.long, device=data_cuda.device)
    local_map[nb_t] = torch.arange(n, device=data_cuda.device)
    keep = (local_map[edge_src] >= 0) & (local_map[edge_dst] >= 0)
    src_local = local_map[edge_src[keep]]
    dst_local = local_map[edge_dst[keep]]
    return sub_set_data, (src_local, dst_local), new_idx


def anisotropic_kneighbors_graph(coordinates, n_neighbors, z_weight=2.0, z_threshold=None, mode='connectivity', n_jobs=8, include_self=True):
    """
    Anisotropic KNN graph construction for 3D data with sparse z-direction.
    
    Args:
        coordinates: Spatial coordinates [N, 3] (x, y, z)
        n_neighbors: Number of neighbors
        z_weight: Weight for z-direction (>1 means reducing the influence of z-direction distance)
        z_threshold: Maximum distance threshold in z-direction (None means no threshold)
        mode: 'connectivity' or 'distance'
        n_jobs: Number of parallel jobs
        include_self: Whether to include self
    
    Returns:
        scipy.sparse.csr_matrix: Adjacency matrix
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse import csr_matrix
    import numpy as np
    
    # If z_threshold is not specified, set it automatically based on data
    if z_threshold is None:
        z_range = coordinates[:, 2].max() - coordinates[:, 2].min()
        # Auto-set threshold (e.g., 30% of z range)
        z_threshold = z_range * 0.3
    
    # Create weighted coordinates (reduce z-direction weight)
    weighted_coords = coordinates.copy()
    weighted_coords[:, 2] = weighted_coords[:, 2] / z_weight
    
    # Build KNN graph
    adj = kneighbors_graph(weighted_coords, n_neighbors, mode=mode, n_jobs=n_jobs, include_self=include_self)
    
    # If z_threshold is specified, remove connections exceeding the threshold
    if z_threshold is not None:
        from scipy.sparse import find
        
        # Get positions of all non-zero elements
        rows, cols, data = find(adj)
        
        # Calculate actual z-direction distances
        z_distances = np.abs(coordinates[rows, 2] - coordinates[cols, 2])
        
        # Remove connections exceeding threshold
        valid_mask = z_distances <= z_threshold
        
        # Rebuild adjacency matrix
        valid_rows = rows[valid_mask]
        valid_cols = cols[valid_mask]
        valid_data = data[valid_mask] if mode == 'distance' else np.ones(valid_mask.sum())
        
        # Create a new sparse matrix
        adj = csr_matrix((valid_data, (valid_rows, valid_cols)), shape=adj.shape)
        
        # Ensure symmetry (if in connectivity mode)
        if mode == 'connectivity':
            adj = (adj + adj.T > 0).astype(float)
    
    return adj

def plot_ST(coordinates, representations=None, spot_size=2, cmap="viridis", title=None):
    if representations is not None and len(representations.shape) == 1:
        # shape = (n_cells,) to (n_cells, 1)
        representations = np.expand_dims(representations, axis=1) 
    assert representations is None or representations.shape[-1] == 3 or representations.shape[-1] == 1
    if coordinates.shape[-1] == 2:
        fig = _plot_slice(coordinates, representations, spot_size, cmap, title)
    elif coordinates.shape[-1] == 3:
        fig = _plot_volume(coordinates, representations, spot_size, cmap, title)
    else:
        raise NotImplementedError
    return fig


def _plot_slice(coordinates, representations=None, spot_size=2, cmap="viridis", title=None):
    fig, ax = plt.subplots()
    ax.axis("equal")
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=14)
    x, y = coordinates[:,0], coordinates[:,1]
    if representations is None:
        ax.scatter(x, y, s=spot_size, cmap=cmap)
    else:
        z = representations
        z_norm = (z - z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0) + 0.00001)
        ax.scatter(x, y, c=z_norm, s=spot_size, cmap=cmap)
    return fig

def _plot_volume(coordinates, representatons=None, spot_size=2, cmap="viridis", title=None):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    if title:
        ax.set_title(title, fontsize=14)
    if representatons is None:
        ax.scatter(coordinates[:,0], coordinates[:,1], coordinates[:,2], s=spot_size, cmap=cmap)
    else:
        z = representatons
        z_norm = (z - z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0) + 0.00001)
        scatter = ax.scatter(coordinates[:,0], coordinates[:,1], coordinates[:,2], c=z_norm, s=spot_size, cmap=cmap)
        fig.colorbar(scatter, ax=ax)
    return fig


def read_anndata(h5ad_file):
    return sc.read_h5ad(h5ad_file)


def read_preprocess_anndata(h5ad_file, min_genes=200, min_cells=3):
    adata = sc.read_h5ad(h5ad_file)
    raw_size = adata.X.shape
    sc.pp.filter_cells(adata, min_counts=min_genes) # filter cells containing less than #min_genes genes
    sc.pp.filter_genes(adata, min_counts=min_cells) # filter genes appearing in less than #min_cells cells
    new_size = adata.X.shape
    print(f"Filtering ST data from {raw_size} to {new_size} ...")
    return adata


def _cosine_similarity(y_true, y_pred, mask=False):
    if mask:
        y_true, y_pred = y_true[y_true>0], y_pred[y_true>0]
    numerator = np.sum(y_true * y_pred, axis=-1)  
    denominator = np.sqrt(np.sum(y_true ** 2, axis=-1)) * np.sqrt(np.sum(y_pred ** 2, axis=-1))  
    pixelwise_cosine = numerator / (denominator + 0.00001)
    return pixelwise_cosine.mean() 

def _spectral_angle_mapper(y_true, y_pred, mask=False):
    if mask:
        y_true, y_pred = y_true[y_true>0], y_pred[y_true>0]
    numerator = np.sum(y_true * y_pred, axis=-1)  
    denominator = np.sqrt(np.sum(y_true ** 2, axis=-1)) * np.sqrt(np.sum(y_pred ** 2, axis=-1))  
    pixelwise_cosine = numerator / denominator
    cos_theta = np.clip(pixelwise_cosine, -1.0, 1.0) 
    sam_angle = np.rad2deg(np.arccos(cos_theta))
    return sam_angle.mean()

def _spearman_r(y_true, y_pred, mask=False):
    if mask:
        corrs = []
        for i in range(y_pred.shape[0]):
            y_t = y_true[i][y_true[i]>0]
            y_p = y_pred[i][y_true[i]>0]
            if len(y_t) >= 2 and len(y_p) >= 2:
                corrs.append(np.nan_to_num(spearmanr(y_t, y_p).statistic))
            else:
                corrs.append(0.0)  # Insufficient data points, set to 0
        corrs = np.array(corrs)
    else:
        corrs = []
        for i in range(y_pred.shape[0]):
            if len(y_true[i]) >= 2 and len(y_pred[i]) >= 2:
                corrs.append(np.nan_to_num(spearmanr(y_true[i], y_pred[i]).statistic))
            else:
                corrs.append(0.0)  # Insufficient data points, set to 0
        corrs = np.array(corrs)
    return corrs.mean()

def _pearson_r(y_true, y_pred, mask=False):
    if mask:
        corrs = []
        for i in range(y_pred.shape[0]):
            y_t = y_true[i][y_true[i]>0]
            y_p = y_pred[i][y_true[i]>0]
            if len(y_t) >= 2 and len(y_p) >= 2:
                corrs.append(np.nan_to_num(pearsonr(y_t, y_p).statistic))
            else:
                corrs.append(0.0)  # Insufficient data points, set to 0
        corrs = np.array(corrs)
    else:
        corrs = []
        for i in range(y_pred.shape[0]):
            if len(y_true[i]) >= 2 and len(y_pred[i]) >= 2:
                corrs.append(np.nan_to_num(pearsonr(y_true[i], y_pred[i]).statistic))
            else:
                corrs.append(0.0)  # Insufficient data points, set to 0
        corrs = np.array(corrs)
    return corrs.mean()

# Intersection over Union of zero-map
def _IoU(y_true, y_pred):
    zero_map_A, zero_map_B = (y_true == 0), (y_pred == 0)
    intersection = np.logical_and(zero_map_A, zero_map_B).sum()
    union = np.logical_or(zero_map_A, zero_map_B).sum()

    iou = intersection / union if union != 0 else 0
    return iou

# Intersection over Union of non-zero-map
def _support_recovery_rate(y_true, y_pred):
    one_map_A, one_map_B = (y_true > 0), (y_pred > 0)
    intersection = np.logical_and(one_map_A, one_map_B).sum()
    union = np.logical_or(one_map_A, one_map_B).sum()
    iou = intersection / union if union != 0 else 0
    return iou

def _masked_MSE(y_true, y_pred):
    return mean_squared_error(y_true[y_true>0], y_pred[y_true>0])

def _masked_MAE(y_true, y_pred):
    return mean_absolute_error(y_true[y_true>0], y_pred[y_true>0])

def metrics(y_true, y_pred, prefix="val", fast=False):
    if isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
    scores = {
        f"{prefix}/mean_absolute_error": mean_absolute_error(y_true, y_pred),
        f"{prefix}/mean_absolute_error_mask": _masked_MAE(y_true, y_pred),
        f"{prefix}/mean_squared_error": mean_squared_error(y_true, y_pred),
        f"{prefix}/mean_squared_error_mask": _masked_MSE(y_true, y_pred),
        f"{prefix}/root_mean_squared_error": root_mean_squared_error(y_true, y_pred),
        f"{prefix}/cosine_similarity": _cosine_similarity(y_true, y_pred),
        f"{prefix}/cosine_similarity_mask": _cosine_similarity(y_true, y_pred, mask=True),
        f"{prefix}/sam": _spectral_angle_mapper(y_true, y_pred),
        # f"{prefix}/sam_mask": _spectral_angle_mapper(y_true, y_pred, mask=True),
        f"{prefix}/iou": _IoU(y_true, y_pred),
    }
    if not fast:
        slow_metrics = {
            # may take some time
            #f"{prefix}/r2_score": r2_score(y_true, y_pred),
            f"{prefix}/pearsonr": _pearson_r(y_true, y_pred),
            f"{prefix}/spearmanr": _spearman_r(y_true, y_pred), 
            f"{prefix}/pearsonr_mask": _pearson_r(y_true, y_pred, mask=True),
            f"{prefix}/spearmanr_mask": _spearman_r(y_true, y_pred, mask=True), 
        }
        scores.update(slow_metrics)

    return scores


if __name__ == "__main__":
    np.random.seed(0)

    # Example arrays
    A = np.random.rand(10000, 20000)
    B = np.random.rand(10000, 20000)

    print(_spectral_angle_mapper(A,B))
    print(_cosine_similarity(A,B))
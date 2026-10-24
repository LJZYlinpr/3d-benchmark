import torch
from torch.utils.data import DataLoader, Subset
import lightning as L
from lightning import pytorch as pl
from sklearn.model_selection import train_test_split
from rich import print
import anndata as ad
import os
import numpy as np
from omegaconf import ListConfig
import importlib
import gc
from datasets import ST2D, GraphST2D, ST3D, GraphST3D
from networks import AE, GAE
from utils import (metrics, plot_ST, construct_subgraph, construct_subgraph_cuda,
                   construct_subgraph_cuda_edges, construct_subgraph_cuda_fast)

class ExtraVariableCallback(pl.Callback):
    def __init__(self, extra_variable):
        self.extra_variable = extra_variable  # Store your extra variable

    def on_train_start(self, trainer, pl_module):
        pl_module.extra_variable = self.extra_variable
        print(f"Extra variable passed to LightningModule for training.")

    def on_validation_start(self, trainer, pl_module):
        pl_module.extra_variable = self.extra_variable
        print(f"Extra variable passed to LightningModule for validation.")

    def on_predict_start(self, trainer, pl_module):
        pl_module.extra_variable = self.extra_variable
        print(f"Extra variable passed to LightningModule for prediction.")

    # These methods are required by some stateful callbacks to avoid the "Expected a parent" error
    def state_dict(self):
        # Return an empty dict since no state needs to be saved
        return {}
    
class EmbedderFittingSystem(L.LightningModule):
    def __init__(self, configs, val_pca=None):
        super().__init__()

        self.save_hyperparameters()

        self.pipeline_configs = configs
        network_configs = self.pipeline_configs.embedder

        if network_configs.model == "AE":
            self.GNN = False
            self.fitting_model = AE(
                dim_in=network_configs.dim_in,
                dim_hidden=network_configs.dim_hidden,
                dim_latent=network_configs.dim_latent
            )
        elif network_configs.model == "GAE":
            self.GNN = True
            self.fitting_model = GAE(
                dim_in=network_configs.dim_in,
                dim_hidden=network_configs.dim_hidden,
                dim_latent=network_configs.dim_latent
            )
        else:
            raise NotImplementedError

        self.val_pca = val_pca
        self.output_cache = []
        
    
    def forward(self, x):
        recon, embd = self.fitting_model(x)
        return recon, embd
    
    def training_step(self, batch, batch_idx):
        if self.GNN:
            idx, neighbors = batch["idx"],batch["neighbors"]
            extra_info = self.extra_variable
            adj = extra_info["adj_train"]
            raw_rep = extra_info["raw_rep_train"]
            if os.environ.get("UNIST_FAST_SUBGRAPH") == "1" and "e_row_start_train" in extra_info:
                # 2026-08-30 GPU-0 acceleration (benchmark/METHOD_CHANGES.md):
                # subgraph build fully on GPU — torch.unique sorted dedup instead
                # of the Python set, CSR row-block candidate edges instead of the
                # full-COO scan, reused local-map buffer, zero per-step .item()
                # or D2H syncs. Same induced subgraph (same node set, same edge
                # multiset, same per-source edge order -> scatter_add bitwise
                # identical); node ORDER differs (graph isomorphism, registered
                # permutation noise class).
                sub_set_y_raw, sub_set_adj, sub_set_idx = construct_subgraph_cuda_fast(
                    extra_info["raw_rep_train_cuda"],
                    extra_info["adj_train_src"], extra_info["adj_train_dst"],
                    extra_info["e_row_start_train"],
                    idx, neighbors,
                    extra_info["local_map_buf_train"])
            else:
                neighbours = list(neighbors.cpu().numpy().flatten())
                idx_np = idx.cpu().numpy()
                # Ensure idx nodes are included in neighbourhoods
                neighbourhoods = list(set(neighbours) | set(idx_np))
                if os.environ.get("UNIST_SPARSE_GCN") == "1" and "adj_train_src" in extra_info:
                    # 2026-08-21 sparse-GCN campaign (benchmark/METHOD_CHANGES.md):
                    # in-batch edge lists + scatter GCN instead of the dense [n,n]
                    # adjacency + torch.mm. Same math (A/B loss rel diff 0.0e+00 at
                    # float32; worst-case ~1e-7 registered noise class); ~1.9x faster
                    # at the official 1M-spot scale.
                    sub_set_y_raw, sub_set_adj, sub_set_idx = construct_subgraph_cuda_edges(
                        extra_info["raw_rep_train_cuda"],
                        extra_info["adj_train_src"], extra_info["adj_train_dst"],
                        neighbourhoods, idx_np)
                elif "adj_train_src" in extra_info:
                    # GPU path (2026-08-20, benchmark/METHOD_CHANGES.md): bit-identical
                    # subgraph tensors computed on GPU (falls back to the upstream path
                    # below when the CUDA prep is absent).
                    sub_set_y_raw, sub_set_adj, sub_set_idx = construct_subgraph_cuda(
                        extra_info["raw_rep_train_cuda"],
                        extra_info["adj_train_src"], extra_info["adj_train_dst"],
                        neighbourhoods, idx_np)
                else:
                    sub_set_y_raw, sub_set_adj,sub_set_idx = construct_subgraph(raw_rep, adj, neighbourhoods,idx_np)
            sub_set_y_raw = sub_set_y_raw.cuda().float() if not sub_set_y_raw.is_floating_point() else sub_set_y_raw.cuda()
            sub_set_adj = sub_set_adj.cuda() if torch.is_tensor(sub_set_adj) else sub_set_adj
            loss, _, _ = self.fitting_model.forward_loss(sub_set_y_raw,sub_set_adj,sub_set_idx)
            # audit 2026-08-14: removed the per-batch `del adj; del raw_rep;
            # gc.collect()` (local no-ops — the dict keeps the arrays) and the
            # per-batch `torch.cuda.empty_cache()` (pure memory-management calls).
            # Computation-organization change only: identical loss math, identical
            # gradients; ~3x faster at the official full-slice scale.
            # Registered in benchmark/METHOD_CHANGES.md.
        else:
            y_raw = batch["raw_representations"]
            # Ensure y_raw is float for loss computation
            y_raw = y_raw.float() if not y_raw.is_floating_point() else y_raw
            loss, _, _ = self.fitting_model.forward_loss(y_raw)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.GNN:
            y_raw, x,idx,neighbors = batch["raw_representations"], batch["coordinates"], batch["idx"],batch["neighbors"]
            extra_info = self.extra_variable
            adj = extra_info["adj_val"]
            raw_rep = extra_info["raw_rep_val"]
            if os.environ.get("UNIST_FAST_SUBGRAPH") == "1" and "e_row_start_val" in extra_info:
                # 2026-08-30 GPU-0 acceleration (benchmark/METHOD_CHANGES.md):
                # all-GPU subgraph build, same as the training fast path.
                sub_set_y_raw, sub_set_adj, sub_set_idx = construct_subgraph_cuda_fast(
                    extra_info["raw_rep_val_cuda"],
                    extra_info["adj_val_src"], extra_info["adj_val_dst"],
                    extra_info["e_row_start_val"],
                    idx, neighbors,
                    extra_info["local_map_buf_val"])
            else:
                neighbours = list(neighbors.cpu().numpy().flatten())
                idx_np = idx.cpu().numpy()
                # Ensure idx nodes are included in neighbourhoods
                neighbourhoods = list(set(neighbours) | set(idx_np))
                if os.environ.get("UNIST_SPARSE_GCN") == "1" and "adj_val_src" in extra_info:
                    # 2026-08-21 sparse-GCN campaign (benchmark/METHOD_CHANGES.md):
                    # same math via in-batch edge lists + scatter GCN.
                    sub_set_y_raw, sub_set_adj, sub_set_idx = construct_subgraph_cuda_edges(
                        extra_info["raw_rep_val_cuda"],
                        extra_info["adj_val_src"], extra_info["adj_val_dst"],
                        neighbourhoods, idx_np)
                elif "adj_val_src" in extra_info:
                    # GPU path (2026-08-20, benchmark/METHOD_CHANGES.md): bit-identical
                    # subgraph tensors computed on GPU.
                    sub_set_y_raw, sub_set_adj, sub_set_idx = construct_subgraph_cuda(
                        extra_info["raw_rep_val_cuda"],
                        extra_info["adj_val_src"], extra_info["adj_val_dst"],
                        neighbourhoods, idx_np)
                else:
                    sub_set_y_raw, sub_set_adj,sub_set_idx = construct_subgraph(raw_rep, adj, neighbourhoods,idx_np)
            sub_set_y_raw = sub_set_y_raw.cuda().float() if not sub_set_y_raw.is_floating_point() else sub_set_y_raw.cuda()
            sub_set_adj = sub_set_adj.cuda() if torch.is_tensor(sub_set_adj) else sub_set_adj
            loss, y_hat, _ = self.fitting_model.forward_loss(sub_set_y_raw,sub_set_adj,sub_set_idx)

        else:
            y_raw, x = batch["raw_representations"], batch["coordinates"]
            # Ensure y_raw is float for loss computation
            y_raw = y_raw.float() if not y_raw.is_floating_point() else y_raw
            loss, y_hat, _ = self.fitting_model.forward_loss(y_raw)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.output_cache.append({
            "y_raw": y_raw.detach().cpu().numpy(),
            "y_hat": y_hat.detach().cpu().numpy(),
            "x": x.detach().cpu().numpy()
        })

    def on_validation_epoch_end(self):
        outputs = self.output_cache
        all_y_raw = np.concatenate([x["y_raw"] for x in outputs], axis=0)
        all_y_hat = np.concatenate([x["y_hat"] for x in outputs], axis=0)
        all_x = np.concatenate([x["x"] for x in outputs], axis=0)
        
        scores = metrics(all_y_raw, all_y_hat)
        self.log_dict(scores, sync_dist=False)
        fig = plot_ST(all_x, self.val_pca.transform(all_y_hat))
        self.logger.experiment.add_figure("val/y_hat", fig, self.current_epoch)
        print(scores)

        # Don't forget to clear the memory for the next epoch!
        self.output_cache.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.fitting_model.parameters(), lr=self.pipeline_configs.optimization.lr)
        return optimizer

    def predict_step(self, batch, batch_idx):
        if self.GNN:
            y_raw, x,neighbors,idx  = batch["raw_representations"], batch["coordinates"],batch["neighbors"], batch["idx"]
            extra_info = self.extra_variable
            adj = extra_info["adj"]
            raw_rep = extra_info["raw_rep"]
            
            neighbours = list(neighbors.cpu().numpy().flatten())
            idx_np = idx.cpu().numpy()
            # Ensure idx nodes are included in neighbourhoods
            neighbourhoods = list(set(neighbours) | set(idx_np))
            sub_set_y_raw, sub_set_adj,sub_set_idx = construct_subgraph(raw_rep, adj, neighbourhoods,idx_np)
            sub_set_y_raw = sub_set_y_raw.cuda()
            sub_set_adj = sub_set_adj.cuda()
            y_hat, embd = self.fitting_model.generate(sub_set_y_raw, sub_set_adj, sub_set_idx)

        else:
            y_raw, x = batch["raw_representations"], batch["coordinates"]
            y_hat, embd = self.fitting_model(y_raw)
        self.output_cache.append({
            "x": x.detach().cpu().numpy(),
            "y_raw": y_raw.detach().cpu().numpy(),
            "y_hat": y_hat.detach().cpu().numpy(),
            "embd": embd.detach().cpu().numpy()
        })
        
    def on_predict_epoch_end(self):
        from scipy.sparse import csr_matrix
        outputs = self.output_cache
        all_y_raw = np.concatenate([x["y_raw"] for x in outputs], axis=0)
        all_y_raw = csr_matrix(all_y_raw)
        #all_y_hat = np.concatenate([x["y_hat"] for x in outputs], axis=0)
        all_x = np.concatenate([x["x"] for x in outputs], axis=0)
        all_embd = np.concatenate([x["embd"] for x in outputs], axis=0)
        del outputs
        del self.output_cache
        gc.collect()
        if isinstance(self.pipeline_configs.embedded_data, ListConfig):
            save_name = self.pipeline_configs.embedded_data.pop(0)
        else:
            save_name = self.pipeline_configs.embedded_data

        write_file = os.path.join(self.logger.log_dir, save_name)
        print(f"Writing to {write_file} ... It may take some time ...")
        if save_name.endswith(".h5ad"):
            embedded_data = ad.AnnData(X=all_y_raw)
            embedded_data.obsm["spatial"] = all_x
            embedded_data.obsm["embeddings"] = all_embd
            embedded_data.write(write_file)
        elif save_name.endswith(".npz"):
            np.savez(write_file, spatial=all_x, raw=all_y_raw, embeddings=all_embd)
        


def extract_neighbors_from_csr(adj, n_neighbors):
    """
    Extract neighbors from a CSR sparse matrix and reshape to (n_nodes, n_neighbors).
    
    Handles cases where rows may have fewer than n_neighbors neighbors (e.g., after filtering).
    Pads with the last neighbor if needed.
    
    Optimized for performance: fast path for standard case, efficient handling of edge cases.
    
    Args:
        adj: scipy.sparse.csr_matrix - adjacency matrix (will be converted to CSR if needed)
        n_neighbors: int - expected number of neighbors per node
    
    Returns:
        numpy.ndarray: array of shape (n_nodes, n_neighbors) containing neighbor indices
    """
    from scipy.sparse import csr_matrix
    # Ensure CSR format
    if not isinstance(adj, csr_matrix):
        adj = adj.tocsr()
    
    n_nodes = adj.shape[0]
    
    # Fast path: check if all rows have exactly n_neighbors neighbors
    row_lengths = adj.indptr[1:] - adj.indptr[:-1]
    
    # If all rows have exactly n_neighbors neighbors, use direct reshape
    if np.all(row_lengths == n_neighbors):
        return adj.indices.reshape(-1, n_neighbors)
    
    # Otherwise, handle variable-length rows efficiently
    print(f"[yellow]Extracting neighbors from CSR matrix ({n_nodes} nodes)...[/yellow]")
    neighbors_array = np.zeros((n_nodes, n_neighbors), dtype=adj.indices.dtype)
    
    # Process rows efficiently with progress updates for large datasets
    update_interval = max(1, n_nodes // 10)  # Update every 10%
    
    for i in range(n_nodes):
        start = adj.indptr[i]
        end = adj.indptr[i + 1]
        n_neigh = end - start
        
        if n_neigh > 0:
            # Take up to n_neighbors neighbors
            n_copy = min(n_neigh, n_neighbors)
            neighbors_array[i, :n_copy] = adj.indices[start:start+n_copy]
            # Pad with last neighbor if needed
            if n_neigh < n_neighbors:
                neighbors_array[i, n_copy:] = adj.indices[end-1]
        else:
            # If no neighbors, use self as neighbor
            neighbors_array[i, :] = i
        
        # Progress update for large datasets
        if n_nodes > 10000 and (i + 1) % update_interval == 0:
            print(f"  Processed {i+1}/{n_nodes} nodes ({100*(i+1)//n_nodes}%)")
    
    print(f"[green]Neighbor extraction completed[/green]")
    return neighbors_array


def train_embedder(configs):
    dataset_configs = configs.dataset
    pipeline_configs = configs.pipeline
    
    pl.seed_everything(pipeline_configs.optimization.seed, workers=True) # fix seed globally
    torch.set_float32_matmul_precision("highest") # make use of tensorcore
    if pipeline_configs.embedder.model == 'GAE':
        import scanpy as sc
        from scipy.sparse import issparse
        from sklearn.neighbors import kneighbors_graph
        from utils import anisotropic_kneighbors_graph
        
        n_neighbors = configs.dataset.n_neighbors
        adata =  sc.read_h5ad(dataset_configs.data_file)
        
        # For 3D data, support anisotropic graph construction (sparse z-direction)
        if dataset_configs.type == 'GraphST3D':
            # Get anisotropic parameters (optional)
            z_weight = getattr(dataset_configs, 'z_weight', 2.0)  # z-direction weight
            z_threshold = getattr(dataset_configs, 'z_threshold', None)  # z-direction threshold
            use_anisotropic = getattr(dataset_configs, 'use_anisotropic_knn', True)  # Whether to use anisotropic
            
            if use_anisotropic:
                print(f"[yellow]Using anisotropic KNN for 3D data: z_weight={z_weight}, z_threshold={z_threshold}[/yellow]")
                adj = anisotropic_kneighbors_graph(
                    adata.obsm['spatial'], 
                    n_neighbors, 
                    z_weight=z_weight, 
                    z_threshold=z_threshold,
                    mode='connectivity',
                    n_jobs=8,
                    include_self=True
                )
            else:
                adj = kneighbors_graph(adata.obsm['spatial'], n_neighbors, mode='connectivity', n_jobs=8, include_self=True)
        else:
            adj = kneighbors_graph(adata.obsm['spatial'], n_neighbors, mode='connectivity', n_jobs=8, include_self=True)
        
        adj = adj.astype(np.float32)
        
        # Get 3D-specific normalization parameters
        preserve_z_scale = getattr(dataset_configs, 'preserve_z_scale', False)
        z_scale_factor = getattr(dataset_configs, 'z_scale_factor', 1.0)
        
        if dataset_configs.type == 'GraphST2D':
            dataset = GraphST2D(h5ad_file=dataset_configs.data_file, neighbors=extract_neighbors_from_csr(adj, n_neighbors),  keep_ratio=dataset_configs.keep_ratio)
        elif dataset_configs.type == 'GraphST3D':
            dataset = GraphST3D(
                h5ad_file=dataset_configs.data_file, 
                neighbors=extract_neighbors_from_csr(adj, n_neighbors), 
                keep_ratio=dataset_configs.keep_ratio,
                preserve_z_scale=preserve_z_scale,
                z_scale_factor=z_scale_factor,
                require_coordnorm=dataset_configs.require_coordnorm
            )
        if issparse(adata.X):
            adata.X = adata.X.toarray()
        raw_rep = torch.tensor(adata.X, dtype=torch.float32)
        train_idx, val_idx = train_test_split(list(range(len(adata))), test_size=dataset_configs.val_proportion)
        adata_train = adata[train_idx]
        adata_val = adata[val_idx]
        print(len(adata_train),len(adata_val))
        print(f"{val_idx[:10]=}") # check whether seed works
        # For 3D data, training and validation sets also use anisotropic graph construction
        if dataset_configs.type == 'GraphST3D' and getattr(dataset_configs, 'use_anisotropic_knn', True):
            z_weight = getattr(dataset_configs, 'z_weight', 2.0)
            z_threshold = getattr(dataset_configs, 'z_threshold', None)
            adj_train = anisotropic_kneighbors_graph(
                adata[train_idx].obsm['spatial'], 
                n_neighbors, 
                z_weight=z_weight, 
                z_threshold=z_threshold,
                mode='connectivity',
                n_jobs=8,
                include_self=True
            )
            adj_val = anisotropic_kneighbors_graph(
                adata[val_idx].obsm['spatial'], 
                n_neighbors, 
                z_weight=z_weight, 
                z_threshold=z_threshold,
                mode='connectivity',
                n_jobs=8,
                include_self=True
            )
        else:
            adj_train = kneighbors_graph(adata[train_idx].obsm['spatial'],n_neighbors,mode='connectivity',n_jobs=8,include_self=True)
            adj_val = kneighbors_graph(adata[val_idx].obsm['spatial'],n_neighbors,mode='connectivity',n_jobs=8,include_self=True)
        
        preserve_z_scale = getattr(dataset_configs, 'preserve_z_scale', False)
        z_scale_factor = getattr(dataset_configs, 'z_scale_factor', 1.0)
        
        if dataset_configs.type == 'GraphST2D':
            train_dataset = GraphST2D(h5ad_file=adata_train,neighbors=extract_neighbors_from_csr(adj_train, n_neighbors) ,keep_ratio=dataset_configs.keep_ratio)
            val_dataset = GraphST2D(h5ad_file=adata_val,neighbors=extract_neighbors_from_csr(adj_val, n_neighbors) , keep_ratio=dataset_configs.keep_ratio)
        elif dataset_configs.type == 'GraphST3D':
            train_dataset = GraphST3D(
                h5ad_file=adata_train, 
                neighbors=extract_neighbors_from_csr(adj_train, n_neighbors), 
                keep_ratio=dataset_configs.keep_ratio,
                preserve_z_scale=preserve_z_scale,
                z_scale_factor=z_scale_factor,
                require_coordnorm=dataset_configs.require_coordnorm
            )
            val_dataset = GraphST3D(
                h5ad_file=adata_val, 
                neighbors=extract_neighbors_from_csr(adj_val, n_neighbors), 
                keep_ratio=dataset_configs.keep_ratio,
                preserve_z_scale=preserve_z_scale,
                z_scale_factor=z_scale_factor,
                require_coordnorm=dataset_configs.require_coordnorm
            )

        batch_size = pipeline_configs.optimization.batch_size
        raw_rep_train = torch.tensor(adata[train_idx].X, dtype=torch.float32)
        raw_rep_val = torch.tensor(adata[val_idx].X, dtype=torch.float32)
        adj_train = adj_train.astype(np.float32)
        adj_val = adj_val.astype(np.float32)

        extra_info = {"raw_rep":raw_rep,"adj": adj,"raw_rep_train": raw_rep_train, "adj_train":adj_train, "raw_rep_val":raw_rep_val, "adj_val":adj_val}
        # GPU-acceleration campaign 2026-08-20 (registered in benchmark/METHOD_CHANGES.md):
        # precompute CUDA copies of the full expression matrices and the COO edge lists of
        # the full graphs; training/validation steps then build the per-batch dense
        # sub-adjacency on GPU with construct_subgraph_cuda (bit-identical outputs).
        def _csr_edges(adj_csr):
            src = np.repeat(np.arange(adj_csr.shape[0]), np.diff(adj_csr.indptr))
            return torch.tensor(src, dtype=torch.long), torch.tensor(adj_csr.indices, dtype=torch.long)
        if torch.cuda.is_available():
            extra_info["raw_rep_train_cuda"] = raw_rep_train.cuda()
            extra_info["raw_rep_val_cuda"] = raw_rep_val.cuda()
            s_tr, d_tr = _csr_edges(adj_train)
            s_val, d_val = _csr_edges(adj_val)
            extra_info["adj_train_src"], extra_info["adj_train_dst"] = s_tr.cuda(), d_tr.cuda()
            extra_info["adj_val_src"], extra_info["adj_val_dst"] = s_val.cuda(), d_val.cuda()
            # 2026-08-30 GPU-0 acceleration (benchmark/METHOD_CHANGES.md): CSR row
            # pointers (indptr -> int64) and a reused -1-filled scratch buffer for
            # the all-GPU subgraph build construct_subgraph_cuda_fast. The buffer
            # is plain per-train/val scratch; per-step state lives inside the fast
            # function.
            extra_info["e_row_start_train"] = torch.tensor(adj_train.indptr, dtype=torch.long).cuda()
            extra_info["e_row_start_val"] = torch.tensor(adj_val.indptr, dtype=torch.long).cuda()
            extra_info["local_map_buf_train"] = torch.full(
                (raw_rep_train.shape[0],), -1, dtype=torch.long,
                device=extra_info["raw_rep_train_cuda"].device)
            extra_info["local_map_buf_val"] = torch.full(
                (raw_rep_val.shape[0],), -1, dtype=torch.long,
                device=extra_info["raw_rep_val_cuda"].device)
        pipeline_configs.embedder.dim_in = train_dataset.n_gene
        extra_callback = ExtraVariableCallback(extra_variable=extra_info)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=18, drop_last=False)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=18, drop_last=False)

    else:
        # dataset configuration
        dataset_class = getattr(importlib.import_module("datasets"), dataset_configs.type)
        dataset = dataset_class(**dataset_configs)

        pipeline_configs.embedder.dim_in = dataset.get_raw_dim()

        print("Fitting embedder ...")

        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=dataset_configs.val_proportion)
        print(f"{val_idx[:10]=}") # check whether seed works
        train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx)
        

        batch_size = pipeline_configs.optimization.batch_size
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=8, drop_last=False)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=8, drop_last=False)
        extra_callback = ExtraVariableCallback(extra_variable=None)
    # pipeline configuration
    fitting_system = EmbedderFittingSystem(pipeline_configs, val_pca=dataset.raw_pca)
    tb_logger = pl.loggers.TensorBoardLogger(pipeline_configs.optimization.logs)

    # log raw representations to tensorboard
    train_raw_fig, val_raw_fig = dataset.plot_raw_representations(train_indices=train_idx, val_indices=val_idx)
    tb_logger.experiment.add_figure("dataset/train_raw", train_raw_fig)
    tb_logger.experiment.add_figure("dataset/val_raw", val_raw_fig)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename="{epoch}", save_last=True)

    trainer = L.Trainer(
                #num_sanity_val_steps=0,
                max_epochs=pipeline_configs.optimization.epochs,
                check_val_every_n_epoch=pipeline_configs.optimization.val_freq,
                log_every_n_steps=1, 
                logger=tb_logger, 
                callbacks=[extra_callback, checkpoint_callback],
                devices=1
            )
    trainer.fit(fitting_system, train_dataloader, val_dataloader)
    # del trainer
    if dataset_configs.type == 'GraphST2D':
        del adj_train
        del adj
        del raw_rep
        del raw_rep_train
    del train_dataset
    del train_dataloader
    del extra_callback
    import gc
    gc.collect()
    # predict
    if pipeline_configs.predict_mode=="all":
        del val_dataset
        del val_dataloader
        gc.collect()
        test_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=1, drop_last=False)
        trainer.predict(fitting_system, dataloaders=test_dataloader)
    elif pipeline_configs.predict_mode=="val":
        test_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=8, drop_last=False)
        trainer.predict(fitting_system, dataloaders=test_dataloader)
    elif pipeline_configs.predict_mode=="resample":
        test_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=8, drop_last=False)
        trainer.predict(fitting_system, dataloaders=test_dataloader)
        test_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=8, drop_last=False)
        trainer.predict(fitting_system, dataloaders=test_dataloader)
    

    else:
        print("End without writing predictions.")

from torch.utils.data import Dataset
import torch
import numpy as np
from einops import rearrange
import scipy.sparse as sp
from sklearn.decomposition import PCA

from utils import plot_ST, read_anndata

class ST2D(Dataset):
    def __init__(self, data_file, require_coordnorm=True, keep_ratio=True, **kwargs):
        super().__init__()
        adata = read_anndata(data_file)
        
        self.coordinates = adata.obsm["spatial"][:, :2].astype(float)
        
        if sp.issparse(adata.X):
            self.raw_representations = adata.X.toarray()
        else:
            self.raw_representations = adata.X
        
        # take statistics
        self.n_cell = self.raw_representations.shape[0]
        self.n_gene = self.raw_representations.shape[1]
        self.n_channels = self.n_gene
        
        if "embeddings" in adata.obsm:
            self.embeddings = adata.obsm["embeddings"]
            assert self.raw_representations.shape[0] == self.embeddings.shape[0]
            self.n_embd = self.embeddings.shape[1]
        else: 
            self.embeddings = None

        if require_coordnorm:
            self._normalize_coordinates(keep_ratio=keep_ratio)

        # PCA: when number of features < 3, n_components cannot exceed number of features
        n_components_raw = min(3, self.n_gene)
        self.raw_pca = PCA(n_components=n_components_raw, random_state=0) # map raw representation dimension to 3 for visualization
        self.raw_pca.fit(self.raw_representations)

        if self.has_embeddings():
            n_components_embd = min(3, self.n_embd)
            self.embd_pca = PCA(n_components=n_components_embd, random_state=0) # map embedding dimension to 3 for visualization
            self.embd_pca.fit(self.embeddings)
        
    
    def has_embeddings(self):
        return False if self.embeddings is None else True

    def plot_raw_representations(self, spot_size=2, train_indices=None, val_indices=None):
        # PCA output may be < 3 dimensions, need to pad to 3 dimensions for visualization
        def _pad_to_3d(rep):
            if rep.shape[1] == 3:
                return rep
            elif rep.shape[1] == 2:
                return np.concatenate([rep, np.zeros((rep.shape[0], 1))], axis=1)
            elif rep.shape[1] == 1:
                return np.repeat(rep, 3, axis=1)
            else:
                return rep
        
        if train_indices and val_indices:
            train_rep = _pad_to_3d(self.raw_pca.transform(self.raw_representations[train_indices,:]))
            val_rep = _pad_to_3d(self.raw_pca.transform(self.raw_representations[val_indices,:]))
            train_fig = plot_ST(self.coordinates[train_indices,:], train_rep, spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], val_rep, spot_size)
            return train_fig, val_fig
        else:
            rep = _pad_to_3d(self.raw_pca.transform(self.raw_representations))
            fig = plot_ST(self.coordinates, rep, spot_size)
        return fig
    
    def plot_embeddings(self, spot_size=2, train_indices=None, val_indices=None):
        assert self.has_embeddings(), "The current adata file has NO embeddings!"
        # PCA output may be < 3 dimensions, need to pad to 3 dimensions for visualization
        def _pad_to_3d(rep):
            if rep.shape[1] == 3:
                return rep
            elif rep.shape[1] == 2:
                return np.concatenate([rep, np.zeros((rep.shape[0], 1))], axis=1)
            elif rep.shape[1] == 1:
                return np.repeat(rep, 3, axis=1)
            else:
                return rep
        
        if train_indices and val_indices:
            train_rep = _pad_to_3d(self.embd_pca.transform(self.embeddings[train_indices,:]))
            val_rep = _pad_to_3d(self.embd_pca.transform(self.embeddings[val_indices,:]))
            train_fig = plot_ST(self.coordinates[train_indices,:], train_rep, spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], val_rep, spot_size)
            return train_fig, val_fig
        else:
            rep = _pad_to_3d(self.embd_pca.transform(self.embeddings))
            fig = plot_ST(self.coordinates, rep, spot_size)
            return fig
    

    # normalize coordinates to [-1.0, +1.0]
    def _normalize_coordinates(self, keep_ratio):
        x_min, y_min = list(self.coordinates.min(axis=0))
        x_max, y_max = list(self.coordinates.max(axis=0))
        x_range, y_range = x_max - x_min, y_max - y_min

        self.coordinates[:,0] = (self.coordinates[:,0] - x_min) / x_range
        self.coordinates[:,1] = (self.coordinates[:,1] - y_min) / y_range

        self.coordinates -= 0.5
        self.coordinates *= 2.0

        if keep_ratio: # may cause waste of space in the short side
            max_range = max(x_range, y_range)
            scale_x, scale_y = x_range / max_range, y_range / max_range
            self.coordinates[:,0] *= scale_x
            self.coordinates[:,1] *= scale_y
    
    def __len__(self):
        return self.n_cell
    
    def get_raw_dim(self):
        return self.n_gene
    
    def get_embd_dim(self):
        if self.has_embeddings():
            return self.n_embd
        else:
            return None
    
    def __getitem__(self, idx):
        if self.has_embeddings():
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "embeddings": torch.Tensor(self.embeddings[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw_representations[idx,:].copy()).float()
            }
        else:
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw_representations[idx,:].copy()).float()
            }

# for eval only
class Full2D(Dataset):
    def __init__(self, side_length=200):
        self.side_length = side_length
        linspace = torch.linspace(-1, 1, side_length)
        self.coordinates = torch.cartesian_prod(linspace, linspace)
    
    def __len__(self):
        return self.side_length ** 2
    
    def __getitem__(self, idx):
        return {
            "idx": idx,
            "coordinates": self.coordinates[idx,:].float(),
        }


class ST3D(Dataset):
    def __init__(self, data_file, require_coordnorm=True, keep_ratio=True, **kwargs):
        super().__init__()

        adata = read_anndata(data_file)

        self.coordinates = adata.obsm["spatial"][:, :3]
        
        if sp.issparse(adata.X):
            self.raw_representations = adata.X.toarray()
        else:
            self.raw_representations = adata.X
        
        # take statistics
        self.n_cell = self.raw_representations.shape[0]
        self.n_gene = self.raw_representations.shape[1]
        self.n_channels = self.n_gene
        
        if "embeddings" in adata.obsm:
            self.embeddings = adata.obsm["embeddings"]
            assert self.raw_representations.shape[0] == self.embeddings.shape[0]
            self.n_embd = self.embeddings.shape[1]
        else: 
            self.embeddings = None

        if require_coordnorm:
            self._normalize_coordinates(keep_ratio=keep_ratio)

        # PCA: when number of features < 3, n_components cannot exceed number of features
        n_components_raw = min(3, self.n_gene)
        self.raw_pca = PCA(n_components=n_components_raw, random_state=0) # map raw representation dimension to 3 for visualization
        self.raw_pca.fit(self.raw_representations)

        if self.has_embeddings():
            n_components_embd = min(3, self.n_embd)
            self.embd_pca = PCA(n_components=n_components_embd, random_state=0) # map embedding dimension to 3 for visualization
            self.embd_pca.fit(self.embeddings)
        
    
    def has_embeddings(self):
        return False if self.embeddings is None else True

    def plot_raw_representations(self, spot_size=2, train_indices=None, val_indices=None):
        # PCA output may be < 3 dimensions, need to pad to 3 dimensions for visualization
        def _pad_to_3d(rep):
            if rep.shape[1] == 3:
                return rep
            elif rep.shape[1] == 2:
                return np.concatenate([rep, np.zeros((rep.shape[0], 1))], axis=1)
            elif rep.shape[1] == 1:
                return np.repeat(rep, 3, axis=1)
            else:
                return rep
        
        if train_indices and val_indices:
            train_rep = _pad_to_3d(self.raw_pca.transform(self.raw_representations[train_indices,:]))
            val_rep = _pad_to_3d(self.raw_pca.transform(self.raw_representations[val_indices,:]))
            train_fig = plot_ST(self.coordinates[train_indices,:], train_rep, spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], val_rep, spot_size)
            return train_fig, val_fig
        else:
            rep = _pad_to_3d(self.raw_pca.transform(self.raw_representations))
            fig = plot_ST(self.coordinates, rep, spot_size)
        return fig
    
    def plot_embeddings(self, spot_size=2, train_indices=None, val_indices=None):
        assert self.has_embeddings(), "The current adata file has NO embeddings!"
        # PCA output may be < 3 dimensions, need to pad to 3 dimensions for visualization
        def _pad_to_3d(rep):
            if rep.shape[1] == 3:
                return rep
            elif rep.shape[1] == 2:
                return np.concatenate([rep, np.zeros((rep.shape[0], 1))], axis=1)
            elif rep.shape[1] == 1:
                return np.repeat(rep, 3, axis=1)
            else:
                return rep
        
        if train_indices and val_indices:
            train_rep = _pad_to_3d(self.embd_pca.transform(self.embeddings[train_indices,:]))
            val_rep = _pad_to_3d(self.embd_pca.transform(self.embeddings[val_indices,:]))
            train_fig = plot_ST(self.coordinates[train_indices,:], train_rep, spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], val_rep, spot_size)
            return train_fig, val_fig
        else:
            rep = _pad_to_3d(self.embd_pca.transform(self.embeddings))
            fig = plot_ST(self.coordinates, rep, spot_size)
            return fig
    

    # normalize coordinates to [-1.0, +1.0]
    def _normalize_coordinates(self, keep_ratio):
        x_min, y_min, z_min = list(self.coordinates.min(axis=0))
        x_max, y_max, z_max = list(self.coordinates.max(axis=0))
        x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min

        self.coordinates[:,0] = (self.coordinates[:,0] - x_min) / x_range
        self.coordinates[:,1] = (self.coordinates[:,1] - y_min) / y_range
        self.coordinates[:,2] = (self.coordinates[:,2] - z_min) / z_range

        self.coordinates -= 0.5
        self.coordinates *= 2.0

        if keep_ratio: # may cause waste of space in the short side
            max_range = max(x_range, y_range, z_range)
            scale_x, scale_y, scale_z = x_range / max_range, y_range / max_range, z_range / max_range
            self.coordinates[:,0] *= scale_x
            self.coordinates[:,1] *= scale_y
            self.coordinates[:,2] *= scale_z
    
    def __len__(self):
        return self.n_cell
    
    def get_raw_dim(self):
        return self.n_gene
    
    def get_embd_dim(self):
        if self.has_embeddings():
            return self.n_embd
        else:
            return None
    
    def __getitem__(self, idx):
        if self.has_embeddings():
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "embeddings": torch.Tensor(self.embeddings[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw_representations[idx,:].copy()).float()
            }
        else:
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw_representations[idx,:].copy()).float()
            }

class HRSS(Dataset):
    def __init__(self, data_file, **kwargs):
        super().__init__()
        self.data = np.load(data_file)

        if "embeddings" in self.data.keys():
            self.embeddings = self.data["embeddings"]
            self.raw = self.data["raw"]
            self.coordinates = self.data["spatial"][:, :2]
            self.dataset_length = self.coordinates.shape[0]
            self.n_channels = self.raw.shape[-1]
            self.n_embd = self.embeddings.shape[-1]
            self.embd_pca = PCA(n_components=3, random_state=0)
            self.embd_pca.fit(self.embeddings)

        else:
            self.raw = self.data["raw"]
            
            # switch normalization strategy here
            #self._uint16_normalize()
            self._minmax_normalize()

            # custom slicing of self.raw
            self.raw = self.raw[:,:,::10] # every 10 channels

            self.height, self.width, self.n_channels = self.raw.shape
            self.dataset_length = self.height * self.width
            h_linspace = torch.linspace(-1, 1, self.height) # rows
            w_linspace = torch.linspace(-1, 1, self.width) # columns
            self.coordinates = torch.cartesian_prod(h_linspace, w_linspace)
            self.embeddings = None
            self.raw = rearrange(self.raw, "H W C -> (H W) C")

            

        # PCA: when number of features < 3, n_components cannot exceed number of features
        n_components_raw = min(3, self.raw.shape[1])
        self.raw_pca = PCA(n_components=n_components_raw, random_state=0)
        self.raw_pca.fit(self.raw)

    def has_embeddings(self):
        return False if self.embeddings is None else True
    
    def _uint16_normalize(self):
        self.raw = self.raw / 65535

    def _minmax_normalize(self):
        normalized_image = np.zeros_like(self.raw, dtype=float)

        for i in range(self.raw.shape[2]):
            channel = self.raw[:, :, i]
            min_val, max_val = np.min(channel), np.max(channel)
            
            if max_val != min_val:
                # normalized_image[:, :, i] = 2 * (channel - min_val) / (max_val - min_val) - 1
                normalized_image[:, :, i] = (channel - min_val) / (max_val - min_val)
            else:
                normalized_image[:, :, i] = 0
        self.raw = normalized_image
    
    def plot_raw_representations(self, train_indices=None, val_indices=None):
        # PCA output may be < 3 dimensions, need to pad to 3 dimensions for visualization
        def _pad_to_3d(rep):
            if rep.shape[1] == 3:
                return rep
            elif rep.shape[1] == 2:
                return np.concatenate([rep, np.zeros((rep.shape[0], 1))], axis=1)
            elif rep.shape[1] == 1:
                return np.repeat(rep, 3, axis=1)
            else:
                return rep
        
        rgb = _pad_to_3d(self.raw_pca.transform(self.raw))
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], rgb[train_indices,:], spot_size=1)
            val_fig = plot_ST(self.coordinates[val_indices,:], rgb[val_indices,:], spot_size=1)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, rgb)
            return fig
    
    def plot_embeddings(self, spot_size=2, train_indices=None, val_indices=None):
        assert self.has_embeddings(), "The current adata file has NO embeddings!"
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], self.embd_pca.transform(self.embeddings[train_indices,:]), spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], self.embd_pca.transform(self.embeddings[val_indices,:]), spot_size)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, self.embd_pca.transform(self.embeddings), spot_size)
            return fig
    
    def __len__(self):
        return self.dataset_length
    
    def get_raw_dim(self):
        return self.n_channels
    
    def get_embd_dim(self):
        if self.has_embeddings():
            return self.n_embd
        else:
            return None

    def __getitem__(self, idx):
        if self.has_embeddings():
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "embeddings": torch.Tensor(self.embeddings[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw[idx,:].copy()).float()
            }
        else:
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:]).float(),
                "raw_representations": torch.Tensor(self.raw[idx,:]).float()
            }

class GraphST2D(Dataset):
    def __init__(self, h5ad_file,neighbors,keep_ratio=True, **kwargs):
        super().__init__()
        #if h5ad_file is not str, adata = h5ad_file
        if type(h5ad_file) == str:

            adata = read_anndata(h5ad_file)
        else:
            adata = h5ad_file
        self.sb = True
        
        self.coordinates = adata.obsm["spatial"][:, :2]
        self.neighbors = neighbors
        if sp.issparse(adata.X):
            self.raw_representations = adata.X.toarray()
        else:
            self.raw_representations = adata.X
        
        # take statistics
        self.n_cell = self.raw_representations.shape[0]
        self.n_gene = self.raw_representations.shape[1]
        
        if "embeddings" in adata.obsm:
            self.embeddings = adata.obsm["embeddings"]
            assert self.raw_representations.shape[0] == self.embeddings.shape[0]
            self.n_embd = self.embeddings.shape[1]
        else: 
            self.embeddings = None

        self._normalize_coordinates(keep_ratio=keep_ratio)

        # PCA: when number of features < 3, n_components cannot exceed number of features
        n_components_raw = min(3, self.n_gene)
        self.raw_pca = PCA(n_components=n_components_raw, random_state=0) # map raw representation dimension to 3 for visualization
        self.raw_pca.fit(self.raw_representations)

        if self.has_embeddings():
            n_components_embd = min(3, self.n_embd)
            self.embd_pca = PCA(n_components=n_components_embd, random_state=0) # map embedding dimension to 3 for visualization
            self.embd_pca.fit(self.embeddings)
        
    
    def has_embeddings(self):
        return False if self.embeddings is None else True

    def plot_raw_representations(self, spot_size=2, train_indices=None, val_indices=None):
        # PCA output may be < 3 dimensions, need to pad to 3 dimensions for visualization
        def _pad_to_3d(rep):
            if rep.shape[1] == 3:
                return rep
            elif rep.shape[1] == 2:
                return np.concatenate([rep, np.zeros((rep.shape[0], 1))], axis=1)
            elif rep.shape[1] == 1:
                return np.repeat(rep, 3, axis=1)
            else:
                return rep
        
        if train_indices and val_indices:
            train_rep = _pad_to_3d(self.raw_pca.transform(self.raw_representations[train_indices,:]))
            val_rep = _pad_to_3d(self.raw_pca.transform(self.raw_representations[val_indices,:]))
            train_fig = plot_ST(self.coordinates[train_indices,:], train_rep, spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], val_rep, spot_size)
            return train_fig, val_fig
        else:
            rep = _pad_to_3d(self.raw_pca.transform(self.raw_representations))
            fig = plot_ST(self.coordinates, rep, spot_size)
        return fig
    
    def plot_embeddings(self, spot_size=2, train_indices=None, val_indices=None):
        assert self.has_embeddings(), "The current adata file has NO embeddings!"
        # PCA output may be < 3 dimensions, need to pad to 3 dimensions for visualization
        def _pad_to_3d(rep):
            if rep.shape[1] == 3:
                return rep
            elif rep.shape[1] == 2:
                return np.concatenate([rep, np.zeros((rep.shape[0], 1))], axis=1)
            elif rep.shape[1] == 1:
                return np.repeat(rep, 3, axis=1)
            else:
                return rep
        
        if train_indices and val_indices:
            train_rep = _pad_to_3d(self.embd_pca.transform(self.embeddings[train_indices,:]))
            val_rep = _pad_to_3d(self.embd_pca.transform(self.embeddings[val_indices,:]))
            train_fig = plot_ST(self.coordinates[train_indices,:], train_rep, spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], val_rep, spot_size)
            return train_fig, val_fig
        else:
            rep = _pad_to_3d(self.embd_pca.transform(self.embeddings))
            fig = plot_ST(self.coordinates, rep, spot_size)
            return fig
    

    # normalize coordinates to [-1.0, +1.0]
    def _normalize_coordinates(self, keep_ratio):
        x_min, y_min = list(self.coordinates.min(axis=0))
        x_max, y_max = list(self.coordinates.max(axis=0))
        self.coordinates = self.coordinates.astype(np.float64)
        x_range, y_range = x_max - x_min, y_max - y_min
        
        self.coordinates[:,0] = (self.coordinates[:,0] - x_min) / x_range
        
        self.coordinates[:,1] = (self.coordinates[:,1] - y_min) / y_range
     
        self.coordinates -= 0.5
        self.coordinates *= 2.0

        if keep_ratio: # may cause waste of space in the short side
            max_range = max(x_range, y_range)
            scale_x, scale_y = x_range / max_range, y_range / max_range
            self.coordinates[:,0] *= scale_x
            self.coordinates[:,1] *= scale_y
    
    def get_raw_dim(self):
        return self.n_gene
    
    def get_embd_dim(self):
        if self.has_embeddings():
            return self.n_embd
        else:
            return None

    def __len__(self):
        return self.n_cell
    
    def __getitem__(self, idx):
        if self.has_embeddings():
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "embeddings": torch.Tensor(self.embeddings[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw_representations[idx,:].copy()).float(),
                "neighbors": self.neighbors[idx]
            }
        else:
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw_representations[idx,:].copy()).float(),
                "neighbors": self.neighbors[idx]
            }

class GraphST3D(Dataset):
    def __init__(self, h5ad_file, neighbors, keep_ratio=True, preserve_z_scale=False, z_scale_factor=1.0, **kwargs):
        super().__init__()

        adata = read_anndata(h5ad_file) if isinstance(h5ad_file, str) else h5ad_file
        self.sb = True

        self.coordinates = adata.obsm["spatial"][:, :3]
        self.neighbors = neighbors

        self.raw_representations = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        self.n_cell, self.n_gene = self.raw_representations.shape

        if "embeddings" in adata.obsm:
            self.embeddings = adata.obsm["embeddings"]
            assert self.embeddings.shape[0] == self.n_cell
            self.n_embd = self.embeddings.shape[1]
        else:
            self.embeddings = None

        self._normalize_coordinates(keep_ratio=keep_ratio, preserve_z_scale=preserve_z_scale, z_scale_factor=z_scale_factor)

        # PCA: when number of features < 3, n_components cannot exceed number of features
        n_components_raw = min(3, self.n_gene)
        self.raw_pca = PCA(n_components=n_components_raw, random_state=0)
        self.raw_pca.fit(self.raw_representations)

        if self.has_embeddings():
            n_components_embd = min(3, self.n_embd)
            self.embd_pca = PCA(n_components=n_components_embd, random_state=0)
            self.embd_pca.fit(self.embeddings)

    def has_embeddings(self):
        return self.embeddings is not None

    def _normalize_coordinates(self, keep_ratio, preserve_z_scale=False, z_scale_factor=1.0):
        """
        Normalize coordinates
        
        Args:
            keep_ratio: Whether to maintain aspect ratio
            preserve_z_scale: Whether to preserve original z-direction scale (for sparse z-direction)
            z_scale_factor: Scaling factor for z-direction (>1 means amplify z-direction importance)
        """
        x_min, y_min, z_min = self.coordinates.min(axis=0)
        x_max, y_max, z_max = self.coordinates.max(axis=0)
        x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min

        self.coordinates = self.coordinates.astype(np.float64)
        self.coordinates[:, 0] = (self.coordinates[:, 0] - x_min) / x_range
        self.coordinates[:, 1] = (self.coordinates[:, 1] - y_min) / y_range
        self.coordinates[:, 2] = (self.coordinates[:, 2] - z_min) / z_range

        self.coordinates -= 0.5
        self.coordinates *= 2.0

        if keep_ratio:
            if preserve_z_scale:
                # Don't compress z-direction, let z maintain a larger range
                max_xy_range = max(x_range, y_range)
                scale_x = x_range / max_xy_range
                scale_y = y_range / max_xy_range
                # z-direction uses original range and can be further amplified
                scale_z = z_scale_factor  # Don't shrink z-direction, or amplify it
            else:
                max_range = max(x_range, y_range, z_range)
                scale_x = x_range / max_range
                scale_y = y_range / max_range
                scale_z = z_range / max_range
            
            self.coordinates[:, 0] *= scale_x
            self.coordinates[:, 1] *= scale_y
            self.coordinates[:, 2] *= scale_z

    def get_raw_dim(self):
        return self.n_gene

    def get_embd_dim(self):
        return self.n_embd if self.has_embeddings() else None

    def __len__(self):
        return self.n_cell

    def __getitem__(self, idx):
        sample = {
            "idx": idx,
            "coordinates": torch.tensor(self.coordinates[idx], dtype=torch.float),
            "raw_representations": torch.tensor(self.raw_representations[idx], dtype=torch.float),
            "neighbors": self.neighbors[idx]
        }
        if self.has_embeddings():
            sample["embeddings"] = torch.tensor(self.embeddings[idx], dtype=torch.float)
        return sample

    def plot_raw_representations(self, spot_size=2, train_indices=None, val_indices=None):
        coords = self.coordinates
        # PCA output may be < 3 dimensions, need to pad to 3 dimensions for visualization
        def _pad_to_3d(rep):
            if rep.shape[1] == 3:
                return rep
            elif rep.shape[1] == 2:
                # Pad third dimension with 0
                return np.concatenate([rep, np.zeros((rep.shape[0], 1))], axis=1)
            elif rep.shape[1] == 1:
                # Replicate to 3 dimensions
                return np.repeat(rep, 3, axis=1)
            else:
                return rep
        
        if train_indices is not None and val_indices is not None:
            train_rep = _pad_to_3d(self.raw_pca.transform(self.raw_representations[train_indices]))
            val_rep = _pad_to_3d(self.raw_pca.transform(self.raw_representations[val_indices]))
            train_fig = plot_ST(coords[train_indices], train_rep, spot_size)
            val_fig = plot_ST(coords[val_indices], val_rep, spot_size)
            return train_fig, val_fig
        else:
            rep = _pad_to_3d(self.raw_pca.transform(self.raw_representations))
            return plot_ST(coords, rep, spot_size)

    def plot_embeddings(self, spot_size=2, train_indices=None, val_indices=None):
        assert self.has_embeddings(), "No embeddings available!"
        coords = self.coordinates
        # PCA output may be < 3 dimensions, need to pad to 3 dimensions for visualization
        def _pad_to_3d(rep):
            if rep.shape[1] == 3:
                return rep
            elif rep.shape[1] == 2:
                # Pad third dimension with 0
                return np.concatenate([rep, np.zeros((rep.shape[0], 1))], axis=1)
            elif rep.shape[1] == 1:
                # Replicate to 3 dimensions
                return np.repeat(rep, 3, axis=1)
            else:
                return rep
        
        if train_indices is not None and val_indices is not None:
            train_rep = _pad_to_3d(self.embd_pca.transform(self.embeddings[train_indices]))
            val_rep = _pad_to_3d(self.embd_pca.transform(self.embeddings[val_indices]))
            train_fig = plot_ST(coords[train_indices], train_rep, spot_size)
            val_fig = plot_ST(coords[val_indices], val_rep, spot_size)
            return train_fig, val_fig
        else:
            rep = _pad_to_3d(self.embd_pca.transform(self.embeddings))
            return plot_ST(coords, rep, spot_size)


if __name__ == "__main__":

    ds = ST3D("./data/preprocessed_data/E9_two_slices.h5ad", True, True)
    raw = ds.raw_representations
    print(raw.max(), raw.min())
    sparsity = (raw==0).sum() / (raw>=0).sum()
    print(sparsity)
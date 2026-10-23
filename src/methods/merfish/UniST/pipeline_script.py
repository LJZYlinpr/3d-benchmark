"""Verbatim SUICA pipeline template used by the published UniST runs."""

PIPELINE_SCRIPT = r'''
import sys, os, warnings, shutil, json, numpy as np, torch
import anndata as ad
from pathlib import Path
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")
os.chdir("{SUICA_DIR}")
sys.path.insert(0, "{SUICA_DIR}")

# GPU checkpoint-loading monkey-patch (PyTorch 2.5 strict device checking)
_orig_load = torch.load
def _gpu_load(*a, **kw):
    kw['map_location'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return _orig_load(*a, **kw)
torch.load = _gpu_load

visible_data = "{visible_data}"
query_coords_file = "{query_coords_file}"
output_dir = Path("{output_dir}")
gae_epochs = {gae_epochs}
inr_epochs = {inr_epochs}

# ============ Stage 1: GAE embedder (official config, visible data only) ============
print("\\n=== Stage 1: GAE embedder ===", flush=True)
logs_gae = output_dir / "logs_gae"
if logs_gae.exists():
    shutil.rmtree(str(logs_gae))

gae_config = OmegaConf.load("configs/ST/embedder_gae_3d_sparse.yaml")
gae_config.dataset.data_file = visible_data
gae_config.pipeline.optimization.epochs = gae_epochs
gae_config.pipeline.optimization.logs = str(logs_gae)
gae_config.pipeline.optimization.device = "cuda" if torch.cuda.is_available() else "cpu"

from systems import train_embedder
train_embedder(gae_config)

embedded_h5ad = Path(str(logs_gae / "lightning_logs" / "version_0" / "embedded-all.h5ad"))
if not embedded_h5ad.exists():
    embedded_h5ad = sorted(Path(str(logs_gae)).rglob("embedded-all.h5ad"))[-1]
print(f"embedded h5ad: {{embedded_h5ad}}")
gae_ckpts = sorted(Path(str(logs_gae)).rglob("*.ckpt"))
assert gae_ckpts, "no GAE checkpoint found"
gae_ckpt = str(gae_ckpts[-1])
print(f"GAE checkpoint: {{gae_ckpt}}")

# ============ Stage 2: INR (coords -> embeddings) + decoder (official config) ============
print("\\n=== Stage 2: INR + decoder ===", flush=True)
logs_inr = output_dir / "logs_inr"
if logs_inr.exists():
    shutil.rmtree(str(logs_inr))

inr_config = OmegaConf.load("configs/ST/inr_embd_3d_sparse.yaml")
inr_config.dataset.data_file = str(embedded_h5ad)
inr_config.pipeline.target = "embeddings"
inr_config.pipeline.inr.decoder.ckpt = gae_ckpt
inr_config.pipeline.optimization.epochs = inr_epochs
inr_config.pipeline.optimization.logs = str(logs_inr)
inr_config.pipeline.optimization.device = "cuda" if torch.cuda.is_available() else "cpu"

from systems import train_inr
train_inr(inr_config)

inr_ckpts = sorted(Path(str(logs_inr)).rglob("*.ckpt"))
assert inr_ckpts, "no INR checkpoint found"
inr_ckpt = str(inr_ckpts[-1])
print(f"INR checkpoint: {{inr_ckpt}}")

# ============ Stage 3: prediction at query coordinates ============
print("\\n=== Stage 3: prediction at query coordinates ===", flush=True)
query_raw = np.load(query_coords_file)

# Normalize query coords EXACTLY like GraphST3D._normalize_coordinates
# (fit on the visible data, the same data the embedder saw)
adata_vis = ad.read_h5ad(visible_data)
coords_ref = adata_vis.obsm["spatial"][:, :3].astype(np.float64)
x_min, y_min, z_min = coords_ref.min(axis=0)
x_max, y_max, z_max = coords_ref.max(axis=0)
x_range, y_range, z_range = (x_max - x_min), (y_max - y_min), (z_max - z_min)
c = query_raw.copy().astype(np.float64)
c[:, 0] = (c[:, 0] - x_min) / x_range
c[:, 1] = (c[:, 1] - y_min) / y_range
c[:, 2] = (c[:, 2] - z_min) / z_range
c -= 0.5
c *= 2.0
# keep_ratio=True + preserve_z_scale=True + z_scale_factor=1.5 (official config)
max_xy_range = max(x_range, y_range)
c[:, 0] *= x_range / max_xy_range
c[:, 1] *= y_range / max_xy_range
c[:, 2] *= 1.5
np.save(output_dir / "query_coords_norm.npy", c.astype(np.float32))
print(f"normalized query coords: {{c.shape}}, z range [{{c[:,2].min():.3f}}, {{c[:,2].max():.3f}}]")

logs_pred = output_dir / "logs_pred"
if logs_pred.exists():
    shutil.rmtree(str(logs_pred))

pred_config = OmegaConf.load("configs/ST/inr_pred_3d_sparse.yaml")
pred_config.dataset.data_file = str(embedded_h5ad)
pred_config.pipeline.target = "embeddings"
pred_config.pipeline.custom_coords_file = str(output_dir / "query_coords_norm.npy")
pred_config.pipeline.predict_mode = "custom"
pred_config.pipeline.prediction.ckpt = inr_ckpt
pred_config.pipeline.inr.decoder.ckpt = gae_ckpt
pred_config.pipeline.optimization.logs = str(logs_pred)
pred_config.pipeline.optimization.device = "cuda" if torch.cuda.is_available() else "cpu"

from systems import predict_inr
predict_inr(pred_config)

pred_logs = sorted(Path(str(logs_pred)).rglob("reconstructed-*.h5ad"))
assert pred_logs, "No prediction output found"
result = ad.read_h5ad(pred_logs[-1])
X_pred = result.obsm.get("reconstructed_raw", result.obsm.get("fitted_raw"))
if hasattr(X_pred, 'toarray'):
    X_pred = X_pred.toarray()
X_pred = np.clip(np.asarray(X_pred), 0, None)
pred_coords = np.asarray(result.obsm["spatial"], dtype=np.float64)

assert pred_coords.shape[0] == c.shape[0], \
    f"pred count {{pred_coords.shape[0]}} != query count {{c.shape[0]}}"
assert np.allclose(pred_coords, c, atol=1e-4), \
    "PREDICTION COORDS != QUERY COORDS — custom query mechanism NOT effective"
np.save(output_dir / "X_pred.npy", X_pred)
np.save(output_dir / "pred_coords_norm.npy", pred_coords)
print(f"Saved X_pred: {{X_pred.shape}} — VERIFIED pred coords == query coords")
print("DONE!")
'''

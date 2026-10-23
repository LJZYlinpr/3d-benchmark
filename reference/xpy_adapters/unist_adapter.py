"""
UniST/SUICA adapter for MERFISH LOSO_z14 task.

    ┌──────────────────────────────────────────────────────────────────────┐
    │ NOT THE PATH THAT PRODUCED THE PUBLISHED UniST RESULTS.              │
    │                                                                      │
    │ This adapter has NO CALLER anywhere in the repository (checked       │
    │ 2026-09-20: the only occurrences are its own class definition, plus  │
    │ doc/inventory/notebook listings -- no code imports it). UniST's      │
    │ benchmark numbers come from `benchmark/scripts/rerun_unist_official.py`,│
    │ which runs the authors' official configuration (GAE 1000 + INR 2000  │
    │ epochs).                                                             │
    │                                                                      │
    │ Wiring this adapter up instead would run a REDUCED configuration:    │
    │ the generated script below sets `epochs = 150  # Reduced for          │
    │ benchmark`, which is non-compliant under EXPERIMENT_RESOURCE_POLICY   │
    │ §1 and would exclude the result from every comparison table.         │
    │                                                                      │
    │ Kept rather than deleted because it is referenced by dated audit      │
    │ inventories; deleting it is a decision for the repo owner (see        │
    │ DECISIONS_PENDING.md D7). Until then: do not call it, and if you ever │
    │ do, fix the epoch count first.                                       │
    └──────────────────────────────────────────────────────────────────────┘

UniST has three modules: upsampling (RepKPU), interpolation (InterpolAI),
gene imputation (SUICA_pro). This adapter uses ONLY the SUICA module for
expression prediction at query coordinates.

SUICA pipeline (two-stage, must be retrained per task to avoid data leakage):
1. GAE embedder: graph autoencoder on spatial KNN → 64-dim latent embeddings
2. FFN (INR): Fourier-feature network maps coordinates → embeddings → expression

The shipped pre-trained model was trained on all slices of a MERFISH-like
dataset (28,317 spots). For LOSO_z14, we MUST retrain on visible slices only
to prevent held-out data leakage.

Uses subprocess + unist conda environment isolation.
"""

import sys, time, json, os
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import anndata as ad
from scipy import sparse

from benchmark.src.adapters.base import (
    assigned_gpu_indices, pick_from_assigned_or_free,
    BaseAdapter, AdapterResult, AdapterError, FailureType,
)

UNIST_DIR = Path("/storage/yuhongtaoLab/linpeiran/xpy/lab/baseline/methods/UniST")
SUICA_DIR = UNIST_DIR / "external" / "SUICA_pro"
UNIST_ENV = "unist"
MERFISH_DIR = Path("/storage/yuhongtaoLab/linpeiran/xpy/data/data/spatialz/merfish")


class UniSTAdapter(BaseAdapter):
    """Adapter for UniST SUICA gene expression imputation."""

    method_name = "UniST"
    conda_env = UNIST_ENV
    method_dir = UNIST_DIR

    def run_impl(
        self, adata_input: ad.AnnData, config: Dict, output_dir: Path
    ) -> ad.AnnData:
        holdout_slices = config.get("holdout_slices", None)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract visible slices and holdout coords
        X_all = adata_input.X.toarray() if sparse.issparse(adata_input.X) else np.asarray(adata_input.X)
        coords_all = adata_input.obsm["spatial"]
        genes = adata_input.var_names.tolist()

        if holdout_slices is not None:
            holdout_mask = adata_input.obs["slice"].isin(holdout_slices).to_numpy()
        else:
            holdout_z = config.get("holdout_z", 14)
            z_all = coords_all[:, 2]
            holdout_mask = np.isclose(z_all, float(holdout_z))
        visible_mask = ~holdout_mask

        X_vis = X_all[visible_mask]
        coords_vis = coords_all[visible_mask]
        coords_holdout = coords_all[holdout_mask]

        # Save visible data as SUICA-compatible h5ad
        vis_path = output_dir / "visible_data.h5ad"
        ad.AnnData(
            X=X_vis,
            obsm={"spatial": coords_vis},
            var=adata_input.var.copy(),
        ).write_h5ad(vis_path)

        # Save holdout coords as query (RAW coordinates — same space as INR training
        # because SUICA train config uses require_coordnorm: False)
        query_coords_path = output_dir / "query_coords.npy"
        np.save(query_coords_path, coords_holdout)

        # GPU selection: explicit config value or auto-pick least-loaded GPU
        gpu = config.get("gpu")
        if gpu is None:
            gpu = self._pick_free_gpu()
        self._gpu = int(gpu)

        # Generate and run SUICA pipeline script
        script = self._generate_script(
            visible_data_path=str(vis_path),
            query_coords_path=str(query_coords_path),
            output_dir=str(output_dir),
            n_genes=len(genes),
            genes=genes,
        )

        script_path = output_dir / "_run_suica.py"
        script_path.write_text(script)

        t0 = time.time()
        import subprocess
        result = subprocess.run(
            ["conda", "run", "-n", self.conda_env, "python", str(script_path)],
            cwd=str(SUICA_DIR),  # Must run from SUICA_dir for relative imports
            capture_output=True, text=True, timeout=28800,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(self._gpu)},
        )
        elapsed = time.time() - t0

        if self.verbose:
            print(result.stdout[-3000:])
            if result.stderr:
                print(result.stderr[-1000:])

        if result.returncode != 0:
            raise AdapterError(
                f"UniST/SUICA failed: {result.stderr[-500:]}",
                failure_type=FailureType.F2_INTERNAL,
                details={"stdout": result.stdout[-2000:], "stderr": result.stderr},
            )

        # Load predictions — try X_pred.npy first (saved by generated script)
        x_pred_npy = output_dir / "X_pred.npy"
        pred_coords_npy = output_dir / "pred_coords.npy"
        if x_pred_npy.exists() and pred_coords_npy.exists():
            X_pred = np.load(x_pred_npy)
            pred_coords = np.load(pred_coords_npy)
            print(f"Loaded X_pred from {x_pred_npy}: {X_pred.shape}")
        else:
            # Fallback: search for reconstructed h5ad files
            pred_files = sorted(output_dir.rglob("reconstructed-*.h5ad"))
            if not pred_files:
                raise AdapterError(
                    f"No predictions found in {output_dir}",
                    failure_type=FailureType.F2_INTERNAL,
                )
            pred_adata = ad.read_h5ad(pred_files[0])
            X_pred = pred_adata.obsm.get("fitted_raw", pred_adata.obsm.get("reconstructed_raw", pred_adata.X))
            pred_coords = pred_adata.obsm.get("spatial")
            if sparse.issparse(X_pred):
                X_pred = X_pred.toarray()
            X_pred = np.asarray(X_pred)
            pred_coords = np.asarray(pred_coords)

        # --- VERIFY query -> prediction coordinate correspondence (core audit fix) ---
        query_coords = np.load(query_coords_path)
        pred_coords = np.asarray(pred_coords, dtype=np.float64)
        if pred_coords.shape[0] != query_coords.shape[0]:
            raise AdapterError(
                f"Prediction count {pred_coords.shape[0]} != query count {query_coords.shape[0]} "
                f"— query mechanism not effective",
                failure_type=FailureType.F2_INTERNAL,
            )
        if pred_coords.shape[1] != 3 or not np.allclose(pred_coords, query_coords, atol=1e-4):
            raise AdapterError(
                "Predicted coordinates do not match query coordinates — "
                "predictions were not made at the query locations",
                failure_type=FailureType.F2_INTERNAL,
            )
        print(f"VERIFIED: {len(pred_coords)} predictions at exact query coordinates")

        X_pred = np.clip(np.asarray(X_pred), 0, None)

        # Build output — rows correspond 1:1 to query (holdout) coords in order
        out_adata = ad.AnnData(
            X=X_pred,
            obsm={"spatial": pred_coords},
            var=adata_input.var.copy(),
        )
        out_adata.obs["is_interpolated"] = True
        out_adata.obs["slice"] = adata_input.obs["slice"][holdout_mask].values[: len(X_pred)]
        out_adata.uns["method"] = self.method_name
        out_adata.uns["task"] = "MERFISH_LOSO"
        out_adata.uns["runtime_sec"] = elapsed
        out_adata.uns["query_coord_correspondence"] = "verified_exact"

        return out_adata

    @staticmethod
    def _pick_free_gpu() -> int:
        """Auto-select the GPU with the most free memory."""
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        free = [(int(l.split(",")[0]), int(l.split(",")[1]))
                for l in out.stdout.strip().splitlines() if l.strip()]
        # An explicit CUDA_VISIBLE_DEVICES assignment wins over "freest card": see
        # assigned_gpu_indices() in base.py for the incident this prevents.
        return pick_from_assigned_or_free(free)

    def _generate_script(self, **kw) -> str:
        return f'''
import sys, os, warnings, time, json, shutil, numpy as np, torch
import anndata as ad
from pathlib import Path
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")
os.chdir("{SUICA_DIR}")
sys.path.insert(0, "{SUICA_DIR}")

visible_data = "{kw['visible_data_path']}"
query_coords_file = "{kw['query_coords_path']}"
output_dir = Path("{kw['output_dir']}")
n_genes = {kw['n_genes']}
genes = {json.dumps(kw['genes'])}

# Monkey-patch torch.load to always map to GPU (fix pretrained embedder device mismatch)
_orig_load = torch.load
def _gpu_load(*a, **kw):
    kw['map_location'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return _orig_load(*a, **kw)
torch.load = _gpu_load

# Fix DataLoader num_workers to avoid multi-process deadlock
import torch.utils.data
_orig_dl_init = torch.utils.data.DataLoader.__init__
def _patched_dl_init(self, *args, **kwargs):
    if 'num_workers' in kwargs and kwargs['num_workers'] > 0:
        kwargs['num_workers'] = 0
    elif len(args) > 5 and args[5] > 0:
        args = list(args); args[5] = 0
    _orig_dl_init(self, *args, **kwargs)
torch.utils.data.DataLoader.__init__ = _patched_dl_init

print("Loading visible data...")
adata_vis = ad.read_h5ad(visible_data)
print(f"Visible: {{adata_vis.n_obs}} spots x {{adata_vis.n_vars}} genes")

# === Train FFN/INR ===
# Predict raw expression directly (target=raw_representations, no decoder needed)
print("\\n=== Training FFN/INR ===")
# Clear previous logs to avoid checkpoint conflicts
import shutil
logs_inr_dir = output_dir / "logs_inr"
if logs_inr_dir.exists():
    shutil.rmtree(str(logs_inr_dir))

inr_config = OmegaConf.load("configs/ST/inr_embd_3d_sparse.yaml")
inr_config.dataset.data_file = visible_data
inr_config.pipeline.target = "raw_representations"  # Predict expression directly
inr_config.pipeline.inr.decoder = None  # No decoder needed for raw expression
inr_config.pipeline.optimization.logs = str(logs_inr_dir)
inr_config.pipeline.optimization.epochs = 150  # Reduced for benchmark
inr_config.pipeline.optimization.device = "cuda" if __import__('torch').cuda.is_available() else "cpu"

from systems import train_inr, predict_inr
try:
    train_inr(inr_config)
    print("INR training complete")
except Exception as e:
    print(f"INR training failed: {{e}}")
    sys.exit(1)

# Find trained checkpoint
logs_dir = Path(str(output_dir / "logs_inr"))
ckpt_files = sorted(logs_dir.rglob("*.ckpt"))
if not ckpt_files:
    print("ERROR: No checkpoint found after training")
    sys.exit(1)
inr_ckpt = str(ckpt_files[-1])
print(f"Using checkpoint: {{inr_ckpt}}")

# === Prediction at query coordinates ===
# AUDIT FIX 2026-08-13:
#  - custom_coords_file must be set under pipeline (SUICA reads pipeline.custom_coords_file)
#  - predict_mode must be "custom" (explicit)
#  - query coords are RAW: INR trains on raw coords (require_coordnorm: False in train config),
#    so query coords must be in the same raw space — do NOT normalize to [-1,1]
print("\\n=== Prediction at query coordinates ===")
query_raw = np.load(query_coords_file)
print(f"Query coords (raw): {{query_raw.shape}}, z range [{{query_raw[:,2].min():.2f}}, {{query_raw[:,2].max():.2f}}]")

# Clear previous prediction logs
logs_pred_dir = output_dir / "logs_pred"
if logs_pred_dir.exists():
    shutil.rmtree(str(logs_pred_dir))

pred_config = OmegaConf.load("configs/ST/inr_pred_3d_sparse.yaml")
pred_config.dataset.data_file = visible_data  # dataset used for dim inference only
pred_config.pipeline.target = "raw_representations"
pred_config.pipeline.custom_coords_file = query_coords_file  # RAW query coords (training space)
pred_config.pipeline.predict_mode = "custom"
pred_config.pipeline.prediction.ckpt = inr_ckpt  # KEY: prediction.ckpt, not inr.ckpt
pred_config.pipeline.inr.decoder = None
pred_config.pipeline.optimization.logs = str(logs_pred_dir)
pred_config.pipeline.optimization.device = "cuda" if __import__('torch').cuda.is_available() else "cpu"

predict_inr(pred_config)

# Find output and VERIFY the prediction coordinates equal the query coordinates
pred_logs = sorted(Path(str(output_dir / "logs_pred")).rglob("reconstructed-*.h5ad"))
if pred_logs:
    result = ad.read_h5ad(pred_logs[-1])
    X_pred = result.obsm.get("fitted_raw", result.obsm.get("reconstructed_raw", result.X))
    if hasattr(X_pred, 'toarray'): X_pred = X_pred.toarray()
    X_pred = np.clip(np.asarray(X_pred), 0, None)
    pred_coords = np.asarray(result.obsm["spatial"], dtype=np.float64)
    assert pred_coords.shape[0] == query_raw.shape[0], \\
        f"pred count {{pred_coords.shape[0]}} != query count {{query_raw.shape[0]}}"
    assert np.allclose(pred_coords, query_raw, atol=1e-4), \\
        "PREDICTION COORDS != QUERY COORDS — custom query mechanism NOT effective"
    np.save(output_dir / "X_pred.npy", X_pred)
    np.save(output_dir / "pred_coords.npy", pred_coords)
    print(f"Saved X_pred: {{X_pred.shape}} — VERIFIED prediction coords == query coords")
else:
    print("ERROR: No prediction output found")
    sys.exit(1)

print("DONE!")
'''

"""
SINTER3D adapter for MERFISH LOSO_z14 task.

SINTER3D learns a continuous INR mapping from 3D coordinates to gene expression,
supporting query at arbitrary coordinates. This adapter:
1. Converts MERFISH slices to SINTER3D's expected format (counts, library_size, 3D_coor)
2. Trains on visible slices (z=4,9,19,24)
3. Predicts expression at held-out z=14 coordinates
4. Returns standardized output AnnData

Uses subprocess + SINTER3D_env conda isolation.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Optional
import tempfile

import numpy as np
import anndata as ad
from scipy import sparse

from benchmark.src.adapters.base import (
    BaseAdapter, AdapterResult, AdapterError, FailureType,
)

# Paths
SINTER3D_DIR = Path("/storage/yuhongtaoLab/linpeiran/xpy/lab/baseline/methods/SINTER3D/SINTER3D-master")
SINTER3D_ENV = "SINTER3D_env"
SINTER3D_PYTHON = "/storage/yuhongtaoLab/linpeiran/env/miniconda/envs/SINTER3D_env/bin/python"
MERFISH_DATA_DIR = Path("/storage/yuhongtaoLab/linpeiran/xpy/data/data/spatialz/merfish")


class SINTER3DAdapter(BaseAdapter):
    """Adapter for SINTER3D implicit neural representation method."""

    method_name = "SINTER3D"
    conda_env = SINTER3D_ENV
    method_dir = SINTER3D_DIR

    def run_impl(
        self,
        adata_input: ad.AnnData,
        config: Dict,
        output_dir: Path,
    ) -> ad.AnnData:
        """Run SINTER3D on MERFISH LOSO task.

        Args:
            adata_input: Task AnnData with all slices, obsm['spatial'] (3D coords),
                        obs['slice'] for slice labels.
            config: May contain:
                - holdout_z: target z for held-out slice
                - training_steps: override default training steps
                - hidden_dims: model hidden dimensions
            output_dir: Output directory.

        Returns:
            Output AnnData with predicted expression.
        """
        holdout_slices = config.get("holdout_slices", None)
        if holdout_slices is None:
            # Backward compat: single holdout_z
            holdout_z = config.get("holdout_z", 14)
            z_coords = adata_input.obsm["spatial"][:, 2]
            holdout_mask = np.isclose(z_coords, float(holdout_z))
        else:
            holdout_mask = adata_input.obs["slice"].isin(holdout_slices).to_numpy()
        training_steps = config.get("training_steps", 3000)
        hidden_dims = config.get("hidden_dims", [512, 128])
        max_points_per_slice = config.get("max_points_per_slice", None)  # subsample if set
        timeout_sec = config.get("timeout_sec", 7200)  # T_max guard (2026-08-20 campaign)

        visible_mask = ~holdout_mask

        # Save visible slices as individual h5ad files
        visible_dir = output_dir / "visible_slices"
        visible_dir.mkdir(parents=True, exist_ok=True)
        slice_files = []
        visible_z = []

        visible_slice_labels = sorted(adata_input.obs["slice"][visible_mask].unique())
        for sl_label in visible_slice_labels:
            sl_mask = adata_input.obs["slice"] == sl_label
            a_slice = adata_input[sl_mask.to_numpy()].copy()
            # Subsample if needed to control memory
            if max_points_per_slice is not None and a_slice.n_obs > max_points_per_slice:
                rng = np.random.RandomState(42)
                idx = rng.choice(a_slice.n_obs, max_points_per_slice, replace=False)
                a_slice = a_slice[idx].copy()
            fname = visible_dir / f"{sl_label}.h5ad"
            a_slice.write_h5ad(fname)
            slice_files.append(str(fname))
            # Track Z value for slice distance computation
            z_val = float(a_slice.obsm["spatial"][0, 2])
            visible_z.append(z_val)

        # Save holdout coordinates for prediction query
        X_holdout = adata_input.X[holdout_mask]
        if sparse.issparse(X_holdout):
            X_holdout = X_holdout.toarray()
        coords_holdout = adata_input.obsm["spatial"][holdout_mask]

        # Save holdout coords
        query_coords_path = output_dir / "query_coords.npy"
        np.save(query_coords_path, coords_holdout)

        # Save ground truth for evaluation
        gt_path = output_dir / "ground_truth.npy"
        np.save(gt_path, X_holdout)

        # Generate adapter script
        script = self._generate_script(
            slice_files=slice_files,
            visible_z=visible_z,
            visible_slice_labels=visible_slice_labels,
            query_coords_path=str(query_coords_path),
            gt_path=str(gt_path),
            output_dir=str(output_dir),
            training_steps=training_steps,
            hidden_dims=hidden_dims,
        )

        # Write and execute script in SINTER3D conda env
        script_path = output_dir / "_run_sinter3d.py"
        script_path.write_text(script)

        t0 = time.time()

        import subprocess
        result = subprocess.run(
            [SINTER3D_PYTHON, str(script_path)],
            cwd=str(output_dir),
            capture_output=True,
            text=True,
            timeout=timeout_sec,  # T_max guard from run config (2026-08-20 campaign)
            env={**__import__('os').environ, "PYTHONPATH": str(SINTER3D_DIR)},
        )

        elapsed = time.time() - t0

        if self.verbose:
            print(f"SINTER3D stdout:\n{result.stdout[-2000:]}")
            if result.stderr:
                print(f"SINTER3D stderr:\n{result.stderr[-1000:]}")

        if result.returncode != 0:
            raise AdapterError(
                f"SINTER3D failed (code {result.returncode}): {result.stderr[-500:]}",
                failure_type=FailureType.F2_INTERNAL,
                details={"stdout": result.stdout, "stderr": result.stderr},
            )

        # Load predictions
        pred_path = Path(output_dir) / "X_pred.npy"
        if not pred_path.exists():
            raise AdapterError(
                f"SINTER3D did not produce predictions at {pred_path}",
                failure_type=FailureType.F2_INTERNAL,
            )

        X_pred = np.load(pred_path)

        # Build output AnnData
        pred_adata = ad.AnnData(
            X=X_pred,
            obsm={"spatial": coords_holdout},
            var=adata_input.var.copy(),
        )
        pred_adata.obs["is_interpolated"] = True
        pred_adata.obs["slice"] = adata_input.obs["slice"][holdout_mask].values
        pred_adata.uns["method"] = self.method_name
        pred_adata.uns["task"] = "MERFISH_LOSO"
        pred_adata.uns["runtime_sec"] = elapsed
        pred_adata.uns["run_config"] = config

        return pred_adata

    def _generate_script(
        self,
        slice_files,
        visible_z,
        visible_slice_labels,
        query_coords_path,
        gt_path,
        output_dir,
        training_steps,
        hidden_dims,
    ) -> str:
        """Generate the Python script to run inside SINTER3D conda env."""
        return f'''
import sys, json, time, warnings, os
import numpy as np
import anndata as ad
from scipy import sparse
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'

sys.path.insert(0, "{SINTER3D_DIR}")

warnings.filterwarnings("ignore")

def main():
    print("Loading MERFISH slices...")
    slice_files = {json.dumps(slice_files)}
    visible_z = {json.dumps(visible_z)}
    visible_slice_labels = {json.dumps([str(s) for s in visible_slice_labels])}

    # Load slices and prepare for SINTER3D
    slices = []
    for i, f in enumerate(slice_files):
        a = ad.read_h5ad(f)
        if sparse.issparse(a.X):
            a.X = a.X.toarray()
        # Ensure X is non-negative
        a.X = np.clip(a.X, 0, None)
        # Use 3D spatial coordinates directly (x, y, z)
        z_val = visible_z[i]
        if a.obsm['spatial'].shape[1] >= 3:
            a.obsm['3D_coor'] = a.obsm['spatial'].copy()
        else:
            # Fallback for 2D spatial: construct [x, z, y]
            coords_2d = a.obsm['spatial']
            a.obsm['3D_coor'] = np.column_stack([
                coords_2d[:, 0],
                np.full(a.n_obs, z_val),
                coords_2d[:, 1],
            ])
        # SINTER3D expects these columns. The slice id is an ORDINAL, which is what the
        # official pipeline itself writes (utils.py:84 `adata_st_new.obs['slice'] = i`),
        # and it has to be numeric: model_basic.py:114 does
        #   obs["slice"] = obs["slice"].map(slice_remap).astype(int)
        # so a task slice label like 'Zhuang-ABCA-2.002' raises
        #   ValueError: invalid literal for int() with base 10
        #
        # 2026-09-23, replacing `str(int(z_val))`, which was injective on neither dataset:
        # on the reference 54 slices collapse to 41 distinct integers, and on the Allen
        # releases (z spanning ~0.1-13.5, ~200 um apart) about ten physical slices share
        # one id. The ordinal is injective by construction and keeps the depth order,
        # because `slice_files` is built from `sorted(visible_slice_labels)`.
        #
        # Traced before changing it, because the value is not obviously inert: the model
        # reads obs['slice'] in exactly one place (model_basic/spatial/complex.py, the
        # `use_type == 'evaluate'` branch) and uses it ONLY to compose unique obs indices
        # (index + "-slice" + value); the training branch uses the enumerate index instead.
        # So the old collision changed no arithmetic -- but it did misdescribe the input.
        a.obs['slice'] = i
        # Use expression directly as pseudo-counts for count-based model
        a.obsm['count'] = a.X.copy()
        a.obs['library_size'] = np.sum(a.X, axis=1)
        # Cell type column
        if 'cell_class' in a.obs.columns:
            a.obs['cell_type'] = a.obs['cell_class']
        elif 'cell_type' not in a.obs.columns:
            a.obs['cell_type'] = 'unknown'
        slices.append(a)
        print(f"  z={{z_val}}: {{a.n_obs}} cells x {{a.n_vars}} genes, cell_types={{a.obs['cell_type'].nunique()}}")

    # Create cell-type basis from visible slices
    all_adata = ad.concat(slices, axis=0, join='inner')
    cell_types = sorted(all_adata.obs['cell_type'].unique())
    basis_X = np.zeros((len(cell_types), all_adata.n_vars))
    for i, ct in enumerate(cell_types):
        mask = all_adata.obs['cell_type'] == ct
        if mask.sum() > 0:
            basis_X[i] = all_adata.X[mask].mean(axis=0)
    adata_basis = ad.AnnData(X=basis_X, var=all_adata.var.copy())
    adata_basis.obs.index = cell_types
    # Add required group column for preprocess
    adata_basis.obs['group'] = 'MERFISH'
    print(f"Basis: {{len(cell_types)}} cell types")

    # Combine and preprocess for 3D reconstruction
    print("\\nPreprocessing with SINTER3D utils...")
    from SINTER3D import utils

    # Compute slice distances from z values
    slice_dists = []
    for j in range(len(visible_z) - 1):
        slice_dists.append(visible_z[j+1] - visible_z[j])
    # Add first and last distance (mirror)
    if slice_dists:
        slice_dists = [slice_dists[0]] + slice_dists + [slice_dists[-1]]
    else:
        slice_dists = [10.0, 10.0]

    try:
        adata_st, adata_basis_processed = utils.preprocess(
            slices,
            adata_basis,
            celltype_ref_col='cell_type',
            sample_col='group',
            n_hvg_group=min(100, all_adata.n_vars),
            slice_dist_micron=slice_dists,
        )
        print(f"Preprocessed: {{adata_st.n_obs}} spots, {{adata_st.n_vars}} genes")
        print(f"3D coords range: x=[{{adata_st.obsm['3D_coor'][:,0].min():.1f}},{{adata_st.obsm['3D_coor'][:,0].max():.1f}}]")
        print(f"                 y=[{{adata_st.obsm['3D_coor'][:,1].min():.1f}},{{adata_st.obsm['3D_coor'][:,1].max():.1f}}]")
        print(f"                 z=[{{adata_st.obsm['3D_coor'][:,2].min():.1f}},{{adata_st.obsm['3D_coor'][:,2].max():.1f}}]")
    except Exception as e:
        print(f"WARNING: preprocess failed ({{e}}), continuing with manual prep")
        # Fall back to manual preparation
        adata_st = all_adata.copy()
        adata_basis_processed = adata_basis.copy()
        # Ensure 3D_coor is set
        if '3D_coor' not in adata_st.obsm:
            # Use 3D spatial directly if available, else construct from 2D
            if all(s.obsm['spatial'].shape[1] >= 3 for s in slices):
                adata_st.obsm['3D_coor'] = np.concatenate([s.obsm['spatial'] for s in slices], axis=0)
            else:
                adata_st.obsm['3D_coor'] = np.column_stack([
                    np.concatenate([s.obsm['spatial'][:, 0] for s in slices]),
                    np.concatenate([np.full(s.n_obs, visible_z[i]) for i, s in enumerate(slices)]),
                    np.concatenate([s.obsm['spatial'][:, 1] for s in slices]),
                ])
        # Ensure count and library_size
        if 'count' not in adata_st.obsm:
            adata_st.obsm['count'] = adata_st.X.copy()
        if 'library_size' not in adata_st.obs.columns:
            adata_st.obs['library_size'] = np.sum(adata_st.X, axis=1)
        # Ensure slice column is numeric
        adata_st.obs['slice'] = adata_st.obs['slice'].astype(int)

    # Train SINTER3D
    print(f"\\nTraining SINTER3D ({training_steps} steps, hidden_dims={hidden_dims})...")
    try:
        # Use model_spatial directly to control hidden_dims (avoid OOM with 40+ slices)
        from SINTER3D.model_spatial import Model as SpatialModel

        model = SpatialModel(
            slices,
            adata_st,
            adata_basis_processed,
            slice_idx=list(range(len(slices))),
            hidden_dims={json.dumps(hidden_dims)},
            training_steps={training_steps},
            lr=0.001,
            seed=42,
            patience=200,
            mid_channel=200,
            save_path='{output_dir}',
            use_type='train',
            normalize=100,
            alpha_poisson=5,
            lambda_feature=4,
            gamma_feature=3,
        )
        model.train()

        # Load query coordinates
        query_coords = np.load('{query_coords_path}')
        print(f"Query coords: {{query_coords.shape}}")

        # Create AnnData for query spots
        query_adata = ad.AnnData(
            X=np.zeros((query_coords.shape[0], adata_st.n_vars)),
            var=adata_st.var.copy(),
            obsm={{'3D_coor': query_coords}},
        )
        query_adata.obs['slice'] = 999  # New slice ID for query

        # Use inference_latent to predict expression at new coordinates
        print("Running inference_latent with decode=True...")
        model.inference_latent(
            query_adata,
            coord_key='3D_coor',
            slice_key='slice',
            method='distance',
            decode=True,
            deconv=False,
        )

        # Extract predicted expression
        if 'X_pred' in query_adata.obsm:
            X_pred = query_adata.obsm['X_pred']
        elif 'pred' in query_adata.obsm:
            X_pred = query_adata.obsm['pred']
        else:
            print(f"WARNING: No prediction found. Available obsm keys: {{list(query_adata.obsm.keys())}}")
            X_pred = np.zeros((query_coords.shape[0], adata_st.n_vars))

        X_pred = np.asarray(X_pred)
        # Clip to non-negative
        X_pred = np.clip(X_pred, 0, None)
        np.save('{output_dir}/X_pred.npy', X_pred)
        print(f"Saved predictions: {{X_pred.shape}}, range=[{{X_pred.min():.2f}}, {{X_pred.max():.2f}}]")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {{e}}")
        sys.exit(1)

if __name__ == '__main__':
    main()
'''

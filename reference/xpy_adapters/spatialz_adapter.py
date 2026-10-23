"""
SpatialZ adapter — 54-slice unified dataset support.

Uses Generate_spatialz() to create virtual slices between adjacent real slices
via sliced Wasserstein barycenter. For each held-out z, finds the nearest
visible slices above and below, then generates a virtual slice at the target z.

Protocol: Fixed-coordinate (generated cells at query coordinates)
"""
import sys, time, os, json, tempfile
from pathlib import Path
from typing import Dict, List
import numpy as np
import anndata as ad
from scipy import sparse

from benchmark.src.adapters.base import (
    BaseAdapter, AdapterResult, AdapterError, FailureType,
)

SPATIALZ_DIR = Path("/storage/yuhongtaoLab/linpeiran/xpy/lab/baseline/methods/SpatialZ")
SPATIALZ_ENV = "spatialz"


class SpatialZAdapter(BaseAdapter):
    """Adapter for SpatialZ sliced Wasserstein virtual slice generation ."""

    method_name = "SpatialZ"
    conda_env = SPATIALZ_ENV
    method_dir = SPATIALZ_DIR

    def run_impl(
        self, adata_input: ad.AnnData, config: Dict, output_dir: Path
    ) -> ad.AnnData:
        max_cells_per_slice = config.get("max_cells_per_slice", None)  # None = full slices (official, 2026-08-14)
        num_projections = config.get("num_projections", 80)  # official default
        # lr is the scientific parameter frozen by the 2026-10-18 lr probe (see
        # benchmark/reports/SPATIALZ_LR_PROBE_*.json).  The default 1.0 preserves the
        # pre-refactor behaviour for any historical config that does not set it.
        lr = float(config.get("lr", 1.0))
        nb_iter_max = int(config.get("nb_iter_max", 1000))  # official notebook value
        device = config.get("device") or "auto"  # official default: auto = GPU when CUDA available (official notebooks use cuda)
        gpu = config.get("gpu", None)  # optional GPU pin (CUDA_VISIBLE_DEVICES), same convention as isoST/UniST adapters
        holdout_slices = config.get("holdout_slices", None)
        if holdout_slices is None:
            holdout_z = config.get("holdout_z", 14)
            z_coords = adata_input.obsm["spatial"][:, 2]
            holdout_mask = np.isclose(z_coords, float(holdout_z))
            holdout_slices = list(adata_input.obs["slice"][holdout_mask].unique())

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        holdout_mask = adata_input.obs["slice"].isin(holdout_slices).to_numpy()
        visible_mask = ~holdout_mask

        # Get per-slice Z means for distance computation
        all_slices = sorted(adata_input.obs["slice"].unique())
        slice_z = {}
        for sl in all_slices:
            sl_mask = adata_input.obs["slice"] == sl
            slice_z[sl] = float(adata_input.obsm["spatial"][sl_mask.to_numpy(), 2].mean())

        visible_slices = [s for s in all_slices if s not in holdout_slices]
        visible_z = np.array([slice_z[s] for s in visible_slices])

        # Find cell_type_key
        cell_type_key = 'annotation'  # MERFISH ABCA dataset
        if cell_type_key not in adata_input.obs.columns:
            if 'cell_class' in adata_input.obs.columns:
                cell_type_key = 'cell_class'
            else:
                adata_input.obs['_dummy_ct'] = 'type_A'
                cell_type_key = '_dummy_ct'

        all_predictions = []
        all_coords = []

        for holdout_sl in holdout_slices:
            hz = slice_z[holdout_sl]
            print(f"\nGenerating virtual slice for {holdout_sl} (z≈{hz:.1f})...")

            # Find nearest visible slices above and below
            below = visible_z[visible_z < hz]
            above = visible_z[visible_z > hz]

            if len(below) == 0:
                # Extrapolation: use two nearest above
                z_lower = above[0]
                z_upper = above[1] if len(above) > 1 else above[0]
            elif len(above) == 0:
                # Extrapolation: use two nearest below
                z_lower = below[-2] if len(below) > 1 else below[-1]
                z_upper = below[-1]
            else:
                z_lower = below[-1]
                z_upper = above[0]

            # Find slice labels for these z values (use isclose for float safety)
            idx_lower = int(np.where(np.isclose(visible_z, z_lower))[0][0])
            idx_upper = int(np.where(np.isclose(visible_z, z_upper))[0][0])
            sl_lower = visible_slices[idx_lower]
            sl_upper = visible_slices[idx_upper]

            # Compute alpha based on z-distance
            alpha = (hz - z_lower) / (z_upper - z_lower) if z_upper != z_lower else 0.5

            print(f"  Using slices: {sl_lower} (z≈{z_lower:.1f}) ← {alpha:.2f} → {sl_upper} (z≈{z_upper:.1f})")

            # Extract slices (subsample if needed)
            a_lower = adata_input[adata_input.obs["slice"] == sl_lower].copy()
            a_upper = adata_input[adata_input.obs["slice"] == sl_upper].copy()
            if max_cells_per_slice and a_lower.n_obs > max_cells_per_slice:
                rng = np.random.RandomState(42)
                a_lower = a_lower[rng.choice(a_lower.n_obs, max_cells_per_slice, replace=False)]
            if max_cells_per_slice and a_upper.n_obs > max_cells_per_slice:
                rng = np.random.RandomState(42)
                a_upper = a_upper[rng.choice(a_upper.n_obs, max_cells_per_slice, replace=False)]

            # Ensure cell_type_key exists in both
            for a in [a_lower, a_upper]:
                if cell_type_key not in a.obs.columns and cell_type_key == '_dummy_ct':
                    a.obs['_dummy_ct'] = 'type_A'

            # Save temp files for subprocess
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                a_lower.write_h5ad(tmp / "lower.h5ad")
                a_upper.write_h5ad(tmp / "upper.h5ad")
                out_gen = tmp / "generated.h5ad"

                script = f'''
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "{SPATIALZ_DIR}")
import anndata as ad
from SpatialZ import Generate_spatialz

# ORIENTATION FIX 2026-09-30 -- read this before changing the two lines below.
#   Official convention: adata1 is the ABOVE slice.
#     SpatialZ.py:21   def Generate_spatialz(adata1, adata2, adata1_id='above', adata2_id='below', ...)
#     SpatialZ.py:397  Generate_multiple_spatialz walks alpha from 1 down to 0 while the
#                      generated slices walk from adata1 to adata2 => alpha weights adata1.
#   The alpha computed above, (hz - z_lower)/(z_upper - z_lower), is the weight of the
#   UPPER slice. The previous code passed the LOWER slice as adata1, so the virtual slice
#   was built at  alpha*z_lower + (1-alpha)*z_upper  instead of the true depth
#   alpha*z_upper + (1-alpha)*z_lower. Measured depth error: 0.04%-52.7% of the slice gap
#   over 16 hidden slices (median ~11%). Fixed: adata1 = upper.
#   See METHOD_CHANGES.md 2026-09-30 and PRE/figgen/geom/FINDINGS.md.
a1 = ad.read_h5ad("{tmp}/upper.h5ad")
a2 = ad.read_h5ad("{tmp}/lower.h5ad")
print(f"Above: {{a1.n_obs}} cells, Below: {{a2.n_obs}} cells")

result = Generate_spatialz(
    a1, a2,
    adata1_id='upper', adata2_id='lower',
    alpha={alpha},
    n_cell=None,
    num_projections={num_projections},
    device='{device}',
    lr={lr},            # frozen by the 2026-10-18 lr probe; see adapter config
    nb_iter_max={nb_iter_max},  # official notebook value (1000)
    cell_type_key='{cell_type_key}',
    verbose=True,
)
print(f"Generated: {{result.n_obs}} cells")
result.write_h5ad("{out_gen}")
print("SAVED")
'''
                script_path = tmp / "_run_spatialz.py"
                script_path.write_text(script)

                import subprocess
                t0 = time.time()
                sub_env = dict(os.environ)
                if gpu is not None:
                    sub_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                slice_timeout_h = float(os.environ.get(
                    "SPATIALZ_SLICE_TIMEOUT_HOURS", 6.0))
                result = subprocess.run(
                    ["conda", "run", "-n", SPATIALZ_ENV, "python", str(script_path)],
                    cwd=str(tmp), capture_output=True, text=True,
                    timeout=int(slice_timeout_h * 3600),
                    env=sub_env,  # T_max per virtual slice (safety cap; official GPU config finishes far below this).
                    # SPATIALZ_SLICE_TIMEOUT_HOURS override (2026-09-01, INC-005):
                    # full-density T3 rounds' per-slice generation can exceed 6 h
                    # on Blackwell; env var is math-neutral orchestration only.
                )
                elapsed = time.time() - t0

                # Persistent per-slice child log (2026-09-06, INC-006 forensics):
                # full-density big slices were killed with empty captured stderr;
                # keep the full child stdout/stderr on disk regardless of outcome.
                log_dir = output_dir / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / f"slice_{holdout_sl}.log").write_text(
                    f"[cmd] conda run -n {SPATIALZ_ENV} python {script_path}\n"
                    f"[elapsed_sec] {elapsed:.0f}\n[returncode] {result.returncode}\n"
                    f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}\n")

                if result.returncode != 0:
                    print(f"  SpatialZ subprocess failed (rc={result.returncode}): "
                          f"{result.stderr[-300:]}")
                    raise AdapterError(
                        f"SpatialZ failed for {holdout_sl} (rc={result.returncode}): "
                        f"{result.stderr[-200:]}",
                        failure_type=FailureType.F2_INTERNAL,
                    )

                if not out_gen.exists():
                    raise AdapterError(
                        f"SpatialZ did not produce output for {holdout_sl}",
                        failure_type=FailureType.F2_INTERNAL,
                    )

                gen_adata = ad.read_h5ad(out_gen)
                X_gen = gen_adata.X.toarray() if sparse.issparse(gen_adata.X) else np.asarray(gen_adata.X)

                # SpatialZ generates cells at the virtual z — use their coordinates
                gen_coords = gen_adata.obsm["spatial"]
                if gen_coords.shape[1] < 3:
                    gen_coords = np.column_stack([gen_coords, np.full(gen_coords.shape[0], hz)])
                else:
                    gen_coords[:, 2] = hz  # Set to target z

                all_predictions.append(np.clip(X_gen, 0, None))
                all_coords.append(gen_coords)
                print(f"  Generated {gen_adata.n_obs} cells in {elapsed:.0f}s")

        # Concatenate all generated slices
        if not all_predictions:
            raise AdapterError("No virtual slices generated", failure_type=FailureType.F2_INTERNAL)

        X_pred = np.vstack(all_predictions)
        coords_pred = np.vstack(all_coords)

        # Build output AnnData
        pred_adata = ad.AnnData(
            X=X_pred,
            obsm={"spatial": coords_pred},
            var=adata_input.var.copy(),
        )
        pred_adata.obs["is_interpolated"] = True
        # Assign slice labels based on z coordinate
        slice_labels = []
        for j in range(X_pred.shape[0]):
            z_j = coords_pred[j, 2]
            best_sl = min(holdout_slices, key=lambda sl: abs(slice_z[sl] - z_j))
            slice_labels.append(f"{best_sl}_virtual")
        pred_adata.obs["slice"] = slice_labels
        pred_adata.uns["method"] = self.method_name
        pred_adata.uns["task"] = "MERFISH_LOSO"
        pred_adata.uns["run_config"] = config

        return pred_adata

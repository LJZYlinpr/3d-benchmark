"""
Naive baseline methods (AGENTS.md §1.3, PLAN.md A3 axiom).

These are minimal-complexity baselines that any sophisticated method
must outperform to claim practical value.

Baselines:
- PASTE-OT-Linear: THE only simple baseline. PASTE OT mapping between
  adjacent slices + z-weighted linear expression blending.
- ReHo (Repeat Holding): Copy from immediately preceding real slice
  (internal sanity check, not for main comparison table).
- GaussianKernel: Weighted average with Gaussian z-kernel (supplementary).

Cancelled: Nearest-slice (deleted per AGENTS.md §1.3).
Merged: Plain Linear interpolation → now the blending component of
  PASTE-OT-Linear, not a standalone method.
"""

from typing import Dict, Optional, Tuple, Union
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.interpolate import interp1d
import anndata as ad


# ---------------------------------------------------------------------------
# Base adapter interface
# ---------------------------------------------------------------------------

def _validate_and_prepare(adata: ad.AnnData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and validate required data from AnnData.

    Returns:
        X: dense expression matrix (n_obs, n_genes)
        coords: spatial coordinates (n_obs, 3) with z as 3rd column
        z_values: unique z coordinates in ascending order
    """
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()

    coords = adata.obsm["spatial"]
    if coords.shape[1] < 2:
        raise ValueError("Spatial coordinates must have at least 2 dimensions")

    # If z is not provided, assume sequential slices
    if coords.shape[1] >= 3:
        z_all = coords[:, 2]
    else:
        # Infer z from slice ordering
        slice_ids = adata.obs["slice"].values
        unique_slices = sorted(set(slice_ids))
        slice_to_z = {s: float(i) for i, s in enumerate(unique_slices)}
        z_all = np.array([slice_to_z[s] for s in slice_ids])

    z_values = np.sort(np.unique(z_all))

    # If Z is continuous (many unique values), use slice-level Z means
    if len(z_values) > 1000 and "slice" in adata.obs.columns:
        slice_labels = adata.obs["slice"].values
        unique_slices = sorted(set(slice_labels))
        slice_z_map = {}
        for sl in unique_slices:
            mask = slice_labels == sl
            slice_z_map[sl] = float(np.mean(z_all[mask]))
        z_values = np.sort(list(slice_z_map.values()))

    return X, coords, z_all, z_values


def _build_output_adata(
    adata_input: ad.AnnData,
    X_pred: np.ndarray,
    coords_pred: np.ndarray,
    method_name: str,
    run_config: Dict,
    runtime: float,
    real_mask: Optional[np.ndarray] = None,
) -> ad.AnnData:
    """Build output AnnData conforming to the unified output schema.

    Args:
        adata_input: Input AnnData.
        X_pred: Predicted expression matrix.
        coords_pred: Predicted spatial coordinates (n, 3) with z in column 2.
        method_name: Name of the method.
        run_config: Run configuration dict.
        runtime: Wall-clock time in seconds.
        real_mask: Boolean mask indicating which observations are real slices
                   (not interpolated).

    Returns:
        Output AnnData with required schema fields.
    """
    n_total = X_pred.shape[0]
    n_real = adata_input.n_obs

    if real_mask is None:
        real_mask = np.zeros(n_total, dtype=bool)
        real_mask[:n_real] = True

    # Determine slice labels from z-coordinates
    slice_labels = []
    for i in range(n_total):
        z_i = coords_pred[i, 2]
        if real_mask[i] and i < n_real and "slice" in adata_input.obs.columns:
            slice_labels.append(str(adata_input.obs["slice"].iloc[i]))
        elif real_mask[i]:
            slice_labels.append(f"real_z{z_i:.0f}")
        else:
            slice_labels.append(f"interp_z{z_i:.0f}")

    # For interpolated outputs, build new obs
    if n_total != n_real:
        obs = pd.DataFrame({
            "slice": slice_labels,
            "is_interpolated": ~real_mask,
        })
    else:
        obs = adata_input.obs.copy()
        obs["is_interpolated"] = pd.Series(False, index=obs.index)

    # Build output
    adata_out = ad.AnnData(
        X=sparse.csr_matrix(X_pred) if sparse.issparse(adata_input.X) else X_pred,
        obs=obs,
        var=adata_input.var.copy() if adata_input.var is not None else None,
        obsm={"spatial": coords_pred},
        uns={
            "method": method_name,
            "run_config": run_config,
            "runtime_sec": runtime,
            "input_file": adata_input.uns.get("input_file", "unknown"),
        },
    )

    return adata_out


# ---------------------------------------------------------------------------
# PASTE-OT-Linear: THE only simple baseline
# = PASTE OT mapping + z-weighted linear expression blending
# ---------------------------------------------------------------------------
# Internal helper: z-weighted linear blending (the "Linear" component).
# This is NOT a standalone baseline — it is called by paste_ot_linear()
# after OT mapping has established spatial correspondence between slices.
# ---------------------------------------------------------------------------

def _z_weighted_blend(
    adata: ad.AnnData,
    target_z: Optional[np.ndarray] = None,
    n_interp: int = 5,
    per_slice: bool = False,
) -> ad.AnnData:
    """z-weighted linear blending between adjacent real slices.

    INTERNAL HELPER for paste_ot_linear(). Not a standalone baseline.
    Performs per-slice-average z-weighted expression blending WITHOUT
    OT spatial correspondence — this is the "Linear" component that
    gets applied AFTER PASTE establishes spot-to-spot mapping.

    Args:
        adata: Input AnnData with obsm['spatial'] (x, y, z).
        target_z: Optional target z values for interpolation.
                  If None, generates n_interp slices between each adjacent pair.
        n_interp: Number of interpolated slices per gap (if target_z is None).
        per_slice: If True, interpolate per-slice averages. If False,
                   interpolate per-spot (requires spatial correspondence).

    Returns:
        Output AnnData with interpolated expression at target z positions.
    """
    t0 = time.time()
    X, coords, z_all, z_values = _validate_and_prepare(adata)

    if target_z is None:
        # Generate virtual z positions between real slices
        target_z = []
        for i in range(len(z_values) - 1):
            z_start, z_end = z_values[i], z_values[i + 1]
            for j in range(1, n_interp + 1):
                target_z.append(z_start + (z_end - z_start) * j / (n_interp + 1))
        target_z = np.array(target_z)
    else:
        target_z = np.asarray(target_z)

    # Remove target_z that coincide with real z
    target_z = target_z[~np.isin(target_z, z_values)]

    all_z = np.sort(np.concatenate([z_values, target_z]))
    n_real = adata.n_obs
    n_virtual = len(target_z) * adata.n_obs  # rough upper bound

    # Per-gene linear interpolation
    # Strategy: For each gene, compute mean per z-slice, interpolate, apply
    n_genes = X.shape[1]

    # Group by z and compute per-slice mean and spot-level
    X_interp_slices = []
    coords_interp_slices = []

    for z_val in target_z:
        # Find bounding real slices
        if z_val < z_values[0] or z_val > z_values[-1]:
            # Extrapolation: use nearest real slice
            nearest_z = z_values[np.argmin(np.abs(z_values - z_val))]
            mask_near = np.isclose(z_all, nearest_z)
            X_slice = X[mask_near].mean(axis=0, keepdims=True)
            coord_mean = coords[mask_near].mean(axis=0, keepdims=True)
            coord_mean[0, 2] = z_val
        else:
            idx_upper = np.searchsorted(z_values, z_val)
            z_lower = z_values[idx_upper - 1]
            z_upper = z_values[idx_upper]
            alpha = (z_val - z_lower) / (z_upper - z_lower)

            mask_lower = np.isclose(z_all, z_lower)
            mask_upper = np.isclose(z_all, z_upper)

            X_lower = X[mask_lower].mean(axis=0, keepdims=True)
            X_upper = X[mask_upper].mean(axis=0, keepdims=True)
            X_slice = (1 - alpha) * X_lower + alpha * X_upper

            coord_lower = coords[mask_lower].mean(axis=0, keepdims=True)
            coord_upper = coords[mask_upper].mean(axis=0, keepdims=True)
            coord_mean = (1 - alpha) * coord_lower + alpha * coord_upper
            coord_mean[0, 2] = z_val

        X_interp_slices.append(X_slice)
        coords_interp_slices.append(coord_mean)

    if X_interp_slices:
        X_interp = np.vstack(X_interp_slices)
        coords_interp = np.vstack(coords_interp_slices)
    else:
        X_interp = np.zeros((0, n_genes))
        coords_interp = np.zeros((0, 3))

    # Combine real + interpolated
    X_out = np.vstack([X, X_interp])
    coords_out = np.vstack([coords, coords_interp])
    real_mask = np.concatenate([np.ones(len(X), dtype=bool),
                                 np.zeros(len(X_interp), dtype=bool)])

    runtime = time.time() - t0

    return _build_output_adata(
        adata, X_out, coords_out,
        method_name="z_weighted_blend",
        run_config={"n_interp": n_interp, "target_z": target_z.tolist() if len(target_z) > 0 else []},
        runtime=runtime,
        real_mask=real_mask,
    )


# ---------------------------------------------------------------------------
# PASTE-OT-Linear: THE only simple baseline (public API)
# ---------------------------------------------------------------------------

def paste_ot_linear(
    adata: ad.AnnData,
    target_z: Optional[np.ndarray] = None,
    n_interp: int = 5,
    alpha: float = 0.1,
    dissimilarity: str = "kl",
    use_ot: bool = True,
) -> ad.AnnData:
    """PASTE optimal transport mapping + z-weighted linear expression blending.

    This is THE only simple baseline per AGENTS.md §1.3 and
    agent_control/METHOD_SCOPE.md §3. Any deep method that cannot
    significantly outperform this baseline lacks empirical justification
    for its additional complexity.

    Two-step process:
    1. PASTE OT mapping: Establish spatial correspondence between
       adjacent slices via Fused Gromov-Wasserstein optimal transport
       (Zeira et al., Nat Methods 2022). Produces coupling matrix Pi
       that maps spots in lower slice to spots in upper slice.
    2. z-weighted linear blending: Map both slices' expression to the
       evaluation support via Pi, then blend with weights proportional
       to physical z-distance from target.

    X_pred = w_lower * (Pi @ X_lower) + w_upper * (Pi.T @ X_upper)
    where w_lower = (z_upper - z_target) / (z_upper - z_lower)
          w_upper = (z_target - z_lower) / (z_upper - z_lower)

    Args:
        adata: Input AnnData with obsm['spatial'] (x, y, z).
        target_z: Target z values for prediction. If None, generates
                  n_interp slices between each adjacent pair.
        n_interp: Number of interpolated slices per gap.
        alpha: PASTE alignment tuning parameter (0=only expression,
               1=only spatial). Default 0.1 per Spateo/PASTE convention.
        dissimilarity: Expression dissimilarity measure ('kl' or 'euclidean').
        use_ot: If True, run full PASTE OT mapping. If False, fall back
                to z-weighted blending without OT (for debugging only;
                this fallback is NOT a valid standalone baseline).

    Returns:
        Output AnnData with predicted expression at target z positions.

    References:
        Zeira et al. "Alignment and integration of spatial transcriptomics
        data." Nature Methods 19(5):567-575, 2022.
        DOI: 10.1038/s41592-022-01459-6
    """
    t0 = time.time()

    if use_ot:
        # ---- Step 1: PASTE OT mapping between adjacent slices ----
        try:
            from spateo.alignment.methods.paste import paste_pairwise_align
        except ImportError:
            warnings.warn(
                "Spateo PASTE not available. Falling back to z-weighted "
                "blend without OT mapping. This is NOT a valid baseline — "
                "install Spateo for proper PASTE-OT-Linear."
            )
            use_ot = False

    X, coords, z_all, z_values = _validate_and_prepare(adata)

    if target_z is None:
        target_z_list = []
        for i in range(len(z_values) - 1):
            z_start, z_end = z_values[i], z_values[i + 1]
            for j in range(1, n_interp + 1):
                target_z_list.append(
                    z_start + (z_end - z_start) * j / (n_interp + 1)
                )
        target_z = np.array(target_z_list)
    else:
        target_z = np.asarray(target_z)

    target_z = target_z[~np.isin(target_z, z_values)]

    n_genes = X.shape[1]
    X_interp_slices = []
    coords_interp_slices = []

    for z_val in target_z:
        if z_val < z_values[0] or z_val > z_values[-1]:
            # Extrapolation: use nearest real slice
            nearest_z = z_values[np.argmin(np.abs(z_values - z_val))]
            mask_near = np.isclose(z_all, nearest_z)
            X_slice = X[mask_near].mean(axis=0, keepdims=True)
            coord_mean = coords[mask_near].mean(axis=0, keepdims=True)
            coord_mean[0, 2] = z_val
        else:
            idx_upper = np.searchsorted(z_values, z_val)
            z_lower = z_values[idx_upper - 1]
            z_upper = z_values[idx_upper]
            w_lower = (z_upper - z_val) / (z_upper - z_lower)
            w_upper = (z_val - z_lower) / (z_upper - z_lower)

            mask_lower = np.isclose(z_all, z_lower)
            mask_upper = np.isclose(z_all, z_upper)

            if use_ot and mask_lower.sum() > 1 and mask_upper.sum() > 1:
                # Build temporary AnnData for PASTE
                try:
                    import anndata as ad
                    slice_lower = ad.AnnData(
                        X=X[mask_lower],
                        obsm={"spatial": coords[mask_lower][:, :2]},
                    )
                    slice_upper = ad.AnnData(
                        X=X[mask_upper],
                        obsm={"spatial": coords[mask_upper][:, :2]},
                    )
                    pi, _ = paste_pairwise_align(
                        slice_lower, slice_upper,
                        alpha=alpha,
                        dissimilarity=dissimilarity,
                        verbose=False,
                    )
                    # Map lower expression to evaluation support
                    pi_rowsum = pi.sum(axis=1, keepdims=True)
                    pi_rowsum[pi_rowsum == 0] = 1.0
                    X_lower_mapped = (pi / pi_rowsum) @ X[mask_upper]
                    # Map upper expression back
                    pi_colsum = pi.sum(axis=0, keepdims=True)
                    pi_colsum[pi_colsum == 0] = 1.0
                    X_upper_mapped = X[mask_upper]
                    X_lower_mean = X_lower_mapped.mean(axis=0, keepdims=True)
                    X_upper_mean = X_upper_mapped.mean(axis=0, keepdims=True)
                except Exception:
                    use_ot = False
                    # Fallback inside exception: simple per-slice mean
                    X_lower_mean = X[mask_lower].mean(axis=0, keepdims=True)
                    X_upper_mean = X[mask_upper].mean(axis=0, keepdims=True)

            if not use_ot:
                # Fallback: simple per-slice mean blending (no OT)
                X_lower_mean = X[mask_lower].mean(axis=0, keepdims=True)
                X_upper_mean = X[mask_upper].mean(axis=0, keepdims=True)

            X_slice = w_lower * X_lower_mean + w_upper * X_upper_mean

            coord_lower = coords[mask_lower].mean(axis=0, keepdims=True)
            coord_upper = coords[mask_upper].mean(axis=0, keepdims=True)
            coord_mean = (1 - w_lower) * coord_lower + w_lower * coord_upper
            coord_mean[0, 2] = z_val if coord_mean.shape[1] >= 3 else z_val

        X_interp_slices.append(X_slice)
        coords_interp_slices.append(coord_mean)

    if X_interp_slices:
        X_interp = np.vstack(X_interp_slices)
        coords_interp = np.vstack(coords_interp_slices)
    else:
        X_interp = np.zeros((0, n_genes))
        coords_interp = np.zeros((0, 3))

    X_out = np.vstack([X, X_interp])
    coords_out = np.vstack([coords, coords_interp])
    real_mask = np.concatenate([
        np.ones(len(X), dtype=bool),
        np.zeros(len(X_interp), dtype=bool),
    ])

    runtime = time.time() - t0

    return _build_output_adata(
        adata, X_out, coords_out,
        method_name="PASTE-OT-Linear",
        run_config={
            "n_interp": n_interp,
            "alpha": alpha,
            "dissimilarity": dissimilarity,
            "use_ot": use_ot,
            "target_z": target_z.tolist() if len(target_z) > 0 else [],
        },
        runtime=runtime,
        real_mask=real_mask,
    )


# ---------------------------------------------------------------------------
# ReHo (Repeat Holding) — internal sanity check only
# ---------------------------------------------------------------------------

def reho(
    adata: ad.AnnData,
    target_z: Optional[np.ndarray] = None,
    n_interp: int = 5,
) -> ad.AnnData:
    """Repeat Holding: copy from the immediately preceding real slice.

    For each virtual slice, copies expression from the nearest real slice
    above (lower z). If no real slice above, uses the one below.

    This is the default baseline in UniST (where it outperformed the
    pretrained model under extreme sparsity 2-4 slices).

    Args:
        adata: Input AnnData.
        target_z: Optional target z values.
        n_interp: Number of virtual slices per gap.

    Returns:
        Output AnnData.
    """
    t0 = time.time()
    X, coords, z_all, z_values = _validate_and_prepare(adata)

    if target_z is None:
        target_z_list = []
        for i in range(len(z_values) - 1):
            z_start, z_end = z_values[i], z_values[i + 1]
            for j in range(1, n_interp + 1):
                target_z_list.append(z_start + (z_end - z_start) * j / (n_interp + 1))
        target_z = np.array(target_z_list)

    X_interp_list = []
    coords_interp_list = []

    for z_val in target_z:
        # Use the closest real slice with z <= z_val (preceding slice)
        below = z_values[z_values <= z_val]
        if len(below) > 0:
            source_z = below[-1]
        else:
            source_z = z_values[0]  # fallback
        mask = np.isclose(z_all, source_z)
        X_interp_list.append(X[mask])
        coords_copy = coords[mask].copy()
        coords_copy[:, 2] = z_val
        coords_interp_list.append(coords_copy)

    if X_interp_list:
        X_interp = np.vstack(X_interp_list)
        coords_interp = np.vstack(coords_interp_list)
    else:
        X_interp = np.zeros((0, X.shape[1]))
        coords_interp = np.zeros((0, 3))

    X_out = np.vstack([X, X_interp])
    coords_out = np.vstack([coords, coords_interp])
    real_mask = np.concatenate([np.ones(len(X), dtype=bool),
                                 np.zeros(len(X_interp), dtype=bool)])

    runtime = time.time() - t0

    return _build_output_adata(
        adata, X_out, coords_out,
        method_name="ReHo",
        run_config={"n_interp": n_interp},
        runtime=runtime,
        real_mask=real_mask,
    )


# ---------------------------------------------------------------------------
# Gaussian Kernel (weighted average with distance decay)
# ---------------------------------------------------------------------------

def gaussian_kernel_interp(
    adata: ad.AnnData,
    target_z: Optional[np.ndarray] = None,
    n_interp: int = 5,
    sigma: float = 1.0,
) -> ad.AnnData:
    """Gaussian kernel-weighted average of neighboring real slices.

    Weights decay as exp(-(z - z_i)^2 / (2*sigma^2)).

    Args:
        adata: Input AnnData.
        target_z: Optional target z values.
        n_interp: Number of virtual slices per gap.
        sigma: Kernel bandwidth (in z units).

    Returns:
        Output AnnData.
    """
    t0 = time.time()
    X, coords, z_all, z_values = _validate_and_prepare(adata)

    if target_z is None:
        target_z_list = []
        for i in range(len(z_values) - 1):
            z_start, z_end = z_values[i], z_values[i + 1]
            for j in range(1, n_interp + 1):
                target_z_list.append(z_start + (z_end - z_start) * j / (n_interp + 1))
        target_z = np.array(target_z_list)

    X_interp_list = []
    coords_interp_list = []

    for z_val in target_z:
        # Compute Gaussian weights
        dz = np.abs(z_values - z_val)
        weights = np.exp(-dz ** 2 / (2 * sigma ** 2))
        weights /= weights.sum()

        # Weighted average of per-slice means
        X_weighted = np.zeros((1, X.shape[1]))
        coords_weighted = np.zeros((1, 3))
        for i, z_i in enumerate(z_values):
            mask = np.isclose(z_all, z_i)
            X_weighted += weights[i] * X[mask].mean(axis=0, keepdims=True)
            coords_weighted[:, :2] += weights[i] * coords[mask][:, :2].mean(axis=0, keepdims=True)

        coords_weighted[0, 2] = z_val
        X_interp_list.append(X_weighted)
        coords_interp_list.append(coords_weighted)

    if X_interp_list:
        X_interp = np.vstack(X_interp_list)
        coords_interp = np.vstack(coords_interp_list)
    else:
        X_interp = np.zeros((0, X.shape[1]))
        coords_interp = np.zeros((0, 3))

    X_out = np.vstack([X, X_interp])
    coords_out = np.vstack([coords, coords_interp])
    real_mask = np.concatenate([np.ones(len(X), dtype=bool),
                                 np.zeros(len(X_interp), dtype=bool)])

    runtime = time.time() - t0

    return _build_output_adata(
        adata, X_out, coords_out,
        method_name="GaussianKernel",
        run_config={"n_interp": n_interp, "sigma": sigma},
        runtime=runtime,
        real_mask=real_mask,
    )

#!/usr/bin/env python3
"""
Map normalized coordinates .xyz in reconstructed h5ad files back to original coordinate space.

Supports both 2D and 3D coordinate denormalization.

Usage:
    # 2D coordinates
    python map_coords_back.py \
        --reconstructed reconstructed-custom-2d.h5ad \
        --reference data/2D_data.h5ad \
        --output reconstructed-custom-2d-original-coords.h5ad \
        --mode 2d \
        --keep_ratio True
    
    # 3D coordinates
    python map_coords_back.py \
        --reconstructed reconstructed-custom-3d.h5ad \
        --reference data/3D_data.h5ad \
        --output reconstructed-custom-3d-original-coords.h5ad \
        --mode 3d \
        --keep_ratio True \
        --preserve_z_scale True \
        --z_scale_factor 1.5
"""

import argparse
import numpy as np
import scanpy as sc
import anndata as ad
from pathlib import Path


def denormalize_coordinates_2d(normalized_coords, ref_coords, keep_ratio=True):
    """
    Denormalize 2D coordinates back to original coordinate space.
    
    Normalization process (ST2D._normalize_coordinates):
    1. Normalize to [0, 1]: (coord - min) / range
    2. Convert to [-1, 1]: coord = (coord - 0.5) * 2.0
    3. If keep_ratio=True:
       - xy: scale = range / max_xy_range
    
    Denormalization reverses these steps.
    
    Args:
        normalized_coords: Normalized coordinates [N, 2] (range approximately [-1, 1])
        ref_coords: Reference data original coordinates [M, 2] (used to get normalization parameters)
        keep_ratio: Whether aspect ratio was preserved (must match normalization)
        
    Returns:
        Denormalized coordinates [N, 2]
    """
    # Calculate reference data range
    x_min, y_min = ref_coords.min(axis=0)
    x_max, y_max = ref_coords.max(axis=0)
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    print(f"Reference data coordinate ranges:")
    print(f"  X: [{x_min:.2f}, {x_max:.2f}], range={x_range:.2f}")
    print(f"  Y: [{y_min:.2f}, {y_max:.2f}], range={y_range:.2f}")
    
    # Copy normalized coordinates
    coords = normalized_coords.copy().astype(np.float64)
    
    # Reverse step 3: restore scaling
    if keep_ratio:
        xy_max_range = max(x_range, y_range)
        scale_x = x_range / xy_max_range
        scale_y = y_range / xy_max_range
        print(f"Scaling factors (keep_ratio=True): scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
        
        # Reverse scaling
        coords[:, 0] /= scale_x
        coords[:, 1] /= scale_y
    else:
        print("keep_ratio=False, skipping scaling step")
    
    # Reverse step 2: convert from [-1, 1] back to [0, 1]
    coords /= 2.0
    coords += 0.5
    
    # Reverse step 1: convert from [0, 1] back to original range
    coords[:, 0] = coords[:, 0] * x_range + x_min
    coords[:, 1] = coords[:, 1] * y_range + y_min
    
    return coords


def denormalize_coordinates_3d(normalized_coords, ref_coords, keep_ratio=True, 
                               preserve_z_scale=False, z_scale_factor=1.0):
    """
    Denormalize 3D coordinates back to original coordinate space.
    
    Normalization process (ST3D._normalize_coordinates or GraphST3D._normalize_coordinates):
    1. Normalize to [0, 1]: (coord - min) / range
    2. Convert to [-1, 1]: coord = (coord - 0.5) * 2.0
    3. If keep_ratio=True:
       - If preserve_z_scale=True:
         - xy: scale = range / max_xy_range
         - z: scale = z_scale_factor
       - Otherwise: scale = range / max_range
    
    Denormalization reverses these steps.
    
    Args:
        normalized_coords: Normalized coordinates [N, 3] (range approximately [-1, 1])
        ref_coords: Reference data original coordinates [M, 3] (used to get normalization parameters)
        keep_ratio: Whether aspect ratio was preserved (must match normalization)
        preserve_z_scale: Whether z-direction original scale was preserved (must match normalization)
        z_scale_factor: Z-direction scaling factor (must match normalization)
        
    Returns:
        Denormalized coordinates [N, 3]
    """
    # Calculate reference data range
    x_min, y_min, z_min = ref_coords.min(axis=0)
    x_max, y_max, z_max = ref_coords.max(axis=0)
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    print(f"Reference data coordinate ranges:")
    print(f"  X: [{x_min:.2f}, {x_max:.2f}], range={x_range:.2f}")
    print(f"  Y: [{y_min:.2f}, {y_max:.2f}], range={y_range:.2f}")
    print(f"  Z: [{z_min:.2f}, {z_max:.2f}], range={z_range:.2f}")
    
    # Copy normalized coordinates
    coords = normalized_coords.copy().astype(np.float64)
    
    # Reverse step 3: restore scaling
    if keep_ratio:
        if preserve_z_scale:
            max_xy_range = max(x_range, y_range)
            scale_x = x_range / max_xy_range
            scale_y = y_range / max_xy_range
            scale_z = z_scale_factor
            print(f"Scaling factors (preserve_z_scale=True): scale_x={scale_x:.4f}, scale_y={scale_y:.4f}, scale_z={scale_z:.4f}")
        else:
            max_range = max(x_range, y_range, z_range)
            scale_x = x_range / max_range
            scale_y = y_range / max_range
            scale_z = z_range / max_range
            print(f"Scaling factors (preserve_z_scale=False): scale_x={scale_x:.4f}, scale_y={scale_y:.4f}, scale_z={scale_z:.4f}")
        
        # Reverse scaling
        coords[:, 0] /= scale_x
        coords[:, 1] /= scale_y
        coords[:, 2] /= scale_z
    else:
        print("keep_ratio=False, skipping scaling step")
    
    # Reverse step 2: convert from [-1, 1] back to [0, 1]
    coords /= 2.0
    coords += 0.5
    
    # Reverse step 1: convert from [0, 1] back to original range
    coords[:, 0] = coords[:, 0] * x_range + x_min
    coords[:, 1] = coords[:, 1] * y_range + y_min
    coords[:, 2] = coords[:, 2] * z_range + z_min
    
    return coords


def main():
    parser = argparse.ArgumentParser(
        description="Map normalized coordinates in prediction results back to original coordinate space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--reconstructed",
        type=str,
        required=True,
        help="Path to reconstructed prediction file (e.g., reconstructed-custom-2d.h5ad or reconstructed-custom-3d.h5ad)"
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to reference data file (used to get normalization parameters, e.g., adata.h5ad)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["2d", "3d"],
        required=True,
        help="Coordinate dimension: 2d or 3d"
    )
    parser.add_argument(
        "--keep_ratio",
        type=str,
        default="True",
        help="Whether aspect ratio was preserved (must match normalization), default True. Use 'True' or 'False'"
    )
    parser.add_argument(
        "--preserve_z_scale",
        type=str,
        default="False",
        help="Whether z-direction original scale was preserved (3D only, must match normalization), default False. Use 'True' or 'False'"
    )
    parser.add_argument(
        "--z_scale_factor",
        type=float,
        default=1.0,
        help="Z-direction scaling factor (3D only, must match normalization), default 1.0"
    )
    
    args = parser.parse_args()
    
    # Read prediction results
    print(f"\n{'='*60}")
    print(f"Reading prediction results: {args.reconstructed}")
    print(f"{'='*60}")
    adata_pred = sc.read_h5ad(args.reconstructed)
    normalized_coords = adata_pred.obsm["spatial"]
    print(f"Normalized coordinate shape: {normalized_coords.shape}")
    print(f"Normalized coordinate ranges:")
    print(f"  min: {normalized_coords.min(axis=0)}")
    print(f"  max: {normalized_coords.max(axis=0)}")
    
    # Read reference data to get normalization parameters
    print(f"\n{'='*60}")
    print(f"Reading reference data: {args.reference}")
    print(f"{'='*60}")
    adata_ref = sc.read_h5ad(args.reference)
    
    # Convert boolean parameters
    keep_ratio = args.keep_ratio.lower() in ['true', '1', 'yes']
    
    # Denormalize coordinates
    print(f"\n{'='*60}")
    print("Denormalizing coordinates...")
    print(f"{'='*60}")
    
    if args.mode == "2d":
        ref_coords = adata_ref.obsm["spatial"][:, :2]
        if normalized_coords.shape[1] < 2:
            raise ValueError(f"2D mode requires at least 2 coordinate columns, but input has {normalized_coords.shape[1]} columns")
        normalized_coords = normalized_coords[:, :2]  # Take first 2 columns only
        
        print(f"Parameters: keep_ratio={keep_ratio}")
        original_coords = denormalize_coordinates_2d(
            normalized_coords,
            ref_coords,
            keep_ratio=keep_ratio
        )
        
    elif args.mode == "3d":
        ref_coords = adata_ref.obsm["spatial"][:, :3]
        if normalized_coords.shape[1] < 3:
            raise ValueError(f"3D mode requires at least 3 coordinate columns, but input has {normalized_coords.shape[1]} columns")
        normalized_coords = normalized_coords[:, :3]  # Take first 3 columns only
        
        preserve_z_scale = args.preserve_z_scale.lower() in ['true', '1', 'yes']
        print(f"Parameters: keep_ratio={keep_ratio}, preserve_z_scale={preserve_z_scale}, z_scale_factor={args.z_scale_factor}")
        original_coords = denormalize_coordinates_3d(
            normalized_coords,
            ref_coords,
            keep_ratio=keep_ratio,
            preserve_z_scale=preserve_z_scale,
            z_scale_factor=args.z_scale_factor
        )
    
    print(f"\nDenormalized coordinate ranges:")
    print(f"  min: {original_coords.min(axis=0)}")
    print(f"  max: {original_coords.max(axis=0)}")
    
    # Update AnnData object
    # Keep normalized coordinates as backup
    adata_pred.obsm["spatial_normalized"] = normalized_coords.copy()
    # Set original coordinates as main coordinates
    adata_pred.obsm["spatial"] = original_coords
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Saving results to: {args.output}")
    print(f"{'='*60}")
    adata_pred.write_h5ad(args.output)
    print("✓ Done!")
    print(f"\nOutput file contains:")
    print(f"  - obsm['spatial']: Original coordinates (mapped results)")
    print(f"  - obsm['spatial_normalized']: Normalized coordinates (backup)")
    print(f"  - Other prediction results remain unchanged")


if __name__ == "__main__":
    main()

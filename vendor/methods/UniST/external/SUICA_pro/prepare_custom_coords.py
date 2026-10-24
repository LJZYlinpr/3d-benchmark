#!/usr/bin/env python3
"""
Prepare custom coordinate files for INR prediction.

Supports normalization of 2D and 3D coordinates to ensure consistency with the coordinate space used during training.

Usage:
    # 2D coordinates
    python prepare_custom_coords.py \
        --mode 2d \
        --reference data/adata.h5ad \
        --coords data/coords.xyz \
        --output data/custom_coords_2d.npy \
        --keep_ratio True
    
    # 3D coordinates
    python prepare_custom_coords.py \
        --mode 3d \
        --reference data/adata.h5ad \
        --coords data/coords.xyz \
        --output data/custom_coords_3d.npy \
        --keep_ratio True \
        --preserve_z_scale True \
        --z_scale_factor 1.5
"""

import numpy as np
import argparse
from pathlib import Path
import scanpy as sc


def load_xyz(path: Path):
    """
    Load coordinates from xyz format file.
    
    Args:
        path: Path to xyz file
        
    Returns:
        numpy array with shape [N, 3]
    """
    coords = []
    with open(path, "r") as f:
        header = f.readline().strip()
        try:
            int(header)
            f.readline()  # Skip comment line
        except ValueError:
            f.seek(0)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = map(float, parts[-3:])
                coords.append((x, y, z))
            except ValueError:
                continue
    if not coords:
        raise RuntimeError(f"No coordinates found in xyz file: {path}")
    return np.array(coords, dtype=np.float64)


def normalize_coords_2d(coords, ref_coords, keep_ratio=True):
    """
    Normalize 2D coordinates to [-1, 1] range.
    
    Args:
        coords: Coordinates to normalize [N, 2]
        ref_coords: Reference coordinates [M, 2], used to determine normalization parameters
        keep_ratio: Whether to preserve aspect ratio
        
    Returns:
        Normalized coordinates [N, 2]
    """
    # Calculate reference data range
    ref_min = ref_coords.min(axis=0)
    ref_max = ref_coords.max(axis=0)
    ref_range = ref_max - ref_min
    
    # Normalize to [0, 1]
    coords = (coords - ref_min) / ref_range
    
    # Convert to [-1, 1]
    coords = (coords - 0.5) * 2.0
    
    # Aspect ratio scaling
    if keep_ratio:
        xy_max_range = max(ref_range[0], ref_range[1])
        coords[:, 0] *= ref_range[0] / xy_max_range
        coords[:, 1] *= ref_range[1] / xy_max_range
    
    return coords


def normalize_coords_3d(coords, ref_coords, keep_ratio=True,
                        preserve_z_scale=False, z_scale_factor=1.0):
    """
    Normalize 3D coordinates to [-1, 1] range.
    
    Args:
        coords: Coordinates to normalize [N, 3]
        ref_coords: Reference coordinates [M, 3], used to determine normalization parameters
        keep_ratio: Whether to preserve aspect ratio
        preserve_z_scale: Whether to preserve original z-direction scale (for sparse z-direction)
        z_scale_factor: Scaling factor for z-direction (>1 means amplifying z-direction importance)
        
    Returns:
        Normalized coordinates [N, 3]
    """
    # Calculate reference data range
    x_min, y_min, z_min = ref_coords.min(axis=0)
    x_max, y_max, z_max = ref_coords.max(axis=0)
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    # Normalize to [0, 1]
    coords = coords.astype(np.float64)
    coords[:, 0] = (coords[:, 0] - x_min) / x_range
    coords[:, 1] = (coords[:, 1] - y_min) / y_range
    coords[:, 2] = (coords[:, 2] - z_min) / z_range
    
    # Convert to [-1, 1]
    coords -= 0.5
    coords *= 2.0
    
    # Aspect ratio scaling
    if keep_ratio:
        if preserve_z_scale:
            max_xy_range = max(x_range, y_range)
            scale_x = x_range / max_xy_range
            scale_y = y_range / max_xy_range
            scale_z = z_scale_factor
        else:
            max_range = max(x_range, y_range, z_range)
            scale_x = x_range / max_range
            scale_y = y_range / max_range
            scale_z = z_range / max_range
        
        coords[:, 0] *= scale_x
        coords[:, 1] *= scale_y
        coords[:, 2] *= scale_z
    
    return coords


def main():
    parser = argparse.ArgumentParser(
        description="Prepare custom coordinate files for INR prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument("--mode", type=str, choices=["2d", "3d"], required=True,
                       help="Coordinate dimension: 2d or 3d")
    parser.add_argument("--reference", type=str, required=True,
                       help="Reference h5ad file path (used to determine normalization parameters)")
    parser.add_argument("--coords", type=str, required=True,
                       help="Input coordinate file path (.xyz or .npy format)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output npy file path")
    
    # Common arguments
    parser.add_argument("--keep_ratio", type=str, default="True",
                       help="Whether to preserve aspect ratio (default: True)")
    
    # 3D specific arguments
    parser.add_argument("--preserve_z_scale", type=str, default="False",
                       help="Whether to preserve original z-direction scale (default: False)")
    parser.add_argument("--z_scale_factor", type=float, default=1.0,
                       help="Scaling factor for z-direction (default: 1.0)")
    
    args = parser.parse_args()
    
    # Path handling
    reference_path = Path(args.reference)
    coords_path = Path(args.coords)
    output_path = Path(args.output)
    
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    if not coords_path.exists():
        raise FileNotFoundError(f"Coordinate file not found: {coords_path}")
    
    # Read reference data
    print(f"Reading reference data: {reference_path}")
    adata_ref = sc.read_h5ad(reference_path)
    
    # Read input coordinates
    print(f"Reading coordinate file: {coords_path}")
    if coords_path.suffix == ".npy":
        coords = np.load(coords_path).astype(np.float64)
    elif coords_path.suffix == ".xyz":
        coords = load_xyz(coords_path)
    else:
        raise ValueError(f"Unsupported coordinate file format: {coords_path.suffix}, supported formats: .npy or .xyz")
    
    print(f"Input coordinate shape: {coords.shape}")
    
    # Normalize
    if args.mode == "2d":
        # Extract XY coordinates only
        ref_coords = adata_ref.obsm["spatial"][:, :2].astype(np.float64)
        if coords.shape[1] < 2:
            raise ValueError(f"2D mode requires at least 2 coordinate columns, but input has {coords.shape[1]} columns")
        coords = coords[:, :2]  # Take first 2 columns only
        
        keep_ratio = args.keep_ratio.lower() in ['true', '1', 'yes']
        coords_normalized = normalize_coords_2d(
            coords, ref_coords, keep_ratio=keep_ratio
        )
        
    elif args.mode == "3d":
        # Extract XYZ coordinates
        ref_coords = adata_ref.obsm["spatial"][:, :3].astype(np.float64)
        if coords.shape[1] < 3:
            raise ValueError(f"3D mode requires at least 3 coordinate columns, but input has {coords.shape[1]} columns")
        coords = coords[:, :3]  # Take first 3 columns only
        
        keep_ratio = args.keep_ratio.lower() in ['true', '1', 'yes']
        preserve_z_scale = args.preserve_z_scale.lower() in ['true', '1', 'yes']
        
        coords_normalized = normalize_coords_3d(
            coords, ref_coords, keep_ratio=keep_ratio,
            preserve_z_scale=preserve_z_scale,
            z_scale_factor=args.z_scale_factor
        )
    
    # Check results
    print(f"\nNormalized coordinate ranges:")
    for i, dim_name in enumerate(["X", "Y", "Z"][:coords_normalized.shape[1]]):
        print(f"  {dim_name}: [{coords_normalized[:, i].min():.4f}, {coords_normalized[:, i].max():.4f}]")
    print(f"Shape: {coords_normalized.shape}")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, coords_normalized.astype(np.float32))
    print(f"\n✅ Saved {coords_normalized.shape[0]} {args.mode.upper()} coordinates to {output_path}")


if __name__ == "__main__":
    main()

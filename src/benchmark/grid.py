#!/usr/bin/env python3
"""Build regular 3D benchmark grids at 10, 20, 40 and 100 micrometres.

Input: an .npz file containing ``x``, ``y`` and ``z`` arrays (or ``xyz``),
with coordinates in millimetres by default.  The output uses the grid schema
used by the original benchmark runners::

    x_centers, y_centers, z_centers
    obs_mask, query_mask, inside
    query_near5 (optional)
    x0, y0, z0, bs, nx, ny, nz

The implementation is deliberately independent of a particular dataset.  It
builds a 2D observed-tissue hull for each z level and uses the nearest observed
z hull for z levels that contain no observed spots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull, QhullError


def _as_xyz(data) -> np.ndarray:
    if isinstance(data, dict):
        if "xyz" in data:
            xyz = np.asarray(data["xyz"], dtype=np.float64)
        elif all(k in data for k in ("x", "y", "z")):
            xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
        else:
            raise KeyError("input .npz must contain xyz or x/y/z")
    else:
        xyz = np.asarray(data, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"expected N x 3 coordinates, got {xyz.shape}")
    if not np.isfinite(xyz).all():
        raise ValueError("coordinates contain NaN/Inf")
    return xyz


def _hull_mask(points_xy: np.ndarray, xc: np.ndarray, yc: np.ndarray, margin_bins: int = 2) -> np.ndarray:
    """Boolean mask inside an expanded convex hull of observed xy points."""
    xx, yy = np.meshgrid(xc, yc)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    if len(points_xy) < 3:
        # Degenerate slice: use a small axis-aligned box around the points.
        lo = points_xy.min(0) - margin_bins * (xc[1] - xc[0])
        hi = points_xy.max(0) + margin_bins * (xc[1] - xc[0])
        inside = ((pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) &
                  (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1]))
        return inside.reshape(len(yc), len(xc))
    try:
        hull = ConvexHull(points_xy)
    except QhullError:
        lo = points_xy.min(0) - margin_bins * (xc[1] - xc[0])
        hi = points_xy.max(0) + margin_bins * (xc[1] - xc[0])
        inside = ((pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) &
                  (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1]))
        return inside.reshape(len(yc), len(xc))
    # Half-space representation: A @ p + b <= 0 inside the hull.
    inside = np.ones(len(pts), dtype=bool)
    for eq in hull.equations:
        inside &= (pts @ eq[:2] + eq[2] <= margin_bins * (xc[1] - xc[0]))
    return inside.reshape(len(yc), len(xc))


def build_grid(xyz, resolution_um: float, unit: str = "mm", margin_um: float = 0.0,
               near_bins: int = 5) -> dict:
    """Build one regular grid from coordinates."""
    xyz = _as_xyz(xyz).copy()
    if unit == "mm":
        xyz *= 1000.0
    elif unit == "um":
        pass
    else:
        raise ValueError("unit must be 'mm' or 'um'")
    res = float(resolution_um)
    margin = float(margin_um)
    lo = np.floor((xyz.min(0) - margin) / res) * res
    hi = np.ceil((xyz.max(0) + margin) / res) * res
    x0, y0, z0 = lo
    nx = int(round((hi[0] - x0) / res)) + 1
    ny = int(round((hi[1] - y0) / res)) + 1
    # Use unique z planes, preserving real section spacing.
    z_unique = np.unique(np.round(xyz[:, 2] / res) * res)
    z_unique.sort()
    xc = x0 + (np.arange(nx) + 0.5) * res
    yc = y0 + (np.arange(ny) + 0.5) * res
    ix = np.clip(((xyz[:, 0] - x0) / res).astype(np.int64), 0, nx - 1)
    iy = np.clip(((xyz[:, 1] - y0) / res).astype(np.int64), 0, ny - 1)
    iz = np.abs(xyz[:, 2][:, None] - z_unique[None, :]).argmin(1)

    obs = np.zeros((len(z_unique), ny, nx), dtype=bool)
    obs[iz, iy, ix] = True

    # Hull at observed z levels; target z levels borrow nearest observed hull.
    hulls = {}
    for k, z in enumerate(z_unique):
        sel = iz == k
        if sel.any():
            hulls[k] = _hull_mask(xyz[sel][:, :2], xc, yc)
    inside = np.zeros_like(obs)
    observed_k = np.array(sorted(hulls))
    for k in range(len(z_unique)):
        if k in hulls:
            inside[k] = hulls[k]
        elif len(observed_k):
            nk = observed_k[np.abs(observed_k - k).argmin()]
            inside[k] = hulls[nk]
    query = inside & ~obs

    # "near5": query voxels within near_bins xy bins of an observed voxel.
    near = np.zeros_like(obs)
    for k in range(len(z_unique)):
        if k in hulls:
            near[k] = binary_dilation(obs[k], iterations=int(near_bins))
    query_near5 = query & near

    return {
        "x_centers": (xc / 1000.0).astype(np.float32),
        "y_centers": (yc / 1000.0).astype(np.float32),
        "z_centers": (z_unique / 1000.0).astype(np.float32),
        "obs_mask": obs,
        "query_mask": query,
        "inside": inside,
        "query_near5": query_near5,
        "x0": float(x0 / 1000.0),
        "y0": float(y0 / 1000.0),
        "z0": float(z0 / 1000.0),
        "bs": float(res / 1000.0),
        "nx": nx,
        "ny": ny,
        "nz": len(z_unique),
        "resolution_um": res,
        "unit": "mm",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", required=True, help="input .npz with xyz or x/y/z")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resolutions", nargs="+", type=float, default=[10, 20, 40, 100])
    ap.add_argument("--unit", choices=["mm", "um"], default="mm")
    ap.add_argument("--margin-um", type=float, default=0.0)
    ap.add_argument("--near-bins", type=int, default=5)
    args = ap.parse_args()

    data = np.load(args.cells, allow_pickle=True)
    data = {k: data[k] for k in data.files}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for res in args.resolutions:
        g = build_grid(data, res, unit=args.unit, margin_um=args.margin_um,
                       near_bins=args.near_bins)
        p = out / f"grid_{int(res)}um.npz"
        np.savez_compressed(p, **g)
        row = {
            "resolution_um": int(res),
            "path": str(p),
            "shape": [int(g["nz"]), int(g["ny"]), int(g["nx"])],
            "observed": int(g["obs_mask"].sum()),
            "query": int(g["query_mask"].sum()),
            "query_near5": int(g["query_near5"].sum()),
        }
        summary.append(row)
        print(json.dumps(row))
    (out / "grid_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

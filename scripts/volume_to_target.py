#!/usr/bin/env python3
"""Convert a full-volume method output (NTF/Spateo) to a target-only prediction.

The volume is a regular 50 um grid (or a generated 3D point cloud).  The
target rows are the observed bins of the held-out sections from
canonical/target_gt.npz.  Each target is mapped to the nearest volume point.

Usage:
    python scripts/volume_to_target.py \
      --volume NTF/NTF_50um_uniform.h5ad \
      --target-gt canonical/target_gt.npz \
      --scenario M1a --method NTF \
      --out predictions/M1a/NTF.npz
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.sparse import issparse
from scipy.spatial import cKDTree


def volume_coords(a: ad.AnnData) -> tuple[np.ndarray, str]:
    for key in ("spatial", "X_aligned"):
        if key in a.obsm:
            arr = np.asarray(a.obsm[key])
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[:, :3].astype(np.float64), f"obsm[{key}]"
            if arr.ndim == 2 and arr.shape[1] == 2:
                for zkey in ("z", "z_um", "z_mm"):
                    if zkey in a.obs:
                        return np.column_stack([arr, np.asarray(a.obs[zkey])]).astype(np.float64), f"obsm[{key}]+obs[{zkey}]"
    for cols in (("x", "y", "z"), ("x_mm", "y_mm", "z_mm"), ("x_um", "y_um", "z_um")):
        if all(c in a.obs for c in cols):
            return np.column_stack([np.asarray(a.obs[c]) for c in cols]).astype(np.float64), f"obs{list(cols)}"
    raise ValueError(f"cannot find 3D coordinates in {a.obs.columns.tolist()} / {list(a.obsm)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True)
    ap.add_argument("--target-gt", required=True)
    ap.add_argument("--scenario", default="M1a")
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-median-um", type=float, default=100.0)
    args = ap.parse_args()

    vol = ad.read_h5ad(args.volume)
    vcoords, csrc = volume_coords(vol)
    tg = np.load(args.target_gt, allow_pickle=False)
    scen = np.asarray(tg["scenario"]).astype(str)
    rows = np.flatnonzero(scen == args.scenario)
    if rows.size == 0:
        raise ValueError(f"no target rows for {args.scenario}")
    tcoords = np.column_stack([np.asarray(tg[k]) for k in ("x", "y", "z")]).astype(np.float64)[rows]

    # Target coords are mm; volumes may be mm or um. Choose the scale with the
    # smallest nearest-neighbour distance, then sanity-check against the grid.
    best = None
    for scale in (1.0, 1000.0, 0.001):
        tree = cKDTree(vcoords)
        d, idx = tree.query(tcoords * scale)
        med = float(np.median(d))
        if best is None or med < best[0]:
            best = (med, scale, d, idx)
    med, scale, dist, idx = best
    if med > args.max_median_um:
        raise RuntimeError(
            f"nearest-neighbour median {med:.2f} (coordinate units) exceeds "
            f"{args.max_median_um}; check volume/target coordinate scales")
    print(f"[map] volume={vcoords.shape} target={tcoords.shape} coord={csrc} "
          f"scale={scale} median={med:.3f} p90={np.percentile(dist,90):.3f}")

    X = vol.X.toarray() if issparse(vol.X) else np.asarray(vol.X)
    X = np.asarray(X, dtype=np.float32)
    vgenes = np.asarray(vol.var_names).astype(str)
    tgenes = np.asarray(tg["genes"]).astype(str)
    if not np.array_equal(vgenes, tgenes):
        pos = {g: i for i, g in enumerate(vgenes)}
        missing = [g for g in tgenes if g not in pos]
        if missing:
            raise ValueError(f"volume is missing {len(missing)} target genes, e.g. {missing[:5]}")
        X = X[:, [pos[g] for g in tgenes]]
    pred = X[idx]
    if not np.isfinite(pred).all():
        raise ValueError("non-finite values in mapped prediction")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        pred=pred.astype(np.float32),
        row_index=rows,
        bin_id=np.asarray(tg["bin_id"])[rows].astype(str),
        section=np.asarray(tg["section"])[rows].astype(str),
        x=np.asarray(tg["x"])[rows],
        y=np.asarray(tg["y"])[rows],
        method=np.asarray(args.method),
        scenario=np.asarray(args.scenario),
        genes=tgenes,
    )
    print(f"[write] {out} pred={pred.shape} median_nn={med:.3f}")


if __name__ == "__main__":
    main()

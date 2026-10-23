#!/usr/bin/env python3
"""Convert a saved point-cloud prediction (Spateo held-out diagnostics) to the
target-only prediction contract.

Input npz: pred (#points, n_genes), x/y/z in mm, optional row_index.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points-npz", required=True)
    ap.add_argument("--target-gt", required=True)
    ap.add_argument("--scenario", default="M1a")
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.points_npz, allow_pickle=False)
    pred_all = np.asarray(z["pred"], dtype=np.float32)
    vcoords = np.column_stack([np.asarray(z[k], dtype=np.float64) for k in ("x", "y", "z")])
    tg = np.load(args.target_gt, allow_pickle=False)
    scen = np.asarray(tg["scenario"]).astype(str)
    rows = np.flatnonzero(scen == args.scenario)
    tcoords = np.column_stack([np.asarray(tg[k]) for k in ("x", "y", "z")]).astype(np.float64)[rows]

    best = None
    for scale in (1.0, 1000.0, 0.001):
        tree = cKDTree(vcoords)
        d, idx = tree.query(tcoords * scale)
        med = float(np.median(d))
        if best is None or med < best[0]:
            best = (med, scale, d, idx)
    med, scale, dist, idx = best
    if med > 100.0:
        raise RuntimeError(f"nearest-neighbour median {med:.2f} is too large")
    print(f"[map] points={vcoords.shape} target={tcoords.shape} scale={scale} "
          f"median={med:.3f} p90={np.percentile(dist,90):.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        pred=pred_all[idx],
        row_index=rows,
        bin_id=np.asarray(tg["bin_id"])[rows].astype(str),
        section=np.asarray(tg["section"])[rows].astype(str),
        x=np.asarray(tg["x"])[rows],
        y=np.asarray(tg["y"])[rows],
        method=np.asarray(args.method),
        scenario=np.asarray(args.scenario),
        genes=np.asarray(tg["genes"]).astype(str),
    )
    print(f"[write] {out} pred={pred_all[idx].shape}")


if __name__ == "__main__":
    main()

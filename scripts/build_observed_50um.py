#!/usr/bin/env python3
"""Rebuild the MERFISH 5-slice 50 um observed input from committed shards.

The repository stores five small per-slice shards:

    data/merfish_5slice/observed_50um_S1.h5ad ... S5.h5ad

Each shard is a lossless float32 CSR AnnData subset (one section, 50 um bins,
1122 genes).  This script concatenates S1..S5 into the single
``observed_50um.h5ad`` consumed by ``src/canonical/merfish_5slice.py``.

Usage:
    python scripts/build_observed_50um.py \
        --shards-dir data/merfish_5slice \
        --out observed_50um.h5ad

    # optional exactness check against the full HPC copy
    python scripts/build_observed_50um.py \
        --shards-dir data/merfish_5slice \
        --out /tmp/observed_50um.h5ad \
        --check /path/to/Workspace/merfish_test/observed_50um.h5ad
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack

SLICES = [
    "C57BL6J-3.006",
    "C57BL6J-3.007",
    "C57BL6J-3.008",
    "C57BL6J-3.009",
    "C57BL6J-3.010",
]
SHARD_ONLY_UNS = ("original_slice", "slice_order")


def read_shards(shards_dir: Path) -> list[ad.AnnData]:
    shards = []
    for i in range(1, 6):
        p = shards_dir / f"observed_50um_S{i}.h5ad"
        if not p.exists():
            raise FileNotFoundError(p)
        a = ad.read_h5ad(p)
        if a.n_obs == 0:
            raise ValueError(f"empty shard: {p}")
        shards.append(a)
    return shards


def merge(shards: list[ad.AnnData]) -> ad.AnnData:
    X = vstack([s.X for s in shards], format="csr").astype(np.float32)
    obs = pd.concat([s.obs for s in shards], axis=0)
    var = shards[0].var.copy()
    obsm = {"X_aligned": np.vstack([np.asarray(s.obsm["X_aligned"]) for s in shards])}
    uns = {k: v for k, v in shards[0].uns.items() if k not in SHARD_ONLY_UNS}
    out = ad.AnnData(X=X, obs=obs, var=var)
    out.obs_names = np.concatenate([s.obs_names.to_numpy() for s in shards])
    for k, v in obsm.items():
        out.obsm[k] = v
    for k, v in uns.items():
        out.uns[k] = v
    return out


def check(out: ad.AnnData, ref_path: Path) -> dict:
    ref = ad.read_h5ad(ref_path)
    if out.shape != ref.shape:
        raise ValueError(f"shape {out.shape} != reference {ref.shape}")
    if not np.array_equal(out.obs_names.to_numpy(), ref.obs_names.to_numpy()):
        raise ValueError("obs_names differ")
    if not np.array_equal(out.var_names.to_numpy(), ref.var_names.to_numpy()):
        raise ValueError("var_names differ")
    X = out.X.toarray() if hasattr(out.X, "toarray") else np.asarray(out.X)
    Xref = ref.X.toarray() if hasattr(ref.X, "toarray") else np.asarray(ref.X)
    max_x = float(np.abs(X.astype(np.float32) - Xref.astype(np.float32)).max())
    if max_x != 0.0:
        raise ValueError(f"X differs from reference by {max_x}")
    max_obsm = float(
        np.abs(
            np.asarray(out.obsm["X_aligned"], dtype=np.float32)
            - np.asarray(ref.obsm["X_aligned"], dtype=np.float32)
        ).max()
    )
    if max_obsm != 0.0:
        raise ValueError(f"X_aligned differs by {max_obsm}")
    common = [c for c in out.obs.columns if c in ref.obs.columns]
    for c in common:
        a = out.obs[c]
        b = ref.obs[c]
        if isinstance(a.dtype, pd.CategoricalDtype) and isinstance(b.dtype, pd.CategoricalDtype):
            if not np.array_equal(a.astype(str).to_numpy(), b.astype(str).to_numpy()):
                raise ValueError(f"obs[{c}] differs")
        elif not np.array_equal(np.asarray(a), np.asarray(b)):
            raise ValueError(f"obs[{c}] differs")
    return {
        "shape": list(out.shape),
        "max_abs_X_diff": max_x,
        "max_abs_X_aligned_diff": max_obsm,
        "obs_columns_checked": common,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", default="data/merfish_5slice")
    ap.add_argument("--out", default="observed_50um.h5ad")
    ap.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    ap.add_argument("--compression-opts", type=int, default=9)
    ap.add_argument("--check", default=None, help="reference observed_50um.h5ad")
    args = ap.parse_args()

    shards_dir = Path(args.shards_dir)
    shards = read_shards(shards_dir)
    out = merge(shards)
    out.uns["source_shards"] = [f"observed_50um_S{i}.h5ad" for i in range(1, 6)]
    out.uns["slice_order"] = SLICES

    report = {
        "shards": [str(shards_dir / f"observed_50um_S{i}.h5ad") for i in range(1, 6)],
        "n_obs": int(out.n_obs),
        "n_vars": int(out.n_vars),
        "nnz": int(out.X.nnz),
        "slices": {s: int((out.obs["slice"].astype(str) == s).sum()) for s in SLICES},
    }
    if args.check:
        report["check"] = check(out, Path(args.check))
    print(json.dumps(report, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comp = None if args.compression == "none" else args.compression
    out.write_h5ad(
        out_path,
        compression=comp,
        compression_opts=args.compression_opts if comp else None,
    )
    print(f"[write] {out_path} {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

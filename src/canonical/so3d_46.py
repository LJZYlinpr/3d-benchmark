#!/usr/bin/env python
"""Build So3D canonical held-out target datasets (CPU only).

Input : Data/benchmark_data/so3d/ge5_slices/h5ad/*.h5ad
Output: Workspace/so3d_bench/canonical/<dataset>/{visible.h5ad,target.npz,meta.json}

Rules:
  - sorted unique slices; target = sorted_slices[n // 2] (interior, never first/last)
  - visible = all other slices, source row order preserved
  - X copied as provided (no renormalization); target X written only for evaluation
  - no target X is read/used when building or validating visible data
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback

import anndata as ad
import numpy as np
import scipy.sparse as sp


def log(fp, msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    fp.write(line + "\n")
    fp.flush()


def x_note(X):
    d = X.data
    return {
        "storage": "scipy.sparse.csr_matrix",
        "dtype": str(X.dtype),
        "implicit_zeros": True,
        "nonzero_fraction": float(X.nnz / np.prod(X.shape)),
        "nonzero_min": float(d.min()) if d.size else 0.0,
        "nonzero_max": float(d.max()) if d.size else 0.0,
        "nonzero_mean": float(d.mean()) if d.size else 0.0,
        "note": (
            "provided normalized expression (log1p-like scale, non-negative); "
            "sparse with implicit zeros; NOT renormalized, rescaled, or filtered by this builder"
        ),
    }


def build_one(src_path, out_root, manifest_rows, logfp):
    dataset = os.path.splitext(os.path.basename(src_path))[0]
    outdir = os.path.join(out_root, dataset)
    os.makedirs(outdir, exist_ok=True)

    a = ad.read_h5ad(src_path)
    slices = np.sort(np.unique(a.obs["slice"].to_numpy().astype(np.int64)))
    n_slices = int(len(slices))
    if n_slices < 5:
        raise ValueError(f"{dataset}: only {n_slices} slices (<5)")
    ti = n_slices // 2
    target_slice = int(slices[ti])
    visible_slices = [int(s) for s in slices if int(s) != target_slice]
    if target_slice in (int(slices[0]), int(slices[-1])):
        raise ValueError(f"{dataset}: target slice {target_slice} is first/last")

    sl = a.obs["slice"].to_numpy().astype(np.int64)
    z = a.obs["z"].to_numpy().astype(np.float32)
    coords = np.asarray(a.obsm["spatial"])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{dataset}: obsm['spatial'] shape {coords.shape} != (n,3)")
    coords = coords.astype(np.float32, copy=False)

    # split by source position (order preserved)
    pos = np.arange(a.n_obs)
    vis_pos = pos[np.isin(sl, visible_slices)]
    tgt_pos = pos[sl == target_slice]
    if vis_pos.size == 0 or tgt_pos.size == 0:
        raise ValueError(f"{dataset}: empty visible or target set")
    vis_names = a.obs_names[vis_pos].to_numpy()
    tgt_names = a.obs_names[tgt_pos].to_numpy()
    if len(np.intersect1d(vis_names, tgt_names)):
        raise ValueError(f"{dataset}: visible/target obs_names overlap")
    if not (a.obs_names.is_unique):
        raise ValueError(f"{dataset}: source obs_names not unique")

    # ---- build visible (never touches target X) ----
    obs_vis = a.obs.iloc[vis_pos].copy()
    obs_vis["slice"] = obs_vis["slice"].astype(np.int64)
    vis = ad.AnnData(
        X=a.X[vis_pos].copy(),
        obs=obs_vis,
        var=a.var.copy(),
        obsm={"spatial": coords[vis_pos].copy()},
    )
    if "so3d" in a.uns:
        vis.uns["so3d"] = dict(a.uns["so3d"])
    vis.uns["so3d_canonical"] = {
        "dataset": dataset,
        "target_slice": target_slice,
        "visible_slices": visible_slices,
    }

    # ---- target (evaluation only) ----
    tgt_X = a.X[tgt_pos].toarray().astype(np.float32) if sp.issparse(a.X) else np.asarray(a.X[tgt_pos], dtype=np.float32)
    tgt = {
        "X": tgt_X,
        "coords": coords[tgt_pos].astype(np.float32),
        "obs_names": tgt_names.astype(str),
        "slice": np.full(tgt_pos.size, target_slice, dtype=np.int64),
        "z": z[tgt_pos].astype(np.float32),
    }

    meta = {
        "dataset": dataset,
        "source_h5ad": os.path.abspath(src_path),
        "n_slices": n_slices,
        "slices": [int(s) for s in slices],
        "target_slice": target_slice,
        "target_index": int(ti),
        "visible_slices": visible_slices,
        "n_visible": int(vis_pos.size),
        "n_target": int(tgt_pos.size),
        "n_genes": int(a.n_vars),
        "genes": [str(g) for g in a.var_names],
        "z_range": {
            "min": float(z.min()),
            "max": float(z.max()),
            "target_z": float(np.unique(z[tgt_pos])[0]),
            "visible_min": float(z[vis_pos].min()),
            "visible_max": float(z[vis_pos].max()),
        },
        "coord_columns": ["x", "y", "z"],
        "target_in_visible": False,
        "X": x_note(a.X),
        "target_X_storage": "dense float32 in target.npz (implicit zeros materialized; evaluation only)",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # ---- write ----
    vis.write_h5ad(os.path.join(outdir, "visible.h5ad"), compression="gzip")
    np.savez_compressed(os.path.join(outdir, "target.npz"), **tgt)
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    # ---- validate (fresh read-back; target X is only compared to source) ----
    errs = []
    v2 = ad.read_h5ad(os.path.join(outdir, "visible.h5ad"))
    t2 = np.load(os.path.join(outdir, "target.npz"), allow_pickle=False)

    for k in ("X", "coords", "obs_names", "slice", "z"):
        if k not in t2.files:
            errs.append(f"target.npz missing key {k}")
    if errs:
        raise ValueError(f"{dataset}: {'; '.join(errs)}")

    # genes / shape
    if list(v2.var_names) != list(a.var_names):
        errs.append("visible var_names != source var_names")
    if v2.n_vars != a.n_vars:
        errs.append("visible n_vars changed")
    if t2["X"].shape != (tgt_pos.size, a.n_vars):
        errs.append(f"target X shape {t2['X'].shape} != {(tgt_pos.size, a.n_vars)}")
    if t2["coords"].shape != (tgt_pos.size, 3):
        errs.append("target coords shape != (n,3)")
    for k in ("z", "slice", "obs_names"):
        if t2[k].shape != (tgt_pos.size,):
            errs.append(f"target {k} shape != (n,)")

    # overlap / uniqueness / coverage
    if len(np.intersect1d(v2.obs_names.to_numpy(), t2["obs_names"])):
        errs.append("visible/target obs_names overlap")
    if len(np.unique(t2["obs_names"])) != tgt_pos.size:
        errs.append("target obs_names not unique")
    if len(np.unique(v2.obs_names.to_numpy())) != vis_pos.size:
        errs.append("visible obs_names not unique")
    if vis_pos.size + tgt_pos.size != a.n_obs:
        errs.append("visible+target != source n_obs")
    if set(v2.obs_names.to_numpy()) | set(t2["obs_names"].tolist()) != set(a.obs_names.to_numpy()):
        errs.append("visible+target obs set != source obs set")

    # finiteness
    if not np.isfinite(v2.X.data).all():
        errs.append("visible X non-finite")
    if not np.isfinite(t2["X"]).all():
        errs.append("target X non-finite")
    if not np.isfinite(t2["coords"]).all():
        errs.append("target coords non-finite")
    if not np.isfinite(v2.obsm["spatial"]).all():
        errs.append("visible coords non-finite")
    if not np.isfinite(t2["z"]).all():
        errs.append("target z non-finite")

    # slices
    if set(np.unique(v2.obs["slice"].to_numpy())) != set(visible_slices):
        errs.append("visible slice values wrong")
    if not np.all(t2["slice"] == target_slice):
        errs.append("target slice values wrong")
    if target_slice in set(np.unique(v2.obs["slice"].to_numpy())):
        errs.append("target slice present in visible")
    for nb in (slices[ti - 1], slices[ti + 1]):
        if int(nb) not in set(np.unique(v2.obs["slice"].to_numpy())):
            errs.append(f"neighbor slice {int(nb)} not visible")
    if ti in (0, n_slices - 1):
        errs.append("target not interior")

    # row alignment to source (by obs_name)
    src_index = {str(n): i for i, n in enumerate(a.obs_names)}
    tgt_src = np.array([src_index[str(n)] for n in t2["obs_names"]], dtype=np.int64)
    if not np.array_equal(tgt_src, tgt_pos):
        errs.append("target row order does not match source order")
    if not np.array_equal(t2["coords"], coords[tgt_pos]):
        errs.append("target coords != source coords")
    if not np.array_equal(t2["z"], z[tgt_pos]):
        errs.append("target z != source z")
    src_tX = a.X[tgt_pos].toarray().astype(np.float32) if sp.issparse(a.X) else np.asarray(a.X[tgt_pos], dtype=np.float32)
    if not np.array_equal(t2["X"], src_tX):
        errs.append("target X != source X")

    vis_src = np.array([src_index[str(n)] for n in v2.obs_names.to_numpy()], dtype=np.int64)
    if not np.array_equal(vis_src, vis_pos):
        errs.append("visible row order does not match source order")
    if not np.array_equal(v2.obsm["spatial"], coords[vis_pos]):
        errs.append("visible coords != source coords")
    src_vX = a.X[vis_pos]
    if sp.issparse(src_vX):
        if not (np.array_equal(v2.X.indptr, src_vX.indptr)
                and np.array_equal(v2.X.indices, src_vX.indices)
                and np.array_equal(v2.X.data, src_vX.data)):
            errs.append("visible X != source X")
    elif not np.array_equal(np.asarray(v2.X), np.asarray(src_vX)):
        errs.append("visible X != source X")

    if errs:
        raise ValueError(f"{dataset}: validation failed: {'; '.join(errs)}")

    xmin = meta["X"]["nonzero_min"]
    xmax = meta["X"]["nonzero_max"]
    manifest_rows.append([
        dataset, n_slices, target_slice, ti, len(visible_slices),
        int(vis_pos.size), int(tgt_pos.size), int(a.n_vars),
        f"{meta['z_range']['min']:.6g}", f"{meta['z_range']['max']:.6g}",
        f"{meta['z_range']['target_z']:.6g}",
        f"{xmin:.6g}", f"{xmax:.6g}", "PASS",
        os.path.join("canonical", dataset, "visible.h5ad"),
        os.path.join("canonical", dataset, "target.npz"),
        os.path.join("canonical", dataset, "meta.json"),
    ])
    return meta


def make_contract(path):
    text = """# So3D canonical runner contract (v1)

Dataset: one held-out middle slice per dataset. Target expression is evaluation-only;
never read `target.npz` inside a method (no GT fitting, calibration, or normalization stats).

## Inputs (read-only)
- `canonical/<dataset>/visible.h5ad` — all spots from non-target slices, all genes.
  `X` is provided normalized expression (sparse, log1p-like; use as-is, do not renormalize),
  `obsm['spatial']` is xyz, `obs['slice']`, `obs['z']`.
- `canonical/<dataset>/meta.json` — `genes` in var order, `target_slice`, `visible_slices`,
  n_visible/n_target, z range, X note.

## Prediction output
`predictions/<dataset>/<method>.npz` with:
- `pred` or `X` — float array `(n_target, n_genes)`, non-negative, all finite.
  Use `pred` when both are present.
- `obs_names` — array of str, exactly `target.npz['obs_names']` in the same order.
- `coords` — `(n_target, 3)` float, exactly `target.npz['coords']` in the same order.
- `slice` — `(n_target,)` int, all equal to `meta.json['target_slice']`.

Row order must equal `canonical/<dataset>/target.npz` (`X`, `coords`, `obs_names`, `slice`, `z`).
Gene order must equal `meta.json['genes']` / visible `var_names`.
Write `predictions/<dataset>/<method>.status.json` with `{"status": "DONE"|"FAILED"|"UNSUPPORTED", "reason": ...}`.
Never fabricate rows: mark failures explicitly.
"""
    with open(path, "w") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    data_dir = args.data or os.path.join(
        os.path.dirname(root), "..", "Data", "benchmark_data", "so3d", "ge5_slices", "h5ad"
    )
    data_dir = os.path.abspath(data_dir)
    out_root = os.path.join(root, "canonical")
    log_dir = os.path.join(root, "logs")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(data_dir, "*.h5ad")))
    log_path = os.path.join(log_dir, "prep.log")
    with open(log_path, "a") as logfp:
        log(logfp, f"prep start: {len(files)} input files from {data_dir}")
        manifest_rows, fails = [], []
        for p in files:
            ds = os.path.splitext(os.path.basename(p))[0]
            try:
                m = build_one(p, out_root, manifest_rows, logfp)
                log(logfp, f"OK {ds}: n_slices={m['n_slices']} target_slice={m['target_slice']} "
                           f"visible={m['n_visible']} target={m['n_target']} genes={m['n_genes']}")
            except Exception as e:
                fails.append((ds, str(e)))
                log(logfp, f"FAIL {ds}: {e}")
                log(logfp, traceback.format_exc())
                logfp.flush()

        if fails:
            log(logfp, f"ABORT: {len(fails)} failures; MANIFEST/PREP_READY not written")
            for ds, e in fails:
                log(logfp, f"  {ds}: {e}")
            return 1

        header = ["dataset", "n_slices", "target_slice", "target_index", "n_visible_slices",
                  "n_visible", "n_target", "n_genes", "z_min", "z_max", "target_z",
                  "X_nonzero_min", "X_nonzero_max", "validation",
                  "visible_path", "target_path", "meta_path"]
        manifest_path = os.path.join(out_root, "MANIFEST.tsv")
        with open(manifest_path, "w") as f:
            f.write("\t".join(header) + "\n")
            for row in manifest_rows:
                f.write("\t".join(str(v) for v in row) + "\n")

        make_contract(os.path.join(root, "FORMAT_CONTRACT.md"))
        with open(os.path.join(root, "PREP_READY"), "w") as f:
            f.write(f"ready_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
            f.write(f"n_datasets={len(manifest_rows)}\n")
            f.write(f"total_visible={sum(r[5] for r in manifest_rows)}\n")
            f.write(f"total_target={sum(r[6] for r in manifest_rows)}\n")
            f.write("canonical=canonical/<dataset>/{visible.h5ad,target.npz,meta.json}\n")
            f.write("manifest=canonical/MANIFEST.tsv\n")
            f.write("contract=FORMAT_CONTRACT.md\n")
        log(logfp, f"PREP_READY: {len(manifest_rows)}/{len(files)} datasets validated; "
                   f"total visible={sum(r[5] for r in manifest_rows)} target={sum(r[6] for r in manifest_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

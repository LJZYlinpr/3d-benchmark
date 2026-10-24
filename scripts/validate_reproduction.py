#!/usr/bin/env python3
"""Strict validation of a fresh MERFISH 5-slice M1a reproduction.

Checks:
  1. data/merfish_5slice/SHA256SUMS (shards + grid)
  2. canonical bins_all X and target_gt array hashes
  3. every prediction: shape, finite, row_index alignment
  4. module-mean gene PCC against manifests/merfish5_m1a_expected.json

Exit code 0 only if all required checks pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_arrays(d: np.lib.npyio.NpzFile) -> str:
    h = hashlib.sha256()
    for k in d.files:
        h.update(np.ascontiguousarray(d[k]).tobytes())
    return h.hexdigest()


def pcc_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    sa = np.sqrt((a * a).sum(axis=0))
    sb = np.sqrt((b * b).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b).sum(axis=0) / (sa * sb)
    r[(sa == 0) | (sb == 0)] = np.nan
    return r


def module_mean_gene_pcc(pred: np.ndarray, gt: np.ndarray, modules: list[dict]) -> float:
    r = pcc_cols(pred.astype(np.float64), gt.astype(np.float64))
    medians = []
    for m in modules:
        rr = r[np.asarray(m["gene_idx"], dtype=int)]
        rr = rr[np.isfinite(rr)]
        if rr.size:
            medians.append(float(np.median(rr)))
    return float(np.mean(medians))


def check_data(repo: Path) -> None:
    ddir = repo / "data" / "merfish_5slice"
    sums = ddir / "SHA256SUMS"
    if not sums.exists():
        raise FileNotFoundError(sums)
    for line in sums.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        expected, name = line.split(None, 1)
        p = ddir / name.lstrip("*")
        if not p.exists():
            raise FileNotFoundError(p)
        got = sha256_file(p)
        if got != expected:
            raise ValueError(f"data hash mismatch: {p} {got} != {expected}")
    print(f"[ok] data shards + grid ({len([l for l in sums.read_text().splitlines() if l.strip()])} files)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--modules", default=None)
    ap.add_argument("--expected", default=None)
    ap.add_argument("--scenario", default="M1a")
    ap.add_argument("--allow-subset", action="store_true",
                    help="do not fail when a method prediction is absent")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    canon = Path(args.canonical).resolve()
    pred_dir = Path(args.pred_dir).resolve()
    modules_path = Path(args.modules) if args.modules else repo / "manifests" / "merfish_5slice_modules.json"
    expected_path = Path(args.expected) if args.expected else repo / "manifests" / "merfish5_m1a_expected.json"
    expected = json.loads(expected_path.read_text())

    check_data(repo)

    bins = ad.read_h5ad(canon / "bins_all.h5ad")
    X = np.asarray(bins.X.toarray() if hasattr(bins.X, "toarray") else bins.X, dtype=np.float32)
    bins_sha = hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()
    if bins_sha != expected["canonical_bins_sha256"]:
        raise ValueError(f"canonical bins X hash mismatch: {bins_sha}")
    print(f"[ok] canonical bins_all X {bins_sha[:16]}…")

    tg = np.load(canon / "target_gt.npz", allow_pickle=False)
    tg_sha = sha256_arrays(tg)
    if tg_sha != expected["target_gt_sha256"]:
        raise ValueError(f"target_gt hash mismatch: {tg_sha}")
    print(f"[ok] target_gt {tg_sha[:16]}…")

    scen = np.asarray(tg["scenario"]).astype(str)
    rows = np.flatnonzero(scen == args.scenario)
    if rows.size == 0:
        raise ValueError(f"scenario {args.scenario} not in target_gt")
    gt = np.asarray(tg["gt"][rows], dtype=np.float64)
    modules = json.loads(modules_path.read_text())["modules"]
    tgenes = np.asarray(tg["genes"]).astype(str)

    failures = 0
    for method, spec in expected["methods"].items():
        p = pred_dir / f"{method}.npz"
        if not p.exists():
            msg = f"[missing] {method}: {p}"
            print(msg)
            if not args.allow_subset:
                failures += 1
            continue
        z = np.load(p, allow_pickle=False)
        pred = np.asarray(z["pred"], dtype=np.float64)
        ri = np.asarray(z["row_index"])
        if pred.shape != gt.shape:
            print(f"[FAIL] {method}: shape {pred.shape} != {gt.shape}")
            failures += 1
            continue
        if not np.array_equal(ri, rows):
            print(f"[FAIL] {method}: row_index does not match {args.scenario}")
            failures += 1
            continue
        if not np.isfinite(pred).all():
            print(f"[FAIL] {method}: non-finite predictions")
            failures += 1
            continue
        if "genes" in z:
            pgenes = np.asarray(z["genes"]).astype(str)
            if not np.array_equal(pgenes, tgenes):
                pos = {g: i for i, g in enumerate(pgenes)}
                pred = pred[:, [pos[g] for g in tgenes]]
        score = module_mean_gene_pcc(pred, gt, modules)
        ref = float(spec["module_pcc_mean"])
        tol = float(spec["tol"])
        ok = abs(score - ref) <= tol
        status = "ok" if ok else "FAIL"
        print(f"[{status}] {method:16s} module_pcc={score:.6f} ref={ref:.6f} "
              f"diff={score-ref:+.6f} tol={tol:g}")
        if not ok:
            failures += 1

    if failures:
        print(f"[FAIL] {failures} validation failure(s)")
        return 1
    print(f"[ok] {args.scenario} reproduction validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())

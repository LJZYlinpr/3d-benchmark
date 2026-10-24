#!/usr/bin/env python3
"""Core expression metrics for MERFISH 5-slice target-only predictions.

Computes the module-mean gene PCC, pooled PCC, RMSE and MAE per method/scenario
from canonical/target_gt.npz and predictions/<scenario>/<method>.npz.  This is
the metric subset used in the reproduction reports; the full paper metric suite
is separate.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def pcc_vector(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    a = pred - pred.mean(axis=0, keepdims=True)
    b = gt - gt.mean(axis=0, keepdims=True)
    sa = np.sqrt((a * a).sum(axis=0))
    sb = np.sqrt((b * b).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b).sum(axis=0) / (sa * sb)
    r[(sa == 0) | (sb == 0)] = np.nan
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--predictions-root", required=True)
    ap.add_argument("--modules", required=True)
    ap.add_argument("--scenarios", default=None, help="comma list; default all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tg = np.load(Path(args.canonical) / "target_gt.npz", allow_pickle=False)
    scen = np.asarray(tg["scenario"]).astype(str)
    tgenes = np.asarray(tg["genes"]).astype(str)
    import json
    mods = json.load(open(args.modules))["modules"]
    for m in mods:
        idx = np.asarray(m["gene_idx"], dtype=int)
        if not np.array_equal(tgenes[idx], np.asarray(m["genes"]).astype(str)):
            raise ValueError(f"module gene indices do not match target genes: {m['name']}")

    scenarios = [s for s in (args.scenarios.split(",") if args.scenarios else np.unique(scen))]
    rows_out = []
    for s in scenarios:
        rows = np.flatnonzero(scen == s)
        if rows.size == 0:
            continue
        gt = np.asarray(tg["gt"][rows], dtype=np.float64)
        pred_dir = Path(args.predictions_root) / s
        files = sorted(pred_dir.glob("*.npz"))
        for p in files:
            z = np.load(p, allow_pickle=False)
            pred = np.asarray(z["pred"], dtype=np.float64)
            ri = np.asarray(z["row_index"])
            method = p.stem
            if pred.shape != gt.shape or not np.array_equal(ri, rows):
                print(f"[skip] {method}/{s}: contract mismatch")
                continue
            if "genes" in z:
                pgenes = np.asarray(z["genes"]).astype(str)
                if not np.array_equal(pgenes, tgenes):
                    pos = {g: i for i, g in enumerate(pgenes)}
                    pred = pred[:, [pos[g] for g in tgenes]]
            r = pcc_vector(pred, gt)
            module_medians = []
            for m in mods:
                rr = r[np.asarray(m["gene_idx"], dtype=int)]
                rr = rr[np.isfinite(rr)]
                if rr.size:
                    module_medians.append(float(np.median(rr)))
            rms = float(np.sqrt(np.mean((pred - gt) ** 2)))
            mae = float(np.mean(np.abs(pred - gt)))
            a = pred.ravel() - pred.mean()
            b = gt.ravel() - gt.mean()
            denom = np.sqrt((a * a).sum() * (b * b).sum())
            pooled = float((a * b).sum() / denom) if denom > 0 else float("nan")
            rows_out.append({
                "method": method, "scenario": s,
                "n_rows": int(rows.size),
                "gene_pcc_median": float(np.nanmedian(r)) if np.isfinite(r).any() else np.nan,
                "module_pcc_mean": float(np.mean(module_medians)) if module_medians else np.nan,
                "pooled_pcc": pooled, "rmse": rms, "mae": mae,
            })
    df = pd.DataFrame(rows_out)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[write] {out} rows={len(df)}")
    if len(df):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()

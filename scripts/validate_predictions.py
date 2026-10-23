#!/usr/bin/env python3
"""Check target-only prediction files against canonical/target_gt.npz."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--scenario", default="M1a")
    args = ap.parse_args()

    tg = np.load(Path(args.canonical) / "target_gt.npz", allow_pickle=False)
    scen = np.asarray(tg["scenario"]).astype(str)
    rows = np.flatnonzero(scen == args.scenario)
    genes = np.asarray(tg["genes"]).astype(str)
    if rows.size == 0:
        raise SystemExit(f"no target rows for {args.scenario}")

    files = sorted(Path(args.predictions).glob("*.npz"))
    if not files:
        raise SystemExit(f"no prediction npz under {args.predictions}")

    failed = 0
    print(f"{'method':16s} {'shape':>16s} {'finite':>7s} {'rows_ok':>8s} {'mean':>10s}")
    for p in files:
        z = np.load(p, allow_pickle=False)
        name = p.stem
        if "pred" not in z or "row_index" not in z:
            print(f"{name:16s} MISSING pred/row_index"); failed += 1; continue
        pred = np.asarray(z["pred"])
        ri = np.asarray(z["row_index"])
        ok_shape = pred.shape == (rows.size, genes.size)
        ok_rows = np.array_equal(ri, rows)
        finite = bool(np.isfinite(pred).all())
        if "genes" in z:
            ok_genes = np.array_equal(np.asarray(z["genes"]).astype(str), genes)
        else:
            ok_genes = None
        mean = float(np.nanmean(pred))
        print(f"{name:16s} {str(pred.shape):>16s} {str(finite):>7s} {str(ok_rows):>8s} {mean:10.4f}"
              + ("" if ok_genes is not False else "  GENES_MISMATCH"))
        if not (ok_shape and ok_rows and finite) or ok_genes is False:
            failed += 1

    if failed:
        raise SystemExit(f"{failed} prediction file(s) failed the contract")
    print(f"[ok] {len(files)} prediction files")


if __name__ == "__main__":
    main()

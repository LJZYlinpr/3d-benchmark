#!/usr/bin/env python
"""Diagnostic only: nearest-visible-bin reference predictions.

This is NOT a benchmark method.  It interpolates each target bin with the
expression of the nearest observed bin of a visible slice (median gap between
adjacent slices is ~0.19 mm), and gives a floor/reference for how much signal
is available at the missing slices without any model.  Written to
diagnostics/nn_reference.csv; never mixed into metrics.csv.
"""
import csv
import os
import sys

import numpy as np
import anndata as ad
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FIELDS = ["combo", "method", "missing", "n_target", "pcc_pooled", "pcc_gene_mean",
          "pcc_gene_median", "pcc_spot_mean", "rmse", "mae"]


def main():
    a = ad.read_h5ad(C.OBS)
    coords = np.asarray(a.obsm["X_aligned"], dtype=np.float64)
    labels = a.obs["slice"].astype(str).to_numpy()
    X = np.asarray(a.X, dtype=np.float64)
    rows = []
    for combo, spec in C.COMBOS.items():
        vis = [C.SLICE_IDS[s] for s in spec["visible"]]
        mis = [C.SLICE_IDS[s] for s in spec["missing"]]
        tr = np.isin(labels, vis)
        ho = np.isin(labels, mis)
        tree = cKDTree(coords[tr])
        _, idx = tree.query(coords[ho], k=1)
        m = C.core_metrics(X[tr][idx], X[ho])
        m.update({"combo": combo, "method": "NN-nearest-visible",
                  "missing": "+".join(spec["missing"])})
        rows.append(m)
        print(f"[{combo}] NN pooled={m['pcc_pooled']:.4f} gene_mean={m['pcc_gene_mean']:.4f} "
              f"spot_mean={m['pcc_spot_mean']:.4f} rmse={m['rmse']:.4f}")
    os.makedirs(os.path.join(C.ROOT, "diagnostics"), exist_ok=True)
    with open(os.path.join(C.ROOT, "diagnostics", "nn_reference.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

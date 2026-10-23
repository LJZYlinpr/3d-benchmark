#!/usr/bin/env python
"""Write targets/<combo>/target.npz (missing-slice coords + GT) for every combo.

Runs in its own process so that the training scripts never have to read target
expression at all.  Only compute_metrics.py consumes these files.
"""
import os
import sys

import numpy as np
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

a = ad.read_h5ad(C.OBS)
coords = np.asarray(a.obsm["X_aligned"], dtype=np.float32)
labels = a.obs["slice"].astype(str).to_numpy()
X = a.X.toarray() if hasattr(a.X, "toarray") else (a.X.to_numpy() if hasattr(a.X, "to_numpy") else np.asarray(a.X))
names = a.obs_names.to_numpy()

for combo, spec in C.COMBOS.items():
    ho = np.isin(labels, [C.SLICE_IDS[s] for s in spec["missing"]])
    p = C.save_target(combo, coords[ho], X[ho], names[ho], labels[ho])
    print(f"{combo}: {int(ho.sum())} target bins, slices {sorted(set(labels[ho]))} -> {p}")

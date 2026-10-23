#!/usr/bin/env python
"""Independent acceptance check for Spateo_50um_uniform.h5ad (run with the benchmark env)."""
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd

ROOT = "/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/Workspace/merfish_test"
path = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/Spateo/Spateo_50um_uniform.h5ad"

g = np.load(f"{ROOT}/grid_50um.npz")
obs = ad.read_h5ad(f"{ROOT}/observed_50um.h5ad")
a = ad.read_h5ad(path)
X = np.asarray(a.X)
res = {}
res["file"] = path
res["shape"] = list(a.shape)
res["genes_identical"] = bool(list(a.var_names) == list(obs.var_names))
res["obs_cols"] = [c for c in ["x_mm", "y_mm", "z_mm", "is_observed", "slice"] if c in a.obs]
res["n_observed"] = int(a.obs["is_observed"].sum())
res["n_query"] = int((~a.obs["is_observed"]).sum())
res["dtype"] = str(X.dtype)
res["all_finite"] = bool(np.isfinite(X).all())
res["nan_count"] = int((~np.isfinite(X)).sum())
res["min"] = float(X.min())
res["max"] = float(X.max())

q = X[~a.obs["is_observed"].values]
res["query_row_std_mean"] = float(q.std(1).mean())
res["query_constant_rows"] = int((q.std(1) == 0).sum())
res["query_gene_std_mean"] = float(q.std(0).mean())

# observed rows must equal the input
obs_rows = np.nonzero(a.obs["is_observed"].values)[0]
vox = (a.obs["iz"].values * 307 * 129 + a.obs["iy"].values * 307 + a.obs["ix"].values)
ovox = (obs.obs["iz"].values * 307 * 129 + obs.obs["iy"].values * 307 + obs.obs["ix"].values)
pos = {int(v): i for i, v in enumerate(ovox)}
src = np.array([pos[int(v)] for v in vox[obs_rows]])
Xin = obs.X[src]
Xin = Xin.toarray() if hasattr(Xin, "toarray") else np.asarray(Xin)
res["obs_match_input"] = bool(np.allclose(X[obs_rows], Xin, atol=0, rtol=0))
res["obs_max_abs_diff"] = float(np.abs(X[obs_rows] - Xin).max())

# coordinates consistent with grid
xi = g["x_centers"][a.obs["ix"].values]
yi = g["y_centers"][a.obs["iy"].values]
zi = g["z_centers"][a.obs["iz"].values]
res["coords_match_grid"] = bool(
    np.allclose(a.obs["x_mm"].values, xi, atol=1e-5)
    and np.allclose(a.obs["y_mm"].values, yi, atol=1e-5)
    and np.allclose(a.obs["z_mm"].values, zi, atol=1e-5)
)
res["slice_values"] = sorted(map(str, pd.unique(a.obs["slice"].astype(str))))

# per-z-layer query prediction summary (smoothness across z)
lay = a.obs["iz"].values
per_z = {}
for iz in range(18):
    m = (lay == iz) & (~a.obs["is_observed"].values)
    if m.sum():
        per_z[int(iz)] = {
            "n_query": int(m.sum()),
            "pred_mean": round(float(X[m].mean()), 5),
            "frac_nonzero": round(float((X[m] > 0).mean()), 4),
            "row_std_mean": round(float(X[m].std(1).mean()), 4),
        }
res["per_z_query"] = per_z

# observed layers: model prediction vs observed in the same layer (in-plane recall)
res["observed_layers"] = sorted({int(v) for v in lay[a.obs["is_observed"].values]})
print(json.dumps(res, indent=2))

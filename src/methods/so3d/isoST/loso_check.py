#!/usr/bin/env python3
"""Leave-one-visible-slice-out check for isoST epoch selection (visible data only)."""
import json
import numpy as np
from types import SimpleNamespace

import run_so3d as R
import _so3d_common as C

DATASETS = ["BRCA__Human__3Ddataset_4", "Hippocampus__Mouse__3Ddataset_1"]
EPOCHS = [5, 10, 20]
OUT = "/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/Workspace/so3d_bench/methods/isoST/loso_report.json"


def score(pred, truth):
    p = pred.ravel().astype(np.float64)
    t = truth.ravel().astype(np.float64)
    pool = float(np.corrcoef(p, t)[0, 1])
    pg = []
    for g in range(pred.shape[1]):
        if truth[:, g].std() > 1e-9 and pred[:, g].std() > 1e-9:
            pg.append(np.corrcoef(pred[:, g], truth[:, g])[0, 1])
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    cos = float(np.mean(np.sum(pred * truth, 1) /
                        (np.linalg.norm(pred, axis=1) * np.linalg.norm(truth, axis=1) + 1e-12)))
    return dict(pool_pcc=pool, gene_pcc_median=float(np.median(pg)) if pg else None,
                gene_pcc_mean=float(np.mean(pg)) if pg else None,
                rmse=rmse, cosine=cos)


def main():
    args = SimpleNamespace(seed=0)
    rows = []
    for ds in DATASETS:
        vis = C.load_visible(ds)
        order, zs = C.slice_order(vis)
        # hold out the visible slice nearest the true target depth but not the
        # extreme bracketing one, so both neighbours remain visible.
        meta = C.load_meta(ds)
        zt = float(meta["z_range"]["target_z"])
        cand = [s for s in order[1:-1]]
        ho = min(cand, key=lambda s: abs(zs[s] - zt))
        rows_ho = np.flatnonzero(vis["slice"] == ho)
        keep = vis["slice"] != ho
        vtr = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[0] == vis["n"] else v)
               for k, v in vis.items()}
        vtr["n"] = int(keep.sum())
        thold = dict(coords=vis["coords"][rows_ho].copy(),
                     z=vis["z"][rows_ho].copy(), n=int(rows_ho.size),
                     obs_names=vis["obs_names"][rows_ho], slice=vis["slice"][rows_ho])
        truth = vis["X"][rows_ho]
        for ep in EPOCHS:
            pred, genes, tgt, st = R.run_dataset(ds, args, [ep, ep, ep], vis=vtr, tgt=thold)
            rec = dict(dataset=ds, holdout_slice=int(ho), epochs=ep,
                       n_holdout=int(rows_ho.size), n_genes=vis["n_genes"],
                       ratio=float(pred.mean() / max(float(truth.mean()), 1e-9)),
                       **score(pred, truth))
            print(json.dumps(rec), flush=True)
            rows.append(rec)
            C.write_json(OUT, {"rows": rows})
            import torch; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

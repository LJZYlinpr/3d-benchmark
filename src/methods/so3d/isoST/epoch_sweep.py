#!/usr/bin/env python3
"""Epoch-budget sweep for isoST on So3D (visible-only diagnostics, no target GT).

For each (dataset, epochs) it fits on the visible slices exactly like the real
runner and reports prediction scale vs visible expression plus timing.  Nothing
is written to predictions/.
"""
import json
import sys
from types import SimpleNamespace

import numpy as np
import anndata as ad

import run_so3d as R
import _so3d_common as C

DATASETS = ["BRCA__Human__3Ddataset_4", "Hippocampus__Mouse__3Ddataset_1"]
EPOCHS = [5, 10, 20, 40, 60]
OUT = "/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/Workspace/so3d_bench/methods/isoST/epoch_sweep.json"


def main():
    args = SimpleNamespace(seed=0)
    rows = []
    for ds in DATASETS:
        vis = C.load_visible(ds)
        X = vis["X"]
        vmean = float(X.mean())
        vmax = X.max(axis=0)
        for ep in EPOCHS:
            try:
                pred, genes, tgt, st = R.run_dataset(ds, args, [ep, ep, ep])
                sat = float((pred >= 0.999 * vmax[None, :]).mean())
                rec = dict(dataset=ds, epochs=ep, n_visible=vis["n"], n_genes=vis["n_genes"],
                           train_sec=st["train_sec"], pred_mean=float(pred.mean()),
                           vis_mean=vmean, ratio=float(pred.mean() / max(vmean, 1e-9)),
                           pred_zero=float((pred == 0).mean()),
                           vis_zero=float((X == 0).mean()),
                           sat_frac=sat, pred_max=float(pred.max()),
                           pgstd=float(pred.std(axis=0).mean()))
                print(json.dumps(rec), flush=True)
            except Exception as e:
                rec = dict(dataset=ds, epochs=ep, error=repr(e))
                print(json.dumps(rec), flush=True)
            rows.append(rec)
            C.write_json(OUT, {"rows": rows})
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    main()

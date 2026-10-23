#!/usr/bin/env python3
"""SpatialZ on the heart 10-slice (21-30) target-only benchmark.

Per hidden block: take the bracketing visible sections and call the official
SpatialZ.Generate_spatialz for every hidden target section depth (alpha from z),
then map the virtual point cloud to that section's target bins (2D NN + bin
mean; empty bins get the nearest virtual point). Only target bins are written.

Usage (inside the granted srun step):
  python run_heart.py --pilot [--scenario <name>]
  python run_heart.py
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import anndata as ad
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

SPATIALZ_DIR = Path("/storage/yuhongtaoLab/linpeiran/xpy/lab/baseline/methods/SpatialZ")
sys.path.insert(0, str(SPATIALZ_DIR))
from SpatialZ import Generate_spatialz  # noqa: E402

METHOD = "SpatialZ"
OUT = HERE
LOG = OUT / "run.log"


def log(msg):
    line = f"[{C.now()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


class Frames:
    """Per-section AnnData templates (xy in mm, one cell_class)."""

    def __init__(self, bins):
        self.bins = bins
        self.xmin, self.ymin = float(bins["x"].min()), float(bins["y"].min())
        self.xmax, self.ymax = float(bins["x"].max()), float(bins["y"].max())
        self._cache = {}

    def section_frame(self, s):
        if s in self._cache:
            return self._cache[s]
        rows = np.flatnonzero(self.bins["section"] == s)
        a = ad.AnnData(X=self.bins["X"][rows].copy(),
                       obs=pd.DataFrame({"cell_class": ["all"] * rows.size},
                                        index=[f"b{r}" for r in rows]),
                       var=pd.DataFrame(index=self.bins["genes"]))
        a.obsm["spatial"] = np.stack([self.bins["x"][rows] / 1000.0,
                                      self.bins["y"][rows] / 1000.0], axis=1)
        self._cache[s] = (a, rows)
        return a, rows


def map_to_bins(xy_gen, X_gen, bin_xy):
    """Assign virtual points to bins by 2D NN, average; empty bins -> NN point."""
    tree = cKDTree(bin_xy)
    dist, j = tree.query(xy_gen)
    n_bins = bin_xy.shape[0]
    W = csr_matrix((np.ones(xy_gen.shape[0], dtype=np.float32),
                    (j, np.arange(xy_gen.shape[0]))), shape=(n_bins, xy_gen.shape[0]))
    acc = np.asarray(W @ X_gen, dtype=np.float32)
    cnt = np.bincount(j, minlength=n_bins)
    filled = cnt > 0
    P = np.zeros((n_bins, X_gen.shape[1]), dtype=np.float32)
    P[filled] = acc[filled] / cnt[filled, None].astype(np.float32)
    fill_dist = np.zeros(0)
    if (~filled).any():
        pt_tree = cKDTree(xy_gen)
        fill_dist, jj = pt_tree.query(bin_xy[~filled])
        P[~filled] = X_gen[jj]
    return P, dict(n_bins=int(n_bins), n_points=int(xy_gen.shape[0]),
                   coverage=round(float(filled.mean()), 4),
                   n_nn_filled=int((~filled).sum()),
                   point_to_bin_um=dict(median=float(np.median(dist) * 1000),
                                        p90=float(np.percentile(dist, 90) * 1000)),
                   nn_fill_um=dict(median=float(np.median(fill_dist) * 1000),
                                   max=float(fill_dist.max() * 1000)) if fill_dist.size else None)


def run_scenario(bins, sc, frames, args):
    section_z = C.section_z(bins["z"], bins["section"])
    section_pred, layer_recs = {}, []
    for block, lo, hi in C.block_bounds(sc):
        za, zb = section_z[lo], section_z[hi]
        a1, rows1 = frames.section_frame(lo)
        a2, rows2 = frames.section_frame(hi)
        for tsec in block:
            zl = section_z[tsec]
            alpha = (zb - zl) / (zb - za)
            t0 = time.time()
            gen = Generate_spatialz(
                a1.copy(), a2.copy(), adata1_id=f"s{lo}", adata2_id=f"s{hi}",
                alpha=alpha, device="cuda:0", n_cell=None, k_neighbors=1,
                n_mag=args.n_mag, lr=args.lr, nb_iter_max=args.nb_iter_max,
                seed=args.seed, num_projections=args.num_projections,
                cell_type_key="cell_class", syn_mode="fast", k_sam=args.k_sam,
                verbose=False,
            )
            torch.cuda.synchronize()
            gen_sec = time.time() - t0
            xy_gen = np.asarray(gen.obsm["spatial"], dtype=np.float64)
            X_gen = np.asarray(gen.X, dtype=np.float32)
            if not np.isfinite(X_gen).all():
                raise RuntimeError(f"section {tsec}: non-finite SpatialZ expression")
            np.maximum(X_gen, 0.0, out=X_gen)
            rows_t = np.flatnonzero(bins["section"] == tsec)
            bin_xy = np.stack([bins["x"][rows_t] / 1000.0,
                               bins["y"][rows_t] / 1000.0], axis=1)
            t1 = time.time()
            P, st = map_to_bins(xy_gen, X_gen, bin_xy)
            map_sec = time.time() - t1
            section_pred[tsec] = (rows_t, P)
            st.update(section=tsec, z_um=zl, alpha=round(float(alpha), 6),
                      bracket=[lo, hi], gen_sec=round(gen_sec, 2),
                      map_sec=round(map_sec, 2), n_gen=int(X_gen.shape[0]))
            layer_recs.append(st)
            log(f"  section {tsec} (z={zl:.0f}) bracket {lo}->{hi} alpha={alpha:.3f}: "
                f"gen={X_gen.shape[0]} {gen_sec:.1f}s cover={st['coverage']} "
                f"nn={st['point_to_bin_um']['median']:.1f}um")
            del gen, X_gen
            torch.cuda.empty_cache()
    return section_pred, layer_recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--scenarios", default=None)
    ap.add_argument("--lr", type=float, default=40.0)
    ap.add_argument("--num-projections", type=int, default=200)
    ap.add_argument("--nb-iter-max", type=int, default=1000)
    ap.add_argument("--k-sam", type=int, default=3)
    ap.add_argument("--n-mag", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    gpu_idx = str(args.gpu).split(",")[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    if args.pilot:
        args.nb_iter_max = min(args.nb_iter_max, 50)
        args.num_projections = min(args.num_projections, 50)

    status = C.Status(OUT / "status.json", METHOD,
                      dict(pilot=bool(args.pilot), config=vars(args), log=str(LOG)))
    mon = C.GpuSampler(); mon.start()
    log(f"START pilot={args.pilot} cfg={vars(args)}")

    C.wait_ready()
    bins = C.load_bins()
    scs = C.load_scenarios()
    tg = C.load_target_gt()
    log(f"loaded bins X={bins['X'].shape} scenarios={len(scs)}")
    if args.scenario:
        scs = [s for s in scs if s["name"] == args.scenario]
    if args.scenarios:
        keep = set(args.scenarios.split(","))
        scs = [s for s in scs if s["name"] in keep]
    if args.pilot:
        scs = scs[:1]
    status.update(status="RUNNING", stage="loaded", scenarios_total=len(scs))

    frames = Frames(bins)
    metrics = {"method": METHOD, "dataset": "heart_10slice_50um", "voxel_um": 50,
               "pilot": bool(args.pilot),
               "config": {k: v for k, v in vars(args).items()},
               "mapping": "virtual cloud -> target section bins (2D NN + bin mean; "
                          "empty bin -> nearest virtual point)",
               "scenarios": []}
    failures = []
    for i, sc in enumerate(scs):
        t0 = time.time()
        status.update(stage=f"generate:{sc['name']}", current=sc["name"], scenario_index=i)
        try:
            section_pred, layers = run_scenario(bins, sc, frames, args)
            rows = C.scenario_rows(tg, sc)
            pred = C.align_prediction(tg, rows, section_pred, bins)
            path = C.save_pred(METHOD, sc, tg, rows, pred, extra=dict(genes=bins["genes"]))
            rec = dict(scenario=sc["name"], k=len(sc["target"]), visible=sc["visible"],
                       target=sc["target"], n_rows=int(rows.size),
                       total_sec=round(time.time() - t0, 1),
                       pred_min=float(pred.min()), pred_max=float(pred.max()),
                       pred_mean=float(pred.mean()), pred_zero_frac=float((pred == 0).mean()),
                       layers=layers, output=str(path))
            metrics["scenarios"].append(rec)
            log(f"[{sc['name']}] DONE rows={rows.size} {rec['total_sec']}s "
                f"mean={rec['pred_mean']:.4f}")
        except Exception as e:
            tb = traceback.format_exc()
            log(f"[{sc['name']}] FAILED {e!r}\n{tb}")
            failures.append(dict(scenario=sc["name"], error=repr(e), traceback=tb[-2500:]))
            status.update(stage=f"failed:{sc['name']}", last_error=repr(e), failures=failures)
            torch.cuda.empty_cache()
        status.update(scenarios_done=i + 1, gpu_peak_mb=int(mon.peak), failures=len(failures))

    mon.stop.set(); mon.join(timeout=5)
    metrics.update(gpu_peak_mb=int(mon.peak), n_done=len(metrics["scenarios"]),
                   n_failed=len(failures), failures=failures)
    C.write_json(OUT / "compute_metrics.json", metrics)
    ok = not failures
    status.update(status="DONE" if ok else "PARTIAL", stage="finished",
                  gpu_peak_mb=int(mon.peak), n_done=len(metrics["scenarios"]),
                  n_failed=len(failures), metrics=str(OUT / "compute_metrics.json"))
    log(f"FINISHED done={metrics['n_done']} failed={len(failures)} gpu_peak={mon.peak}MiB")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

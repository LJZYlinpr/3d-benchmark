#!/usr/bin/env python
"""SpatialZ on the So3D 46-dataset held-out-middle-slice benchmark (runner B).

Adapted from Workspace/merfish_test/SpatialZ/run_spatialz_50um.py and
Workspace/merfish_test/heart_10slice/methods/SpatialZ/run_heart.py.  Uses the
official `SpatialZ.Generate_spatialz` unmodified.

For the single held-out middle slice of each dataset:
  1. bracket = nearest visible slice below and above the target z (native z,
     ratio only); alpha = (z_hi - z_t) / (z_hi - z_lo).
  2. Generate_spatialz(visible_lo, visible_hi, alpha) synthesises a virtual
     point cloud; xy is fed in a common pseudo-mm frame (median within-slice
     visible NN -> 0.05 mm, i.e. a 50 um spot spacing, matching the validated
     MERFISH/heart config where lr=40 mm, nb_iter_max=1000, k_sam=3, syn_mode
     'fast' were tuned).  z is only used to compute alpha.
  3. The virtual cloud is mapped to the target slice's own coordinates
     (2D NN + mean; target points with no virtual point get the nearest virtual
     point).  Only the target rows are written.

Canonical spots carry no cell-type labels usable as a per-spot class for the
virtual slice, so every visible slice is one class ("all"), as in the heart
adapter: cell-type transfer is then a no-op and expression synthesis samples
same-class nearest neighbours.

Canonical X is used directly (already normalised by the producer, never
renormalised, never rescaled).  Target expression is never read.

Outputs: predictions/<dataset>/SpatialZ.npz, methods/SpatialZ/*.
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import anndata as ad
import torch
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS_DIR = os.path.dirname(HERE)
if METHODS_DIR not in sys.path:
    sys.path.insert(0, METHODS_DIR)
import _so3d_common as C  # noqa: E402

SPATIALZ_DIR = "/storage/yuhongtaoLab/linpeiran/xpy/lab/baseline/methods/SpatialZ"
if SPATIALZ_DIR not in sys.path:
    sys.path.insert(0, SPATIALZ_DIR)
from SpatialZ import Generate_spatialz  # noqa: E402

METHOD = "SpatialZ"
OUTDIR = os.path.join(C.METHODS_ROOT, METHOD)
SPOT_MM = 0.05          # pseudo spot spacing fed to SpatialZ (50 um in mm)
MIN_VISIBLE = 100
MIN_GENES = 10
MIN_SLICE_CELLS = 3


def dist_stats(d):
    d = np.asarray(d, dtype=np.float64)
    if d.size == 0:
        return None
    return {"median": float(np.median(d)), "mean": float(np.mean(d)),
            "p90": float(np.percentile(d, 90)), "max": float(np.max(d))}


def slice_frame(vis, s, xy_mm_all):
    rows = np.flatnonzero(vis["slice"] == s)
    a = ad.AnnData(
        X=np.ascontiguousarray(vis["X"][rows], dtype=np.float32),
        obs=pd.DataFrame({"cell_class": ["all"] * rows.size},
                         index=[f"b{r}" for r in rows]),
        var=pd.DataFrame(index=vis["genes"]))
    a.obsm["spatial"] = xy_mm_all[rows].astype(np.float64)
    return a


def map_to_target(xy_gen, X_gen, xy_t):
    tree = cKDTree(xy_t)
    dist, j = tree.query(xy_gen)
    n = xy_t.shape[0]
    W = csr_matrix((np.ones(xy_gen.shape[0], dtype=np.float32),
                    (j, np.arange(xy_gen.shape[0]))), shape=(n, xy_gen.shape[0]))
    acc = np.asarray(W @ X_gen, dtype=np.float32)
    cnt = np.bincount(j, minlength=n)
    filled = cnt > 0
    P = np.zeros((n, X_gen.shape[1]), dtype=np.float32)
    P[filled] = acc[filled] / cnt[filled, None].astype(np.float32)
    fill_dist = np.zeros(0)
    if (~filled).any():
        pt_tree = cKDTree(xy_gen)
        fill_dist, jj = pt_tree.query(xy_t[~filled])
        P[~filled] = X_gen[jj]
    stats = {
        "n_target": int(n),
        "n_points": int(xy_gen.shape[0]),
        "target_coverage": round(float(filled.mean()), 4),
        "n_target_nn_filled": int((~filled).sum()),
        "point_to_target_mm": dist_stats(dist),
        "nn_fill_mm": dist_stats(fill_dist),
    }
    return P, stats


def run_dataset(ds, args, status, sampler):
    import torch

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    t0 = time.time()
    status.log(f"[{ds}] load canonical")
    vis = C.load_visible(ds)
    tgt = C.load_target(ds)
    meta = C.load_meta(ds)
    assert np.array_equal(np.asarray(meta["genes"]).astype(str), vis["genes"]), \
        f"{ds}: visible var_names != meta['genes']"
    vis_slices = set(int(s) for s in np.unique(vis["slice"]))
    tgt_slices = set(int(s) for s in np.unique(tgt["slice"]))
    if vis_slices & tgt_slices:
        raise RuntimeError(f"target slice {sorted(tgt_slices)} overlaps visible {sorted(vis_slices)}")
    if vis["n"] < MIN_VISIBLE or vis["n_genes"] < MIN_GENES:
        return dict(status="UNSUPPORTED",
                    reason=f"too small: n_visible={vis['n']}, n_genes={vis['n_genes']}",
                    n_visible=int(vis["n"]), n_target=int(tgt["n"]),
                    n_genes=int(vis["n_genes"]), sec=round(time.time() - t0, 1))

    z_t = float(np.median(tgt["z"]))
    lo, hi, zs, order = C.bracket_slices(vis, z_t)
    if lo is None or hi is None:
        return dict(status="UNSUPPORTED",
                    reason=f"target z={z_t} not bracketed by visible slices "
                           f"(lo={lo}, hi={hi}); SpatialZ needs both sides",
                    n_target=int(tgt["n"]), sec=round(time.time() - t0, 1))

    # visible-only geometry scale; target xy uses the same transform
    xy = vis["coords"][:, :2]
    nn_list = []
    for s in order:
        m = vis["slice"] == s
        if m.sum() >= 3:
            d, _ = cKDTree(xy[m]).query(xy[m], k=2)
            nn_list.append(float(np.median(d[:, 1])))
    nn_xy = float(np.median(nn_list))
    if not np.isfinite(nn_xy) or nn_xy <= 0:
        raise ValueError("cannot estimate within-slice xy spacing")
    s_xy = SPOT_MM / nn_xy
    xy_mm = vis["coords"][:, :2] * s_xy
    xy_t = tgt["coords"][:, :2] * s_xy
    za, zb, zl = zs[lo], zs[hi], z_t
    alpha = (zb - zl) / (zb - za)
    status.log(f"[{ds}] visible={vis['n']} ({len(order)} slices) target={tgt['n']} "
               f"genes={vis['n_genes']} bracket=({lo},{hi}) alpha={alpha:.4f} "
               f"nn_xy={nn_xy:.4g} z_t={zl:.5g} target_slice={sorted(tgt_slices)}")
    if not (0.0 <= alpha <= 1.0):
        return dict(status="UNSUPPORTED",
                    reason=f"alpha={alpha:.4f} outside [0,1] (z_t={zl} not between "
                           f"visible z {za},{zb})",
                    n_target=int(tgt["n"]), sec=round(time.time() - t0, 1))
    n_lo = int((vis["slice"] == lo).sum())
    n_hi = int((vis["slice"] == hi).sum())
    if n_lo < MIN_SLICE_CELLS or n_hi < MIN_SLICE_CELLS:
        return dict(status="UNSUPPORTED",
                    reason=f"bracket slice too small: lo={lo}:{n_lo}, hi={hi}:{n_hi}",
                    n_target=int(tgt["n"]), sec=round(time.time() - t0, 1))

    a1 = slice_frame(vis, lo, xy_mm)
    a2 = slice_frame(vis, hi, xy_mm)
    status.write(stage="generate", current_dataset=ds)

    t1 = time.time()
    gen = Generate_spatialz(
        a1, a2, adata1_id=f"s{lo}", adata2_id=f"s{hi}",
        alpha=float(alpha), device="cuda:0" if torch.cuda.is_available() else "cpu",
        n_cell=None, k_neighbors=1, n_mag=args.n_mag, lr=args.lr,
        nb_iter_max=args.nb_iter_max, seed=args.seed,
        num_projections=args.num_projections, cell_type_key="cell_class",
        syn_mode="fast", k_sam=args.k_sam, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_sec = time.time() - t1
    xy_gen = np.asarray(gen.obsm["spatial"], dtype=np.float64)
    X_gen = np.asarray(gen.X, dtype=np.float32)
    if not np.isfinite(X_gen).all():
        raise RuntimeError(f"{ds}: non-finite SpatialZ expression")
    np.maximum(X_gen, 0.0, out=X_gen)
    init = np.asarray(gen.obsm["spatial"], dtype=np.float64)
    del gen

    t2 = time.time()
    P, mstats = map_to_target(xy_gen, X_gen, xy_t)
    map_sec = time.time() - t2
    if not np.isfinite(P).all():
        raise RuntimeError(f"{ds}: non-finite mapped prediction")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    C.write_prediction(METHOD, ds, P, tgt, vis["genes"])
    ver = C.verify_prediction(METHOD, ds, vis["genes"], tgt)
    if not ver["pass"]:
        raise RuntimeError(f"{ds}: prediction verification failed: {ver}")
    rec = dict(status="DONE",
               n_visible=int(vis["n"]), n_target=int(tgt["n"]),
               n_genes=int(vis["n_genes"]), n_visible_slices=len(order),
               target_slice=sorted(tgt_slices), visible_slices=sorted(vis_slices),
               bracket=[int(lo), int(hi)], alpha=round(float(alpha), 6),
               z_native={"lo": float(za), "hi": float(zb), "target": float(zl)},
               n_generated=int(X_gen.shape[0]),
               gen_sec=round(gen_sec, 1), map_sec=round(map_sec, 1),
               total_sec=round(time.time() - t0, 1),
               torch_alloc_peak_mb=(round(torch.cuda.max_memory_allocated() / 2**20, 1)
                                    if torch.cuda.is_available() else None),
               xy_frame={"spot_mm": SPOT_MM, "nn_xy_native": nn_xy, "scale": s_xy},
               mapping=mstats, verify=ver, output=C.pred_path(METHOD, ds))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--datasets", default=None, help="comma separated list")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--lr", type=float, default=40.0)
    ap.add_argument("--num-projections", type=int, default=200)
    ap.add_argument("--nb-iter-max", type=int, default=1000)
    ap.add_argument("--k-sam", type=int, default=3)
    ap.add_argument("--n-mag", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    status = C.Status(METHOD, OUTDIR, dict(command=" ".join([sys.executable] + sys.argv),
                                           config=vars(args)))
    sampler = C.GpuSampler()
    sampler.start()

    try:
        status.log(f"waiting for PREP_READY {C.READY_CANDIDATES}")
        C.wait_ready(log=status.log)
        all_ds = C.dataset_dirs()
        if args.dataset:
            sel = [args.dataset]
        elif args.datasets:
            sel = [d.strip() for d in args.datasets.split(",") if d.strip()]
        elif args.all:
            sel = all_ds
        else:
            raise SystemExit("pass --all / --dataset / --datasets")
        missing = [d for d in sel if d not in all_ds]
        if missing:
            raise SystemExit(f"canonical incomplete for: {missing}")
        if args.limit:
            sel = sel[: args.limit]
        status.log(f"selected {len(sel)}/{len(all_ds)} canonical datasets; "
                   f"lr={args.lr} iters={args.nb_iter_max} proj={args.num_projections}")
        status.write(stage="ready", datasets_selected=len(sel),
                     datasets_available=len(all_ds))

        t_all = time.time()
        n_skip = 0
        for i, ds in enumerate(sel, 1):
            status.write(stage="dataset", current_dataset=ds, dataset_index=i,
                         gpu_peak_mb=int(sampler.peak))
            try:
                if args.skip_existing:
                    vis = C.load_visible(ds)
                    tgt = C.load_target(ds)
                    if C.existing_ok(METHOD, ds, vis["genes"], tgt):
                        status.log(f"[{ds}] skip (existing valid prediction)")
                        n_skip += 1
                        continue
                    del vis, tgt
                rec = run_dataset(ds, args, status, sampler)
            except Exception as e:
                tb = traceback.format_exc()
                status.log(f"[{ds}] FAILED {e!r}\n{tb}")
                rec = dict(status="FAILED", error=repr(e), traceback=tb[-2500:])
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            rec["finished_at"] = C.now()
            status.record(ds, rec)
            C.write_dataset_status(METHOD, ds, rec)
            status.write(stage="dataset_done", current_dataset=ds,
                         datasets_done=status.d["datasets_done"],
                         datasets_failed=status.d["datasets_failed"],
                         datasets_unsupported=status.d["datasets_unsupported"],
                         gpu_peak_mb=int(sampler.peak))
            status.log(f"[{ds}] {rec['status']} "
                       f"({rec.get('total_sec', 0)}s, {rec.get('output', rec.get('reason', ''))})")

        sampler.stop_evt.set()
        total = time.time() - t_all
        recs = status.d["records"]
        done = {k: v for k, v in recs.items() if v.get("status") == "DONE"}
        failed = {k: v for k, v in recs.items() if v.get("status") == "FAILED"}
        unsup = {k: v for k, v in recs.items() if v.get("status") == "UNSUPPORTED"}
        full_scope = (len(sel) == C.N_DATASETS)
        metrics = {
            "method": METHOD,
            "benchmark": "so3d 46 datasets, one held-out middle slice, target-only",
            "lane": C.LANE,
            "command": " ".join([sys.executable] + sys.argv),
            "model": "official SpatialZ.Generate_spatialz (unmodified), syn_mode=fast",
            "input_mode": "canonical visible.h5ad (all visible slices), canonical X used "
                          "as-is; target.npz X never read",
            "output": "predictions/<dataset>/SpatialZ.npz (target rows only, no volume)",
            "config": dict(lr_mm=args.lr, nb_iter_max=args.nb_iter_max,
                           num_projections=args.num_projections, k_sam=args.k_sam,
                           n_mag=args.n_mag, seed=args.seed, syn_mode="fast",
                           xy_frame_mm=SPOT_MM,
                           mapping="virtual cloud -> target coords (2D NN + mean; "
                                   "unfilled target points -> nearest virtual point)"),
            "datasets_total": C.N_DATASETS,
            "datasets_selected": len(sel),
            "full_scope": bool(full_scope),
            "n_done": len(done), "n_failed": len(failed), "n_unsupported": len(unsup),
            "n_skipped": n_skip,
            "coverage": f"{len(done)}/{len(sel)}",
            "missing": sorted(set(sel) - set(done)),
            "failures": {k: (v.get("error") or v.get("reason")) for k, v in recs.items()
                         if v.get("status") != "DONE"},
            "total_sec": round(total, 1),
            "sum_dataset_sec": round(sum(v.get("total_sec", 0) for v in recs.values()), 1),
            "mean_dataset_sec": round(float(np.mean([v["total_sec"] for v in done.values()])), 1)
            if done else None,
            "output_bytes_total": int(sum(v.get("verify", {}).get("bytes", 0)
                                          for v in done.values())),
            "gpu_peak_mb_device": int(sampler.peak),
            "gpu_peak_util_pct": int(sampler.peak_util),
            "records": recs,
            "finished_at": C.now(),
        }
        C.write_json(os.path.join(OUTDIR, "compute_metrics.json"), metrics)
        st = "DONE" if (full_scope and len(done) == C.N_DATASETS
                        and not failed and not unsup) else "PARTIAL"
        if not done:
            st = "FAILED"
        status.finish(st, n_done=len(done), n_failed=len(failed),
                      n_unsupported=len(unsup), total_sec=round(total, 1),
                      gpu_peak_mb=int(sampler.peak),
                      metrics="methods/SpatialZ/compute_metrics.json")
        C.update_runner_status(METHOD, {
            "status": st, "final": bool(full_scope and len(done) == C.N_DATASETS),
            "n_datasets": C.N_DATASETS, "n_done": len(done),
            "n_failed": len(failed), "n_unsupported": len(unsup),
            "coverage": f"{len(done)}/{C.N_DATASETS}",
            "missing": sorted(set(all_ds) - set(done)),
            "failures": {k: (v.get("error") or v.get("reason")) for k, v in recs.items()
                         if v.get("status") != "DONE"},
            "total_sec": round(total, 1),
            "mean_dataset_sec": metrics["mean_dataset_sec"],
            "gpu_peak_mb": int(sampler.peak),
            "config": metrics["config"],
            "metrics": "methods/SpatialZ/compute_metrics.json",
            "status_file": "methods/SpatialZ/status.json",
            "log": "methods/SpatialZ/run.log",
            "updated_at": C.now(),
        })
        status.log(f"{st}: done={len(done)} failed={len(failed)} unsup={len(unsup)} "
                   f"skipped={n_skip} in {total / 60:.1f} min")
        return 0 if st == "DONE" else 1
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        with open(status.logpath, "a") as f:
            f.write(tb + "\n")
        status.finish("FAILED", error=tb[-2000:])
        C.update_runner_status(METHOD, {"status": "FAILED", "error": tb[-1000:],
                                        "updated_at": C.now()})
        raise


if __name__ == "__main__":
    sys.exit(main())

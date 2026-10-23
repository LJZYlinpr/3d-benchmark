#!/usr/bin/env python3
"""DeepSpatial on the MERFISH 5-slice (.006-.010) target-only benchmark.

Per scenario: fit the official DeepSpatial UOT + flow-matching model on the
VISIBLE sections only, then for every hidden block run the official
reconstruct_between_slices between the bracketing visible sections and map the
generated points to that block's target bins (3D NN). The segment is kept in
memory only; the output file holds target bins exclusively.

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
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

METHOD_DIR = Path(os.environ["MERFISH5_UPSTREAM"]) / "DeepSpatial"
sys.path.insert(0, str(METHOD_DIR))
import deepspatial as ds  # noqa: E402

METHOD = "DeepSpatial"
OUT = C.METHODS_ROOT / METHOD
LOG = OUT / "run.log"


def log(msg):
    line = f"[{C.now()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def gpu_mb():
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2**20, 1)


def make_frames(bins, visible):
    frames = []
    for s in visible:
        rows = np.flatnonzero(bins["section"] == s)
        a = ad.AnnData(X=bins["X"][rows].copy(),
                       obs=pd.DataFrame({"cell_class": ["all"] * rows.size,
                                         "z_coord": np.full(rows.size, float(bins["z"][rows].mean())),
                                         "section": [s] * rows.size},
                                        index=[f"{s}_{i}" for i in range(rows.size)]),
                       var=pd.DataFrame(index=bins["genes"]))
        a.obsm["spatial_xy"] = np.stack([bins["x"][rows], bins["y"][rows]], axis=1).astype(np.float64)
        frames.append((a, rows))
    frames.sort(key=lambda f: float(f[0].obs["z_coord"].iloc[0]))
    return frames


def map_block(seg, bins, tsecs, block_note):
    z = np.asarray(seg.obs["z_coord"], dtype=np.float64)
    xy = np.asarray(seg.obsm["spatial_xy"], dtype=np.float64)[:, :2]
    X = seg.X.toarray() if hasattr(seg.X, "toarray") else np.asarray(seg.X)
    X = np.asarray(X, dtype=np.float32)
    n_neg = int((X < 0).sum())
    np.maximum(X, 0.0, out=X)
    if not np.isfinite(X).all():
        raise RuntimeError(f"non-finite DeepSpatial segment values ({block_note})")
    pts = np.column_stack([xy, z])
    tree = cKDTree(pts)
    out, stats = {}, []
    for tsec in tsecs:
        rows = np.flatnonzero(bins["section"] == tsec)
        q = np.column_stack([bins["x"][rows], bins["y"][rows],
                             np.full(rows.size, float(np.mean(bins["z"][rows])))])
        dist, j = tree.query(q, k=1)
        out[tsec] = (rows, X[j])
        stats.append(dict(section=tsec, n_bins=int(rows.size), n_gen=int(X.shape[0]),
                          nn_um_median=float(np.median(dist)),
                          nn_um_p90=float(np.percentile(dist, 90)),
                          nn_um_max=float(dist.max())))
        log(f"  section {tsec}: mapped {rows.size} bins from {X.shape[0]} generated pts, "
            f"nn med={np.median(dist):.1f}um p90={np.percentile(dist,90):.1f}um")
    return out, stats, n_neg


def run_scenario(bins, sc, args, exp_dir):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    frames = make_frames(bins, sc["visible"])
    model = ds.DeepSpatial()

    t0 = time.time()
    model.setup_data(adata_list=[f[0] for f in frames], spatial_key="spatial_xy",
                     z_key="z_coord", label_key="cell_class",
                     batch_size=args.batch_size, num_workers=0,
                     n_samples_base=args.n_samples)
    setup_sec = time.time() - t0
    model.build_model(patch_size=args.patch_size, hidden_size=args.hidden_size,
                      depth=args.depth, num_heads=args.num_heads, lr=2e-4,
                      sampling_method="dopri5", atol=1e-5, rtol=1e-5)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    model.fit(max_epochs=args.epochs, save_dir=str(exp_dir), accelerator="gpu",
              devices=[0], save_ckpt=False)
    torch.cuda.synchronize()
    train_sec = time.time() - t0
    peak_train = gpu_mb()
    log(f"  trained {args.epochs} epochs in {train_sec:.1f}s "
        f"(setup {setup_sec:.1f}s) peak={peak_train}MB")

    section_pred, block_stats = {}, []
    infer_t0 = time.time()
    for block, lo, hi in C.block_bounds(sc):
        fmap = {str(a.obs["section"].iloc[0]): a for a, _ in frames}
        if lo not in fmap or hi not in fmap:
            raise RuntimeError(f"bracket {lo}/{hi} missing from visible frames")
        ad0, ad1 = fmap[lo], fmap[hi]
        t1 = time.time()
        seg = model.reconstruct_between_slices(adata0=ad0, adata1=ad1,
                                               thickness=args.thickness,
                                               steps=args.steps,
                                               chunk_size=args.chunk_size,
                                               device="cuda")
        torch.cuda.synchronize()
        gen_sec = time.time() - t1
        out, stats, n_neg = map_block(seg, bins, block, f"{lo}->{hi}")
        section_pred.update(out)
        block_stats.append(dict(block=block, bracket=[lo, hi], gen_sec=round(gen_sec, 2),
                                n_negative_clipped=n_neg, sections=stats))
        del seg
        torch.cuda.empty_cache()
    infer_sec = time.time() - infer_t0
    return dict(section_pred=section_pred, setup_sec=round(setup_sec, 2),
                train_sec=round(train_sec, 2), infer_sec=round(infer_sec, 2),
                peak_train=peak_train, peak_mb=gpu_mb(), blocks=block_stats,
                n_visible_sections=len(frames))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--scenarios", default=None)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-samples", type=int, default=30000)
    ap.add_argument("--patch-size", type=int, default=8)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--chunk-size", type=int, default=2048)
    ap.add_argument("--thickness", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    gpu_idx = str(args.gpu).split(",")[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    if args.pilot:
        args.epochs = min(args.epochs, 1)
        args.n_samples = min(args.n_samples, 2000)
        args.batch_size = min(args.batch_size, 256)
        args.steps = min(args.steps, 8)

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

    metrics = {"method": METHOD, "dataset": "merfish_5slice_50um", "voxel_um": 50,
               "pilot": bool(args.pilot), "config": vars(args),
               "mapping": "reconstruct_between_slices per hidden block; target bins -> "
                          "3D NN over generated points (in memory only)",
               "scenarios": []}
    failures = []
    for i, sc in enumerate(scs):
        t0 = time.time()
        status.update(stage=f"fit:{sc['name']}", current=sc["name"], scenario_index=i)
        try:
            exp_dir = OUT / "checkpoints" / sc["name"]
            exp_dir.mkdir(parents=True, exist_ok=True)
            r = run_scenario(bins, sc, args, exp_dir)
            rows = C.scenario_rows(tg, sc)
            pred = C.align_prediction(tg, rows, r["section_pred"], bins)
            path = C.save_pred(METHOD, sc, tg, rows, pred, extra=dict(genes=bins["genes"]))
            rec = dict(scenario=sc["name"], k=len(sc["target"]), visible=sc["visible"],
                       target=sc["target"], n_rows=int(rows.size),
                       setup_sec=r["setup_sec"], train_sec=r["train_sec"],
                       infer_sec=r["infer_sec"], total_sec=round(time.time() - t0, 1),
                       peak_train_mb=r["peak_train"], peak_mb=r["peak_mb"],
                       n_visible_sections=r["n_visible_sections"],
                       pred_min=float(pred.min()), pred_max=float(pred.max()),
                       pred_mean=float(pred.mean()), pred_zero_frac=float((pred == 0).mean()),
                       blocks=r["blocks"], output=str(path))
            metrics["scenarios"].append(rec)
            log(f"[{sc['name']}] DONE rows={rows.size} total={rec['total_sec']}s "
                f"mean={rec['pred_mean']:.4f}")
            del r
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

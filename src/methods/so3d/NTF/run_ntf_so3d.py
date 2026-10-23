#!/usr/bin/env python
"""NTF for the So3D 46 held-out-slice benchmark.

Adapted from Workspace/merfish_test/NTF/run_ntf_50um.py, but trains on the
canonical visible slices only and predicts the held-out target spots directly
(target-only output, no uniform volume).

Model: NTF GeneINR (tiny-cuda-nn hash grid + MLP), param-group Adam
(hash grid 1e-2, rest 1e-3), MultiStepLR, single precision, no dropout.
The target slice has no slice embedding at inference; expression is sampled
from the coordinate INR, so only visible-slice expression is fitted.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
import _so3d_common as C  # noqa: E402

METHOD = "NTF"
NTF_ROOT = "/storage/yuhongtaoLab/linpeiran/xpy/lab/baseline/methods/NTF"
sys.path.insert(0, NTF_ROOT)


def estimate_resolution(xyz, slice_labels):
    """xy spacing = median within-slice NN distance; z spacing = median slice gap."""
    from scipy.spatial import cKDTree
    xyz = np.asarray(xyz, dtype=np.float64)
    sl = np.asarray(slice_labels).astype(str)
    n = xyz.shape[0]
    if n < 2:
        return np.array([1e-3, 1e-3, 1e-3], dtype=np.float64)
    ds = []
    for s in np.unique(sl):
        p = xyz[sl == s]
        if p.shape[0] < 3:
            continue
        k = min(2000, p.shape[0])
        idx = np.random.default_rng(0).choice(p.shape[0], k, replace=False)
        d, _ = cKDTree(p[:, :2]).query(p[idx, :2], k=2)
        ds.append(float(np.median(d[:, 1])))
    d_xy = float(np.median(ds)) if ds else 1e-3
    zs = np.array([np.median(xyz[sl == s, 2]) for s in np.unique(sl)])
    zs = np.unique(np.sort(zs))
    dz = np.diff(zs)
    dz = dz[dz > 0]
    d_z = float(np.median(dz)) if dz.size else d_xy
    if not np.isfinite(d_xy) or d_xy <= 0:
        d_xy = 1e-3
    if not np.isfinite(d_z) or d_z <= 0:
        d_z = d_xy
    return np.array([d_xy, d_xy, d_z], dtype=np.float64)


def build_adata(vis):
    import anndata as ad
    sl = pd.Categorical(vis["slice"], categories=sorted(set(vis["slice"].tolist())))
    obs = pd.DataFrame({"slice": sl}, index=vis["obs_names"])
    a = ad.AnnData(X=np.ascontiguousarray(vis["X"], dtype=np.float32), obs=obs)
    a.var_names = list(vis["genes"])
    a.obsm["spatial"] = np.ascontiguousarray(vis["coords"], dtype=np.float32)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--n-iter", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--enc-lr", type=float, default=1e-2)
    ap.add_argument("--net-lr", type=float, default=1e-3)
    ap.add_argument("--milestones", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--n-levels", type=int, default=13)
    ap.add_argument("--log2-hashmap-size", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--weight-expr", type=float, default=1.0)
    ap.add_argument("--sample-chunk", type=int, default=65536)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = C.METHODS_ROOT / METHOD
    outdir.mkdir(parents=True, exist_ok=True)
    log = C.RunLog(outdir / "run.log")
    status = C.Status(outdir / "status.json", METHOD)
    status.update(status="RUNNING", stage="load", pilot=args.pilot,
                  command=" ".join([sys.executable] + sys.argv))
    mon = C.GpuSampler()
    mon.start()
    C.set_method(METHOD, "RUNNING", stage="load", datasets_done=0, coverage=0.0,
                 metrics=str(outdir / "compute_metrics.json"))

    import torch
    import anndata as ad
    import NTF.data as ndmod
    import NTF.models as ntfm
    from NTF.config import add_shared_args, process_args
    from NTF.data import SpatialOmicsDataset
    from NTF.sample import sample_points

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = add_shared_args(description="NTF so3d")
    ntf_args = parser.parse_args([
        "--input_data", "unused",
        "--output_dir", str(outdir),
        "--slice_id", "slice",
        "--log_dir", str(outdir / "runs"),
        "--device", args.device,
        "--n_iter", str(args.n_iter),
        "--batch_size", str(args.batch_size),
        "--n_levels", str(args.n_levels),
        "--log2_hashmap_size", str(args.log2_hashmap_size),
        "--n_samples", str(args.n_samples),
        "--weight_expr", str(args.weight_expr),
        "--single_precision",
        "--no_dropout",
    ])
    ntf_args = process_args(ntf_args)

    def train_one(ds, vis, tgt):
        dev = args.device if torch.cuda.is_available() else "cpu"
        ntf_args.device = dev
        adata = build_adata(vis)
        dataset = SpatialOmicsDataset(adata, "slice")
        res = estimate_resolution(vis["coords"], vis["slice"])
        res_t = torch.tensor(res, dtype=torch.float32)
        xyz_min = vis["coords"].min(0) - 0.5 * res
        xyz_max = vis["coords"].max(0) + 0.5 * res
        BB = torch.tensor(np.stack([xyz_min, xyz_max]).astype(np.float32))
        ndmod.SpatialOmicsDataset.bounding_box = property(lambda self: BB)
        ndmod.SpatialOmicsDataset._estimate_resolution = staticmethod(
            lambda xyz, n_samples=2000: res_t.clone()
        )
        dataset.resolution = res_t
        model = ntfm.NeuralTranscriptomicField(
            n_genes=dataset.n_genes, n_slices=dataset.n_slices,
            resolution=dataset.resolution, bounding_box=dataset.bounding_box,
            args=ntf_args, pos_weight=dataset.nonzero_to_zero_ratio,
        )
        enc_ids = {id(p) for p in model.inr.encoding.parameters()}
        enc_pars = [p for p in model.parameters() if id(p) in enc_ids]
        net_pars = [p for p in model.parameters() if id(p) not in enc_ids]
        optimizer = torch.optim.Adam(
            [{"params": enc_pars, "lr": args.enc_lr},
             {"params": net_pars, "lr": args.net_lr}],
            betas=(0.9, 0.99), eps=1e-15, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(m * args.n_iter) for m in args.milestones],
            gamma=args.gamma)
        model.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_train = time.time()
        avg = {}
        alpha = 0.999
        for i in range(1, args.n_iter + 1):
            batch = dataset.get_batch(args.batch_size, dev)
            losses = model(**batch)
            loss = losses[ntfm.DS_LOSS] + args.weight_expr * losses.get(ntfm.E_REG, 0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            cur = {"loss": float(loss), "mse": float(losses[ntfm.D_LOSS]),
                   "logvar": float(losses[ntfm.S_LOSS])}
            if not avg:
                avg = dict(cur)
            else:
                for k in avg:
                    avg[k] = avg[k] * alpha + cur[k] * (1 - alpha)
            if i % 1000 == 0 or i == args.n_iter:
                log(f"[{ds} iter {i}/{args.n_iter}] " + json.dumps(
                    {k: round(v, 6) for k, v in avg.items()}))
        train_sec = time.time() - t_train
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        alloc = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        resv = torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_available() else 0.0

        model.inr.eval()
        coords = np.ascontiguousarray(tgt["coords"], dtype=np.float32)
        pred = np.empty((coords.shape[0], dataset.n_genes), dtype=np.float32)
        t_inf = time.time()
        for s in range(0, len(coords), args.sample_chunk):
            xb = torch.from_numpy(coords[s:s + args.sample_chunk]).to(dev)
            with torch.no_grad():
                out = sample_points(model.inr, xb)
            pred[s:s + args.sample_chunk] = out["expression"].float().cpu().numpy()
        inf_sec = time.time() - t_inf
        # in-sample fit check on a visible subset (never target expression)
        n_chk = min(2000, vis["n_obs"])
        sel = np.linspace(0, vis["n_obs"] - 1, n_chk).astype(np.int64)
        with torch.no_grad():
            o = sample_points(model.inr, torch.from_numpy(
                np.ascontiguousarray(vis["coords"][sel], dtype=np.float32)).to(dev))
        pv = o["expression"].float().cpu().numpy()
        gv = vis["X"][sel]
        rmse = float(np.sqrt(((pv - gv) ** 2).mean()))
        fin = float(np.isfinite(pred).mean())
        del model, optimizer, dataset, adata
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return pred, dict(
            train_sec=round(train_sec, 2), infer_sec=round(inf_sec, 2),
            n_iter=args.n_iter, batch_size=args.batch_size,
            n_visible_spots=int(vis["n_obs"]), n_visible_slices=int(len(set(vis["slice"]))),
            n_target=int(tgt["n"]), n_genes=int(dataset_n_genes(vis)),
            resolution=list(map(float, res)),
            bbox=np.stack([xyz_min, xyz_max]).tolist(),
            gpu_alloc_peak_mb=round(float(alloc), 1),
            gpu_reserved_peak_mb=round(float(resv), 1),
            in_sample=dict(n=n_chk, rmse=rmse),
            final_loss={k: round(v, 6) for k, v in avg.items()},
            finite_frac=fin,
        )

    def dataset_n_genes(vis):
        return vis["n_vars"]

    def body():
        t0 = time.time()
        torch.manual_seed(1234)
        np.random.seed(1234)
        dss = C.select_datasets(args.datasets)
        if args.pilot:
            dss = dss[:1]
        log(f"[data] {len(dss)} datasets pilot={args.pilot} host={os.uname().nodename} "
            f"cuda={torch.cuda.is_available()}")
        status.update(stage="train", datasets_total=len(dss))
        per, ver, failed = {}, [], {}
        for i, ds in enumerate(dss):
            ts = time.time()
            try:
                vis = C.load_visible(ds)
                tgt = C.load_target(ds)
                C.check_pair(ds, vis, tgt)
                if args.skip_existing and C.existing_ok(ds, METHOD, tgt):
                    v = C.verify_prediction(ds, METHOD, tgt)
                    ver.append(v)
                    per[ds] = {"skipped_existing": True, "sec": 0.0,
                               "n_target": int(tgt["n"]), "n_genes": int(vis["n_vars"]),
                               "pass": v["pass"]}
                    C.write_ds_status(ds, METHOD, "DONE", "existing prediction verified",
                                      n_target=int(tgt["n"]), n_genes=int(vis["n_vars"]))
                    log(f"[skip] {ds} existing ok")
                    continue
                status.update(stage="train", current_dataset=ds)
                pred, meta = train_one(ds, vis, tgt)
                C.write_prediction(ds, METHOD, pred, tgt,
                                   extra={"genes": vis["genes"].astype(str)})
                v = C.verify_prediction(ds, METHOD, tgt)
                ver.append(v)
                per[ds] = {"sec": round(time.time() - ts, 2),
                           "n_target": int(tgt["n"]), "n_genes": int(vis["n_vars"]),
                           "n_visible_spots": int(vis["n_obs"]),
                           "target_slices": sorted(set(tgt["slice"].tolist())),
                           "visible_slices": sorted(set(vis["slice"].tolist())),
                           **meta, "verification": v}
                C.write_ds_status(ds, METHOD, "DONE", "",
                                  n_target=int(tgt["n"]), n_genes=int(vis["n_vars"]),
                                  runtime_sec=round(time.time() - ts, 2))
                log(f"[pred] {ds} n_t={tgt['n']} g={vis['n_vars']} "
                    f"train={meta['train_sec']}s infer={meta['infer_sec']}s "
                    f"rmse_vis={meta['in_sample']['rmse']:.4f} pass={v['pass']}")
            except Exception:
                tb = traceback.format_exc()
                failed[ds] = {"status": "FAILED", "error": tb[-3000:]}
                C.write_ds_status(ds, METHOD, "FAILED",
                                  tb.strip().splitlines()[-1] if tb.strip() else "error",
                                  error=tb[-3000:])
                log(f"[fail] {ds}\n{tb}")
            status.update(stage="train", datasets_done=i + 1, last_dataset=ds,
                          failures=len(failed), gpu_peak_mb=mon.peak_mb)

        mon.stop()
        total = time.time() - t0
        all_pass = all(v["pass"] for v in ver) and not failed
        metrics = dict(
            method=METHOD,
            method_full="NTF GeneINR hash-grid INR (tiny-cuda-nn), visible-only fit, "
                        "coordinate sampling at held-out spots",
            status="DONE" if all_pass else "FAILED",
            lane=C.LANE, host=os.uname().nodename,
            slurm_job=os.environ.get("SLURM_JOB_ID"), pilot=args.pilot,
            n_datasets=len(dss), datasets=dss, n_done=len(per), n_failed=len(failed),
            coverage=round(len(per) / len(dss), 4) if dss else 0.0,
            total_target_rows=int(sum(p["n_target"] for p in per.values())),
            config=dict(n_iter=args.n_iter, batch_size=args.batch_size,
                        enc_lr=args.enc_lr, net_lr=args.net_lr,
                        milestones=args.milestones, gamma=args.gamma,
                        n_levels=args.n_levels, log2_hashmap_size=args.log2_hashmap_size,
                        n_samples=args.n_samples, weight_expr=args.weight_expr,
                        single_precision=True, no_dropout=True,
                        bbox="visible-coordinate tight bbox +-0.5*median spacing",
                        psf="per-axis median nearest-neighbour spacing",
                        device=args.device),
            compute=dict(cpu_cores=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                         gpu_peak_mb=mon.peak_mb, gpu_util_peak_pct=mon.peak_util,
                         ram_peak_mb=round(C.ram_peak_mb(), 1),
                         total_sec=round(total, 2)),
            per_dataset=per, failures=failed, finished_at=C.now(),
        )
        C.write_json(outdir / ("compute_metrics_pilot.json" if args.pilot else "compute_metrics.json"),
                     metrics)
        status.update(status=metrics["status"], stage="done", n_datasets=len(dss),
                      n_done=len(per), n_failed=len(failed), coverage=metrics["coverage"],
                      total_sec=round(total, 2), gpu_peak_mb=mon.peak_mb,
                      total_target_rows=metrics["total_target_rows"])
        C.set_method(METHOD, metrics["status"], datasets_total=len(dss),
                     datasets_done=len(per), coverage=metrics["coverage"],
                     n_failed=len(failed), total_sec=round(total, 2),
                     total_target_rows=metrics["total_target_rows"],
                     gpu_peak_mb=mon.peak_mb,
                     metrics=str(outdir / "compute_metrics.json"), failures=list(failed))
        log(f"[done] {len(per)}/{len(dss)} datasets rows={metrics['total_target_rows']} "
            f"total={total:.1f}s gpu_peak={mon.peak_mb}MB all_pass={all_pass}")

    C.run_guard(METHOD, status, log, body)


if __name__ == "__main__":
    main()

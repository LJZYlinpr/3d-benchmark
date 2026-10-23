#!/usr/bin/env python
"""NTF (Neural Transcriptomic Field) for the heart 10-slice target-only benchmark.

Per scenario: train on visible-section bins only, predict the hidden target bins,
write predictions/<scenario>/NTF.npz aligned to canonical/target_gt.npz.

Adapted from Workspace/merfish_test/NTF/run_ntf_50um.py: same library model/loss,
same param-group Adam (hash grid 1e-2, rest 1e-3), MultiStepLR, tight bbox and
isotropic 50 um PSF. The GeneINR expression path is coordinate-only, so hidden
sections need no slice embedding. No target expression is ever read.
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _heart_common as hc  # noqa: E402

METHOD = "NTF"
OUTDIR = os.path.join(hc.METHODS_ROOT, "NTF")
NTF_ROOT = "/storage/yuhongtaoLab/linpeiran/xpy/lab/baseline/methods/NTF"
sys.path.insert(0, NTF_ROOT)


def run_scenario(scen, bins, tmeta, args, status, mon):
    import torch
    import anndata as ad
    import pandas as pd

    import NTF.data as ndmod
    import NTF.models as ntfm
    from NTF.config import add_shared_args, process_args
    from NTF.data import SpatialOmicsDataset
    from NTF.sample import sample_points

    prob = hc.build_problem(scen, bins, tmeta)
    assert not (set(prob["visible"]) & set(prob["target"]))
    assert not np.isin(bins["section"][prob["tr_idx"]], prob["target"]).any()

    # ---------------- training AnnData: visible bins only ----------------
    tr_idx = prob["tr_idx"]
    sec_tr = bins["section"][tr_idx]
    X = np.ascontiguousarray(prob["expr_tr"], dtype=np.float32)
    coords_mm = (prob["coords_tr"] / 1000.0).astype(np.float32)
    te_mm = (prob["coords_te"] / 1000.0).astype(np.float32)

    a = ad.AnnData(X=X)
    a.obs = pd.DataFrame(
        {"slice": pd.Categorical(sec_tr, categories=[str(s) for s in prob["visible"]])},
        index=[f"v{i}" for i in range(X.shape[0])])
    a.obsm["spatial"] = coords_mm
    a.var_names = bins["genes"]
    a.var["gene_name"] = bins["genes"]

    bs = 50.0 / 1000.0  # 50 um voxel, mm units
    allc = np.vstack([coords_mm, te_mm]).astype(np.float32)
    bb = np.stack([allc.min(0) - 0.5 * bs, allc.max(0) + 0.5 * bs]).astype(np.float32)
    BB = torch.tensor(bb, dtype=torch.float32)
    ndmod.SpatialOmicsDataset.bounding_box = property(lambda self: BB)
    ndmod.SpatialOmicsDataset._estimate_resolution = staticmethod(
        lambda xyz, n_samples=2000: torch.tensor([bs, bs, bs], dtype=torch.float32))

    parser = add_shared_args(description="NTF heart 10-slice")
    ntf_args = parser.parse_args([
        "--input_data", "unused.h5ad",
        "--output_dir", OUTDIR,
        "--slice_id", "slice",
        "--log_dir", os.path.join(OUTDIR, "runs"),
        "--device", "cuda:0",
        "--n_iter", str(args.iters),
        "--batch_size", str(args.batch_size),
        "--learning_rate", "1e-4",
        "--n_levels", str(args.n_levels),
        "--log2_hashmap_size", str(args.log2_hashmap_size),
        "--n_samples", str(args.n_samples),
        "--weight_expr", "1.0",
        "--early_stopping_check_interval", "200",
        "--single_precision",
        "--no_dropout",
    ])
    ntf_args = process_args(ntf_args)
    ntf_args.early_stopping_check_interval = 200
    ntf_args.early_stopping_patience = 10 ** 9

    dataset = SpatialOmicsDataset(a, "slice")
    model = ntfm.NeuralTranscriptomicField(
        n_genes=dataset.n_genes, n_slices=dataset.n_slices,
        resolution=dataset.resolution, bounding_box=dataset.bounding_box,
        args=ntf_args, pos_weight=dataset.nonzero_to_zero_ratio)

    enc_ids = {id(p) for p in model.inr.encoding.parameters()}
    enc_pars = [p for p in model.parameters() if id(p) in enc_ids]
    net_pars = [p for p in model.parameters() if id(p) not in enc_ids]
    optimizer = torch.optim.Adam(
        [{"params": enc_pars, "lr": args.enc_lr}, {"params": net_pars, "lr": args.net_lr}],
        betas=(0.9, 0.99), eps=1e-15, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(m * args.iters) for m in args.milestones], gamma=0.5)

    status.log(f"[{prob['name']}] visible={prob['visible']} target={prob['target']} "
               f"train={X.shape} predict={te_mm.shape} bbox_mm={bb.tolist()}")

    torch.cuda.reset_peak_memory_stats()
    model.train()
    avg, alpha = None, 0.999
    t0 = time.time()
    for i in range(1, args.iters + 1):
        batch = dataset.get_batch(args.batch_size, "cuda:0")
        losses = model(**batch)
        loss = losses[ntfm.DS_LOSS] + 1.0 * losses.get(ntfm.E_REG, 0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        cur = {"loss": float(loss), "mse": float(losses[ntfm.D_LOSS])}
        avg = cur if avg is None else {k: avg[k] * alpha + cur[k] * (1 - alpha) for k in cur}
        if i % 1000 == 0 or i == args.iters:
            status.log(f"  iter {i}/{args.iters} " + json.dumps({k: round(v, 6) for k, v in avg.items()})
                       + f" lr_enc={optimizer.param_groups[0]['lr']:.2e} net={optimizer.param_groups[1]['lr']:.2e}"
                       + f" {(time.time()-t0)/i*1000:.1f} ms/iter")
            status.write(stage="train", current_scenario=prob["name"], iter=i, max_iter=args.iters,
                         train_loss=round(avg["loss"], 6))
    train_sec = time.time() - t0
    alloc = torch.cuda.max_memory_allocated() / 2**20

    model.eval()
    model.inr.eval()
    t1 = time.time()
    pred = np.empty((te_mm.shape[0], X.shape[1]), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, te_mm.shape[0], 65536):
            res = sample_points(model.inr, torch.from_numpy(te_mm[s:s + 65536]).to("cuda:0"))
            pred[s:s + 65536] = res["expression"].float().cpu().numpy()
    infer_sec = time.time() - t1

    ins_rmse = None
    if args.in_sample:
        t2 = time.time()
        rec = np.empty_like(X)
        with torch.no_grad():
            for s in range(0, coords_mm.shape[0], 65536):
                res = sample_points(model.inr, torch.from_numpy(coords_mm[s:s + 65536]).to("cuda:0"))
                rec[s:s + 65536] = res["expression"].float().cpu().numpy()
        ins_rmse = float(np.sqrt(np.mean((rec - X) ** 2)))
        del rec
        infer_sec_in = time.time() - t2
        status.log(f"[{prob['name']}] in-sample visible RMSE {ins_rmse:.5f} ({infer_sec_in:.1f}s)")
    torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()

    path = hc.write_prediction(METHOD, prob, pred)
    res = hc.verify_prediction(path, METHOD, prob)
    res.update({
        "scenario": prob["name"], "k": prob["k"],
        "visible": prob["visible"], "target": prob["target"],
        "n_train_bins": int(X.shape[0]), "n_target_bins": int(prob["row_index"].size),
        "train_sec": round(train_sec, 1), "infer_sec": round(infer_sec, 1),
        "scenario_sec": round(train_sec + infer_sec, 1),
        "torch_alloc_peak_mb": round(alloc, 1),
        "final_loss": round(float(avg["loss"]), 6),
        "in_sample_visible_rmse": ins_rmse,
    })
    cfg = dict(iters=args.iters, batch_size=args.batch_size, enc_lr=args.enc_lr,
               net_lr=args.net_lr, milestones=args.milestones, gamma=0.5,
               n_levels=args.n_levels, log2_hashmap_size=args.log2_hashmap_size,
               n_samples=args.n_samples, bounding_box_mm=bb.tolist(),
               psf_resolution_mm=[bs, bs, bs], single_precision=True, no_dropout=True)
    return res, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--enc-lr", type=float, default=1e-2)
    ap.add_argument("--net-lr", type=float, default=1e-3)
    ap.add_argument("--n-levels", type=int, default=13)
    ap.add_argument("--log2-hashmap-size", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--milestones", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--in-sample", action="store_true", default=True)
    ap.add_argument("--no-in-sample", dest="in_sample", action="store_false")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    status = hc.Status(METHOD, OUTDIR)
    hc.set_method(METHOD, "RUNNING", stage="load canonical", lane=hc.LANE)
    mon = hc.ResourceMonitor(period=3.0, use_gpu=True)
    mon.start()
    try:
        scen_all = hc.load_scenarios()
        if args.scenario:
            sel = [s for s in scen_all if s["name"] == args.scenario]
            assert sel, f"unknown scenario {args.scenario}"
        elif args.all:
            sel = scen_all
        else:
            raise SystemExit("pass --scenario NAME or --all")
        bins = hc.load_bins()
        tmeta = hc.load_target_meta()
        status.log(f"canonical bins={bins['X'].shape} target_rows={tmeta['n_rows']} n_scen={len(sel)}")
        status.write(stage="ready", scenarios_total=len(sel))

        per, cfg = {}, None
        t_all = time.time()
        for i, scen in enumerate(sel, 1):
            prob_meta = hc.build_problem(scen, bins, tmeta)
            if args.skip_existing and hc.existing_ok(METHOD, prob_meta):
                status.log(f"[{scen['name']}] skip (existing valid prediction)")
                per[scen["name"]] = {"skipped": True}
                continue
            status.write(stage="scenario", current_scenario=scen["name"], scenario_index=i,
                         scenarios_done=len(per))
            res, cfg = run_scenario(scen, bins, tmeta, args, status, mon)
            per[scen["name"]] = res
            status.scenario_done(scen["name"], res)
            hc.set_method(METHOD, "RUNNING", scenarios_done=len(per), scenarios_total=len(sel),
                          latest=res["scenario"], coverage=1.0)
            status.log(f"[{scen['name']}] wrote {res['path']} {res['shape']} "
                       f"{res['bytes']/1e6:.1f} MB in {res['scenario_sec']:.0f}s")

        mon.stop()
        total = time.time() - t_all
        sizes = sum(v.get("bytes", 0) for v in per.values() if isinstance(v, dict))
        metrics = {
            "method": METHOD, "dataset": "heart_10slice",
            "model": "NTF GeneINR hash-grid INR + slice-conditional field, 238 genes",
            "input_mode": "canonical 50um bins, visible sections only",
            "output": "predictions/<scenario>/NTF.npz (target bins only, no volume)",
            "scenarios_run": len(per),
            "scenarios_skipped": sum(1 for v in per.values() if v.get("skipped")),
            "config": cfg, "total_sec": round(total, 1),
            "sum_scenario_sec": round(sum(v.get("scenario_sec", 0) for v in per.values()), 1),
            "runtime_sec_per_scenario_mean": round(np.mean(
                [v["scenario_sec"] for v in per.values() if "scenario_sec" in v]) if per else 0, 1),
            "output_bytes_total": sizes,
            "coverage_mean": 1.0, "coverage_min": 1.0,
            "per_scenario": per, "resources": mon.summary(),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with open(os.path.join(OUTDIR, "compute_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        status.finish("DONE", scenarios_done=len(per), total_sec=round(total, 1),
                      resources=mon.summary())
        hc.set_method(METHOD, "DONE", scenarios_done=len(per), scenarios_total=len(sel),
                      coverage=1.0, metrics="methods/NTF/compute_metrics.json",
                      output_bytes_total=sizes, runtime_sec=round(total, 1),
                      resources=mon.summary(), failures=0)
        status.log(f"DONE {len(per)} scenarios in {total/60:.1f} min")
    except Exception:
        tb = traceback.format_exc()
        with open(status.logpath, "a") as f:
            f.write(tb + "\n")
        print(tb, flush=True)
        status.finish("FAILED", error=tb[-2000:])
        hc.set_method(METHOD, "FAILED", error=tb[-500:])
        raise


if __name__ == "__main__":
    main()

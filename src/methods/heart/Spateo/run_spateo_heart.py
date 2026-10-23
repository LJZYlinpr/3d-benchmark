#!/usr/bin/env python
"""Spateo DeepInterpolation for the heart 10-slice target-only benchmark.

Per scenario (26 contiguous hidden blocks, k=1..4):
  * train only on bins of the visible sections
  * predict only the hidden target bins listed in canonical/target_gt.npz
  * write predictions/<scenario>/Spateo.npz  (pred aligned to that scenario's rows)

Adapted from Workspace/merfish_test/Spateo/run_spateo_50um.py (same model,
loss, optimiser, coordinate normalisation, and float64 GPU training loop).
No target expression is read at any point; only canonical geometry is used.
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

# spateo imports statsmodels.sandbox.stats.multicomp.multipletests at module load;
# this env only has statsmodels.stats.multitest.multipletests. Same shim as
# Workspace/merfish_test/Spateo/run_spateo_50um.py.
import statsmodels.sandbox.stats.multicomp as _mc
from statsmodels.stats.multitest import multipletests as _multipletests
_mc.multipletests = _multipletests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _heart_common as hc  # noqa: E402

METHOD = "Spateo"
OUTDIR = os.path.join(hc.METHODS_ROOT, "Spateo")


def predict_chunks(model, coords_norm, device, chunk=30000):
    import torch

    out = np.empty((coords_norm.shape[0], model.output_network_dim), dtype=np.float64)
    model.main_flow_func.eval()
    with torch.no_grad():
        for i in range(0, coords_norm.shape[0], chunk):
            xb = torch.as_tensor(coords_norm[i:i + chunk], dtype=torch.float64, device=device)
            out[i:i + chunk] = (model.main_flow_func(t=None, x=xb) / model.normalization_factor).cpu().numpy()
    model.main_flow_func.train()
    return out


def train_gpu(model, coords, expr, iters, batch_size, lr, lr_min, device="cuda", seed=0,
              log_every=1000, log=None, monitor_cb=None):
    """Spateo DeepInterpolation.train loop on GPU in float64, no regularisers."""
    import torch
    import torch.optim as optim

    torch.manual_seed(seed)
    X = torch.as_tensor(coords, dtype=torch.float64, device=device)
    Y = torch.as_tensor(expr, dtype=torch.float64, device=device)
    n = X.shape[0]
    model.main_flow_func = model.main_flow_func.to(device).double()
    model.optimizer = optim.Adam(model.h.parameters(), lr=lr, betas=(0.5, 0.9),
                                 weight_decay=2.5e-5, amsgrad=True)
    loss_fn = model.loss_function
    losses = []
    t0 = time.time()
    model.main_flow_func.train()
    for it in range(iters):
        if lr_min is not None:
            frac = it / max(1, iters - 1)
            cur_lr = lr_min + 0.5 * (lr - lr_min) * (1 + np.cos(np.pi * frac))
            for g in model.optimizer.param_groups:
                g["lr"] = cur_lr
        idx = torch.randint(0, n, (batch_size,), device=device)
        model.optimizer.zero_grad(set_to_none=True)
        pred = model.main_flow_func(t=None, x=X[idx])
        loss = loss_fn(pred, Y[idx], None)
        loss.backward()
        model.optimizer.step()
        losses.append(float(loss.detach()))
        if (it + 1) % log_every == 0 or it == 0:
            log(f"  iter {it+1:6d}/{iters} loss {np.mean(losses[-log_every:]):.5f} "
                f"lr {model.optimizer.param_groups[0]['lr']:.2e} "
                f"alloc {torch.cuda.max_memory_allocated()/2**20:.0f}MiB "
                f"{(time.time()-t0)/(it+1)*1000:.1f} ms/iter")
            if monitor_cb is not None:
                monitor_cb(it + 1, iters, float(np.mean(losses[-log_every:])))
    return np.asarray(losses), time.time() - t0


def run_scenario(scen, bins, tmeta, args, status, mon):
    import torch
    from spateo.tdr.interpolations.interpolation_deeplearn import (
        DataSampler, DeepInterpolation, interpolation_nn)
    from spateo.tdr.interpolations.interpolation_deeplearn.nn_losses import weighted_mse

    prob = hc.build_problem(scen, bins, tmeta)
    # leakage guards: no visible/target overlap, training rows never target sections
    assert not (set(prob["visible"]) & set(prob["target"]))
    assert not np.isin(bins["section"][prob["tr_idx"]], prob["target"]).any()

    tr, te = prob["coords_tr"], prob["coords_te"]
    lo = np.minimum(tr.min(0), te.min(0))
    hi = np.maximum(tr.max(0), te.max(0))
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    tr_n = 2.0 * (tr - lo) / span - 1.0
    te_n = 2.0 * (te - lo) / span - 1.0
    expr = prob["expr_tr"].astype(np.float64)

    status.log(f"[{prob['name']}] visible={prob['visible']} target={prob['target']} "
               f"train={tr.shape} predict={te.shape}")
    t0 = time.time()
    cfg = dict(iters=args.iters, batch_size=args.batch_size, lr=args.lr, lr_min=args.lr_min,
               hidden_features=args.hidden_features, hidden_layers=args.hidden_layers,
               dtype="float64", optimizer="Adam(betas=(0.5,0.9), wd=2.5e-5, amsgrad=True)",
               lr_schedule="cosine", smoothing_factor=None, stability_factor=None,
               enforce_positivity=False, sirens=False, seed=args.seed,
               coordinates="per-axis min-max [-1,1] over visible+target extent (geometry only)")
    if args.pilot_iters > 0:
        pidx = np.random.default_rng(0).choice(tr.shape[0], size=min(8000, tr.shape[0]), replace=False)
        pmod = DeepInterpolation(model=interpolation_nn,
                                 data_sampler=DataSampler(data={"X": tr_n[pidx], "Y": expr[pidx]}),
                                 sirens=False, enforce_positivity=False, loss_function=weighted_mse(),
                                 smoothing_factor=None, stability_factor=None,
                                 buffer_path=os.path.join(OUTDIR, "buffers", "pilot"),
                                 hidden_features=args.hidden_features,
                                 hidden_layers=args.hidden_layers)
        _, psec = train_gpu(pmod, tr_n[pidx], expr[pidx], args.pilot_iters, args.batch_size,
                            args.lr, None, log_every=max(1, args.pilot_iters // 4),
                            log=status.log, seed=args.seed)
        status.log(f"[{prob['name']}] pilot {args.pilot_iters} iters: "
                   f"{psec/max(1,args.pilot_iters)*1000:.2f} ms/iter -> "
                   f"projected full {psec/max(1,args.pilot_iters)*args.iters/60:.1f} min")
        del pmod
        torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    model = DeepInterpolation(model=interpolation_nn,
                              data_sampler=DataSampler(data={"X": tr_n, "Y": expr}),
                              sirens=False, enforce_positivity=False, loss_function=weighted_mse(),
                              smoothing_factor=None, stability_factor=None,
                              buffer_path=os.path.join(OUTDIR, "buffers", prob["name"]),
                              hidden_features=args.hidden_features,
                              hidden_layers=args.hidden_layers)

    def mon_cb(it, total, loss):
        if it % 5000 == 0 or it == total:
            status.write(stage="train", current_scenario=prob["name"], iter=it, max_iter=total,
                         train_loss=round(loss, 6))
            hc.set_method(METHOD, "RUNNING", stage=f"train {prob['name']}", scenarios_done=status.state["scenarios_done"])

    losses, train_sec = train_gpu(model, tr_n, expr, args.iters, args.batch_size, args.lr,
                                  args.lr_min, log=status.log, monitor_cb=mon_cb, seed=args.seed)
    t_train = time.time() - t0
    # in-sample reconstruction on visible bins (sanity only; target GT never used)
    ins_rmse = None
    if args.in_sample:
        rec = np.clip(predict_chunks(model, tr_n, "cuda"), 0, None)
        ins_rmse = float(np.sqrt(np.mean((rec - prob["expr_tr"].astype(np.float64)) ** 2)))
        del rec
    pred = np.clip(predict_chunks(model, te_n, "cuda"), 0, None)
    infer_sec = time.time() - t0 - t_train
    alloc = torch.cuda.max_memory_allocated() / 2**20
    del model
    torch.cuda.empty_cache()

    path = hc.write_prediction(METHOD, prob, pred)
    res = hc.verify_prediction(path, METHOD, prob)
    res.update({
        "scenario": prob["name"], "k": prob["k"],
        "visible": prob["visible"], "target": prob["target"],
        "n_train_bins": int(prob["tr_idx"].size), "n_target_bins": int(prob["row_index"].size),
        "train_sec": round(train_sec, 1), "scenario_sec": round(time.time() - t0, 1),
        "infer_sec": round(infer_sec, 1),
        "torch_alloc_peak_mb": round(alloc, 1),
        "final_loss": round(float(losses[-200:].mean()), 6),
        "in_sample_visible_rmse": ins_rmse,
    })
    return res, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--iters", type=int, default=50000)
    ap.add_argument("--pilot-iters", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--hidden-features", type=int, default=512)
    ap.add_argument("--hidden-layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--in-sample", action="store_true", default=True)
    ap.add_argument("--no-in-sample", dest="in_sample", action="store_false")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, "buffers"), exist_ok=True)
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
        status.log(f"canonical: bins={bins['X'].shape} target_rows={tmeta['n_rows']} "
                   f"scenarios={len(sel)} genes={bins['genes'].size}")
        status.write(stage="ready", scenarios_total=len(sel))

        per = {}
        t_all = time.time()
        for i, scen in enumerate(sel, 1):
            prob_meta = hc.build_problem(scen, bins, tmeta)
            if args.skip_existing and hc.existing_ok(METHOD, prob_meta):
                status.log(f"[{scen['name']}] skip (existing valid prediction)")
                per[scen["name"]] = {"skipped": True}
                continue
            status.write(stage="scenario", current_scenario=scen["name"],
                         scenarios_done=len(per), scenario_index=i)
            res, cfg = run_scenario(scen, bins, tmeta, args, status, mon)
            # checkpoint the model for the in-sample check, then drop it (space)
            per[scen["name"]] = res
            status.scenario_done(scen["name"], res)
            hc.set_method(METHOD, "RUNNING", scenarios_done=len(per),
                          scenarios_total=len(sel), input_mode="target-only npz",
                          latest=res["scenario"], coverage=1.0)
            status.log(f"[{scen['name']}] wrote {res['path']} {res['shape']} "
                       f"{res['bytes']/1e6:.1f} MB in {res['scenario_sec']:.0f}s")

        mon.stop()
        total = time.time() - t_all
        sizes = sum(v.get("bytes", 0) for v in per.values() if isinstance(v, dict))
        metrics = {
            "method": METHOD,
            "dataset": "heart_10slice",
            "model": "Spateo DeepInterpolation (leaky-ReLU MLP 3->512x4->238), weighted MSE",
            "input_mode": "canonical 50um bins, visible sections only",
            "output": "predictions/<scenario>/Spateo.npz (target bins only, no volume)",
            "scenarios_run": len(per),
            "scenarios_skipped": sum(1 for v in per.values() if v.get("skipped")),
            "config": cfg if per else None,
            "total_sec": round(total, 1),
            "sum_scenario_sec": round(sum(v.get("scenario_sec", 0) for v in per.values()), 1),
            "runtime_sec_per_scenario_mean": round(
                np.mean([v["scenario_sec"] for v in per.values() if "scenario_sec" in v])
                if per else 0, 1),
            "output_bytes_total": sizes,
            "coverage_mean": 1.0,
            "coverage_min": 1.0,
            "per_scenario": per,
            "resources": mon.summary(),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with open(os.path.join(OUTDIR, "compute_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        status.finish("DONE", scenarios_done=len(per), total_sec=round(total, 1),
                      resources=mon.summary())
        hc.set_method(METHOD, "DONE", scenarios_done=len(per),
                      scenarios_total=len(sel), coverage=1.0,
                      metrics="methods/Spateo/compute_metrics.json",
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

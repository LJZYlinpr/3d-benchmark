#!/usr/bin/env python
"""Spateo DeepInterpolation on the MERFISH 50 um benchmark grid.

Fidelity notes
--------------
* Model, loss and optimiser are Spateo's own (`DeepInterpolation`,
  `interpolation_nn`, `nn_losses.weighted_mse`).  Spateo's `train()` has no
  device handling (it hard-codes CPU tensors and float64), so the identical
  optimisation loop is re-implemented here to run on one A100 in float64.
  No smoothing / Lyapunov-stability regulariser (both default to `True` in
  Spateo, where they are meant for velocity fields); expression fitting only
  needs the weighted MSE term.  Documented in metrics.json.
* Coordinates are min-max normalised per axis to [-1, 1] (input to the MLP).
* Prediction is chunked (30k points) on GPU; negatives are clipped to 0 so the
  output is a valid expression matrix (consistent with the other baselines).
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback

import statsmodels.sandbox.stats.multicomp as mc
from statsmodels.stats.multitest import multipletests

mc.multipletests = multipletests

import numpy as np

T0 = time.time()

ROOT = "/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/Workspace/merfish_test"
OUTDIR = os.path.join(ROOT, "Spateo")
GRID = os.path.join(ROOT, "grid_50um.npz")
OBS = os.path.join(ROOT, "observed_50um.h5ad")
STATUS_PATH = os.path.join(OUTDIR, "status.json")
LOG_PATH = os.path.join(OUTDIR, "run.log")

N_GENES = 1122
STATE = {
    "method": "Spateo-DeepInterpolation",
    "dataset": "merfish_3d_recon_50um",
    "voxel_um": 50,
    "status": "RUNNING",
    "stage": "init",
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "pid": os.getpid(),
    "host": os.uname().nodename,
    "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
    "elapsed_sec": 0.0,
    "gpu_peak_mb": 0,
    "gpu_alloc_peak_mb": 0.0,
    "gpu_reserved_peak_mb": 0.0,
}
_LOCK = threading.Lock()


def _write_status():
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(STATE, f, indent=2, ensure_ascii=False)
    os.replace(tmp, STATUS_PATH)


def set_status(**kw):
    with _LOCK:
        STATE.update(kw)
        STATE["elapsed_sec"] = round(time.time() - T0, 1)
        _write_status()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:7.1f}s] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


class GpuMonitor(threading.Thread):
    """Device-wide nvidia-smi peak (co-tenant lanes included) + torch peaks."""

    def __init__(self, period=5.0):
        super().__init__(daemon=True)
        self.period = period
        self.stop_evt = threading.Event()
        self.peak_mb = 0

    def _poll(self):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            vals = [int(v) for v in r.stdout.split() if v.strip().isdigit()]
            if vals:
                self.peak_mb = max(self.peak_mb, max(vals))
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                with _LOCK:
                    STATE["gpu_alloc_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
                    STATE["gpu_reserved_peak_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        except Exception:
            pass

    def run(self):
        while not self.stop_evt.is_set():
            self._poll()
            with _LOCK:
                STATE["gpu_peak_mb"] = max(STATE.get("gpu_peak_mb", 0), self.peak_mb)
            self.stop_evt.wait(self.period)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=15)
        self._poll()
        with _LOCK:
            STATE["gpu_peak_mb"] = max(STATE.get("gpu_peak_mb", 0), self.peak_mb)


def predict_chunks(model, coords_norm, device, chunk=30000):
    import torch

    out = np.empty((coords_norm.shape[0], model.output_network_dim), dtype=np.float64)
    model.main_flow_func.eval()
    with torch.no_grad():
        for i in range(0, coords_norm.shape[0], chunk):
            xb = torch.as_tensor(coords_norm[i : i + chunk], dtype=torch.float64, device=device)
            yb = model.main_flow_func(t=None, x=xb) / model.normalization_factor
            out[i : i + chunk] = yb.detach().cpu().numpy()
    model.main_flow_func.train()
    return out


def train_gpu(model, coords, expr, iters, batch_size, lr, lr_min, log_every=200,
              device="cuda", seed=0, monitor_cb=None):
    """Spateo `DeepInterpolation.train` loop, on GPU, float64, no regularisers."""
    import torch
    import torch.optim as optim

    torch.manual_seed(seed)
    X = torch.as_tensor(coords, dtype=torch.float64, device=device)
    Y = torch.as_tensor(expr, dtype=torch.float64, device=device)
    n = X.shape[0]

    model.main_flow_func = model.main_flow_func.to(device).double()
    model.optimizer = optim.Adam(
        model.h.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=2.5e-5, amsgrad=True
    )
    loss_fn = model.loss_function
    losses = []
    t_start = time.time()
    model.main_flow_func.train()

    for it in range(iters):
        if lr_min is not None:
            frac = it / max(1, iters - 1)
            cur_lr = lr_min + 0.5 * (lr - lr_min) * (1 + np.cos(np.pi * frac))
            for g in model.optimizer.param_groups:
                g["lr"] = cur_lr

        idx = torch.randint(0, n, (batch_size,), device=device)
        xb, yb = X[idx], Y[idx]

        model.optimizer.zero_grad(set_to_none=True)
        pred = model.main_flow_func(t=None, x=xb)
        loss = loss_fn(pred, yb, None)
        loss.backward()
        model.optimizer.step()

        losses.append(float(loss.detach()))
        if (it + 1) % log_every == 0 or it == 0:
            log(
                f"  iter {it+1:6d}/{iters}  loss {np.mean(losses[-log_every:]):.5f}  "
                f"lr {model.optimizer.param_groups[0]['lr']:.2e}  "
                f"gpu_alloc {torch.cuda.max_memory_allocated()/2**20:.0f}MiB  "
                f"{(time.time()-t_start)/(it+1)*1000:.1f} ms/iter"
            )
            if monitor_cb is not None:
                monitor_cb(it + 1, iters, float(np.mean(losses[-log_every:])))
    return np.asarray(losses), time.time() - t_start


def pcc_stats(pred, truth):
    pred = pred.astype(np.float64)
    truth = truth.astype(np.float64)
    # pooled over all entries
    a = pred.ravel() - pred.mean()
    b = truth.ravel() - truth.mean()
    pooled = float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    # per gene
    pm, tm = pred.mean(0), truth.mean(0)
    pc = pred - pm
    tc = truth - tm
    denom = np.linalg.norm(pc, axis=0) * np.linalg.norm(tc, axis=0)
    ok = denom > 1e-12
    r = (pc * tc).sum(0)[ok] / denom[ok]
    mse = float(np.mean((pred - truth) ** 2))
    return {
        "pearson_pooled": pooled,
        "pearson_gene_mean": float(r.mean()) if r.size else None,
        "pearson_gene_median": float(np.median(r)) if r.size else None,
        "pearson_gene_p10": float(np.percentile(r, 10)) if r.size else None,
        "n_genes_with_corr": int(r.size),
        "mse_all": mse,
        "rmse_all": float(np.sqrt(mse)),
        "pred_mean": float(pred.mean()),
        "gt_mean": float(truth.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--pilot-iters", type=int, default=600)
    ap.add_argument("--pilot-samples", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-min", type=float, default=1e-4)
    ap.add_argument("--hidden-features", type=int, default=256)
    ap.add_argument("--hidden-layers", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sirens", action="store_true")
    ap.add_argument("--first-omega", type=float, default=30.0)
    ap.add_argument("--hidden-omega", type=float, default=30.0)
    ap.add_argument("--holdout-iz", type=int, default=None,
                    help="diagnostic: drop this observed z layer from training and score it (leave-one-slice-out)")
    ap.add_argument("--no-write", action="store_true", help="diagnostic: skip the (2.3 GB) h5ad write")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    out_h5 = os.path.join(OUTDIR, "Spateo_50um_uniform.h5ad")
    metrics_path = os.path.join(OUTDIR, "metrics.json")
    if args.tag:
        globals()["STATUS_PATH"] = os.path.join(OUTDIR, f"status_{args.tag}.json")
        globals()["LOG_PATH"] = os.path.join(OUTDIR, f"run_{args.tag}.log")
        STATE["tag"] = args.tag
        out_h5 = os.path.join(OUTDIR, f"Spateo_50um_{args.tag}.h5ad")
        metrics_path = os.path.join(OUTDIR, f"metrics_{args.tag}.json")

    os.makedirs(OUTDIR, exist_ok=True)
    mon = GpuMonitor()
    mon.start()
    set_status(stage="import")

    import torch
    import anndata as ad
    from spateo.tdr.interpolations.interpolation_deeplearn import DataSampler, DeepInterpolation, interpolation_nn
    from spateo.tdr.interpolations.interpolation_deeplearn.nn_losses import weighted_mse

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"torch {torch.__version__} device={device} nthreads={torch.get_num_threads()}")
    set_status(device=device, torch=torch.__version__)

    # ---------------- load ----------------
    set_status(stage="load")
    t_load = time.time()
    g = np.load(GRID)
    xc, yc, zc = g["x_centers"], g["y_centers"], g["z_centers"]
    obs_mask, q_mask, inside = g["obs_mask"], g["query_mask"], g["inside"]
    nx, ny, nz = len(xc), len(yc), len(zc)
    obs = ad.read_h5ad(OBS)
    genes = np.asarray(obs.var_names).astype(str)
    assert genes.size == N_GENES, genes.size
    Xobs = obs.X.toarray() if hasattr(obs.X, "toarray") else np.asarray(obs.X)
    Xobs = np.ascontiguousarray(Xobs, dtype=np.float64)

    vox_obs = (obs.obs["iz"].values * nx * ny + obs.obs["iy"].values * nx + obs.obs["ix"].values).astype(np.int64)
    lut = np.full(nx * ny * nz, -1, dtype=np.int64)
    lut[vox_obs] = np.arange(obs.n_obs)

    iz_i, iy_i, ix_i = np.nonzero(inside)  # C order: iz, iy, ix
    vox_in = iz_i * nx * ny + iy_i * nx + ix_i
    is_obs = obs_mask[iz_i, iy_i, ix_i]
    row_in_lut = lut[vox_in]
    assert np.array_equal(is_obs, row_in_lut >= 0), "inside/obs_mask disagree with observed rows"
    n_out, n_query = vox_in.size, int(q_mask.sum())
    log(f"grid {nz}x{ny}x{nx} inside={n_out} observed_in={int(is_obs.sum())} query={n_query}")
    assert n_out == 523163 and n_query == 400677

    coords_mm = np.stack([xc[ix_i], yc[iy_i], zc[iz_i]], axis=1).astype(np.float64)

    lo = np.array([xc.min(), yc.min(), zc.min()])
    hi = np.array([xc.max(), yc.max(), zc.max()])
    coords_norm = 2.0 * (coords_mm - lo) / (hi - lo) - 1.0

    # slice label: nearest observed z layer (ties -> lower iz), matching other baselines
    z_layers = sorted(set(np.unique(obs.obs["iz"].values).tolist()))
    layer_slice = {iz: str(obs.obs.loc[obs.obs["iz"] == iz, "slice"].iloc[0]) for iz in z_layers}
    slice_of_iz = []
    for iz in range(nz):
        d = [abs(iz - L) for L in z_layers]
        slice_of_iz.append(layer_slice[z_layers[int(np.argmin(d))]])
    slice_names = np.array([slice_of_iz[iz] for iz in iz_i])

    if args.holdout_iz is None:
        tr_rows = np.nonzero(is_obs)[0]
        ho_rows = np.array([], dtype=np.int64)
    else:
        ho_mask = is_obs & (iz_i == args.holdout_iz)
        tr_rows = np.nonzero(is_obs & ~ho_mask)[0]
        ho_rows = np.nonzero(ho_mask)[0]
        log(f"HOLDOUT diagnostic: iz={args.holdout_iz} -> {ho_rows.size} eval points, "
            f"{tr_rows.size} training points")
    tr_norm = coords_norm[tr_rows]
    tr_expr = Xobs[row_in_lut[tr_rows]]
    log(f"load {time.time()-t_load:.1f}s  train={tr_norm.shape}  query={n_query}")
    set_status(stage="load", n_out=n_out, n_query=n_query, n_observed_train=int(is_obs.sum()))

    # ---------------- pilot ----------------
    set_status(stage="pilot")
    cfg = dict(
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_min=args.lr_min,
        lr_schedule="cosine",
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        dtype="float64",
        optimizer="Adam(betas=(0.5,0.9), weight_decay=2.5e-5, amsgrad=True)",
        smoothing_factor=None,
        stability_factor=None,
        enforce_positivity=False,
        coordinates="per-axis min-max normalised to [-1,1] (grid extent)",
        clip_negative=True,
        train_loop_note=(
            "Spateo DeepInterpolation.train() has no device support (hard-coded CPU float64 tensors); "
            "same model/loss/optimiser/regulariser-off loop re-implemented for one A100 in float64"
        ),
        seed=args.seed,
        sirens=bool(args.sirens),
    )
    set_status(config=cfg, stage="pilot")
    log("pilot start")
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(args.seed)
    pidx = rng.choice(tr_norm.shape[0], size=min(args.pilot_samples, tr_norm.shape[0]), replace=False)
    pilot = DeepInterpolation(
        model=interpolation_nn,
        data_sampler=DataSampler(data={"X": tr_norm[pidx], "Y": tr_expr[pidx]}),
        sirens=args.sirens,
        enforce_positivity=False,
        loss_function=weighted_mse(),
        smoothing_factor=None,
        stability_factor=None,
        buffer_path=os.path.join(OUTDIR, "model_buffer_pilot"),
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        first_omega_0=args.first_omega,
        hidden_omega_0=args.hidden_omega,
    )
    _, pilot_sec = train_gpu(
        pilot, tr_norm[pidx], tr_expr[pidx], args.pilot_iters, args.batch_size,
        args.lr, None, log_every=100, device=device, seed=args.seed,
    )
    pilot_mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    pil_pred = predict_chunks(pilot, tr_norm[pidx[:8000]], device, chunk=args.chunk)
    pil_stats = pcc_stats(np.clip(pil_pred, 0, None), tr_expr[pidx[:8000]])
    sec_per_iter = pilot_sec / args.pilot_iters
    log(
        f"pilot done: {sec_per_iter*1000:.1f} ms/iter, torch peak {pilot_mem:.0f}MiB, "
        f"in-sample pooled PCC {pil_stats['pearson_pooled']:.4f}; "
        f"projected full {args.iters*sec_per_iter/60:.1f} min"
    )
    set_status(
        stage="pilot_done",
        pilot=dict(sec_per_iter=sec_per_iter, torch_peak_mb=round(pilot_mem, 1), stats=pil_stats),
    )
    del pilot
    torch.cuda.empty_cache()

    # ---------------- full training ----------------
    set_status(stage="train", iter=0, max_iter=args.iters)
    log(f"full training: {args.iters} iters, batch {args.batch_size}, lr {args.lr}->{args.lr_min}, "
        f"hidden {args.hidden_features}x{args.hidden_layers}, n={tr_norm.shape[0]}")
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = DeepInterpolation(
        model=interpolation_nn,
        data_sampler=DataSampler(data={"X": tr_norm, "Y": tr_expr}),
        sirens=args.sirens,
        enforce_positivity=False,
        loss_function=weighted_mse(),
        smoothing_factor=None,
        stability_factor=None,
        buffer_path=os.path.join(OUTDIR, "model_buffer"),
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        first_omega_0=args.first_omega,
        hidden_omega_0=args.hidden_omega,
    )
    def mon_cb(it, total, loss):
        set_status(iter=it, max_iter=total, train_loss=round(loss, 6), stage="train")

    losses, train_sec = train_gpu(
        model, tr_norm, tr_expr, args.iters, args.batch_size, args.lr, args.lr_min,
        log_every=200, device=device, seed=args.seed, monitor_cb=mon_cb,
    )
    loss_log = os.path.join(OUTDIR, f"train_loss_{args.tag}.csv" if args.tag else "train_loss.csv")
    np.savetxt(loss_log, losses, fmt="%.6f")
    torch.save(model.h.state_dict(), os.path.join(OUTDIR, "spateo_h_state_dict.pt"))
    log(f"training done in {train_sec/60:.2f} min, final loss(mean last 200) {losses[-200:].mean():.5f}")
    set_status(stage="train_done", train_sec=round(train_sec, 1), train_loss=round(float(losses[-200:].mean()), 6))

    # ---------------- in-sample eval ----------------
    set_status(stage="eval")
    t_ev = time.time()
    obs_rows = np.nonzero(is_obs)[0]
    if ho_rows.size:
        ev_rows, ev_truth = ho_rows, Xobs[row_in_lut[ho_rows]]
        ev_label = f"HOLDOUT iz={args.holdout_iz}"
    else:
        ev_rows, ev_truth = tr_rows, tr_expr
        ev_label = f"in-sample ({ev_rows.size} obs pts)"
    pred_obs = predict_chunks(model, coords_norm[ev_rows], device, chunk=args.chunk)
    ins = pcc_stats(np.clip(pred_obs, 0, None), ev_truth)
    ins["eval_sec"] = round(time.time() - t_ev, 1)
    ins["n_points"] = int(ev_rows.size)
    log(
        f"{ev_label}: pooled PCC {ins['pearson_pooled']:.4f}, "
        f"gene-mean PCC {ins['pearson_gene_mean']:.4f}, RMSE {ins['rmse_all']:.4f}"
    )
    if ho_rows.size:
        ins["holdout_iz"] = args.holdout_iz
        ins["note"] = "layer never seen in training; measures z-interpolation, not memorisation"

    # ---------------- query inference ----------------
    set_status(stage="infer")
    q_rows = np.nonzero(~is_obs)[0]
    t_inf = time.time()
    pred_q = predict_chunks(model, coords_norm[q_rows], device, chunk=args.chunk)
    infer_sec = time.time() - t_inf
    infer_peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    log(f"query inference {pred_q.shape} in {infer_sec:.1f}s, torch peak {infer_peak:.0f}MiB")

    # ---------------- assemble + write ----------------
    set_status(stage="write")
    t_w = time.time()
    Xout = np.zeros((n_out, N_GENES), dtype=np.float32)
    Xout[q_rows] = np.clip(pred_q, 0, None).astype(np.float32)
    Xout[obs_rows] = Xobs[row_in_lut[obs_rows]].astype(np.float32)
    del pred_q

    qpred = Xout[q_rows]
    q_std = qpred.std(axis=1)
    metrics = {
        "method": "Spateo-DeepInterpolation",
        "dataset": "merfish_3d_recon_50um",
        "voxel_um": 50,
        "train_sec": round(train_sec, 1),
        "pilot_sec": round(pilot_sec, 1),
        "infer_sec": round(infer_sec, 1),
        "eval_sec": ins["eval_sec"],
        "write_sec": None,
        "load_sec": round(t_load, 1),
        "total_sec": None,
        "gpu_peak_mb": STATE.get("gpu_peak_mb", 0),
        "gpu_alloc_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else 0.0,
        "gpu_reserved_peak_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1) if device == "cuda" else 0.0,
        "gpu_peak_note": "gpu_peak_mb = device-wide nvidia-smi peak incl. co-tenant xpy job; alloc/reserved are this process",
        "n_query": n_query,
        "n_observed": int(is_obs.sum()),
        "n_observed_input": int(obs.n_obs),
        "n_observed_dropped_outside_mask": int(obs.n_obs - is_obs.sum()),
        "n_out": n_out,
        "n_genes": N_GENES,
        "query_finite_frac": float(np.isfinite(qpred).mean()),
        "all_finite_frac": float(np.isfinite(Xout).mean()),
        "nan_count": int((~np.isfinite(Xout)).sum()),
        "query_pred_min": float(qpred.min()),
        "query_pred_max": float(qpred.max()),
        "query_pred_mean": float(qpred.mean()),
        "query_constant_rows": int((q_std == 0).sum()),
        "query_row_std_mean": float(q_std.mean()),
        "obs_match_input": bool(np.array_equal(Xout[obs_rows], tr_expr.astype(np.float32))),
        "in_sample": ins,
        "pilot": STATE.get("pilot", {}),
        "config": cfg,
        "settings_notes": [
            "Spateo DeepInterpolation (3->256->256->256->1122 MLP, leaky-ReLU, no SIREN), weighted MSE loss",
            "smoothing_factor / stability_factor disabled: Spateo defaults were built for velocity fields; "
            "stability term would push predictions away from expression",
            "training on 122,486 observed voxels (all 5 observed z layers, 1122 genes) - no GP",
            "query = 400,677 grid voxels interior to the brain mask, predicted in 30k chunks",
            "negative predictions clipped to 0 (expression); observed rows keep raw input values",
            "coords in mm in obs/h5ad; model input = per-axis normalised [-1,1]",
        ],
        "host": os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
    }

    adata = ad.AnnData(X=Xout)
    adata.obs_names = [f"v{ix}_{iy}_{iz}" for ix, iy, iz in zip(ix_i, iy_i, iz_i)]
    adata.var_names = genes
    adata.var["gene_name"] = genes
    adata.obs["x_mm"] = coords_mm[:, 0].astype(np.float32)
    adata.obs["y_mm"] = coords_mm[:, 1].astype(np.float32)
    adata.obs["z_mm"] = coords_mm[:, 2].astype(np.float32)
    adata.obs["ix"] = ix_i.astype(np.int32)
    adata.obs["iy"] = iy_i.astype(np.int32)
    adata.obs["iz"] = iz_i.astype(np.int32)
    adata.obs["is_observed"] = is_obs
    adata.obs["slice"] = slice_names
    adata.obsm["spatial"] = coords_mm.astype(np.float32)
    adata.obsm["X_aligned"] = coords_mm.astype(np.float32)
    adata.uns["spateo"] = {
        "method": "DeepInterpolation",
        "hidden_features": args.hidden_features,
        "hidden_layers": args.hidden_layers,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "lr": "%.1e->%.1e cosine" % (args.lr, args.lr_min),
        "regularisers": "none",
        "coord_norm": "per-axis min-max [-1,1]",
        "bbox_mm": [lo.tolist(), hi.tolist()],
    }
    out = out_h5
    if not args.no_write:
        adata.write_h5ad(out, compression=None)
    metrics["write_sec"] = round(time.time() - t_w, 1)
    metrics["total_sec"] = round(time.time() - T0, 1)
    metrics["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    mon.stop()
    set_status(
        status="DONE",
        stage="done",
        iter=args.iters,
        gpu_peak_mb=STATE.get("gpu_peak_mb", 0),
        train_sec=round(train_sec, 1),
        infer_sec=round(infer_sec, 1),
        n_query=n_query,
        query_finite_frac=metrics["query_finite_frac"],
        query_row_std_mean=metrics["query_row_std_mean"],
        in_sample_pcc=ins["pearson_pooled"],
        output=out,
    )
    log(f"DONE -> {out}  total {metrics['total_sec']/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        with open(LOG_PATH, "a") as f:
            f.write(tb + "\n")
        print(tb, flush=True)
        set_status(status="FAILED", stage="failed", error=tb[-2000:])
        sys.exit(1)

#!/usr/bin/env python
"""Thin wrapper: train NTF on the 5 observed 50um MERFISH slices and export a
uniform 50x50x50 um 3D gene-expression h5ad.

Run on a GPU node (tinycudann). Does not modify the shared NTF repo; all
overrides (bounding box, PSF resolution) are monkeypatches local to this file.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback

NTF_ROOT = os.path.join(os.environ["MERFISH5_UPSTREAM"], "NTF")
sys.path.insert(0, NTF_ROOT)

BASE = os.environ.get("MERFISH5_NTF_BASE", os.environ.get("MERFISH5_ROOT", os.getcwd()))
GRID = os.environ.get("MERFISH5_GRID", os.path.join(BASE, "grid_50um.npz"))
OBS = os.environ.get("MERFISH5_OBSERVED", os.path.join(BASE, "observed_50um.h5ad"))
OUT = os.environ.get("MERFISH5_NTF_OUT", os.path.join(BASE, "NTF"))
STATUS = os.path.join(OUT, "status.json")
METRICS = os.path.join(OUT, "metrics.json")

T0 = time.time()


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_status(**kw):
    """Merge-update status.json atomically."""
    cur = {}
    if os.path.exists(STATUS):
        try:
            with open(STATUS) as f:
                cur = json.load(f)
        except Exception:
            cur = {}
    cur.update(kw)
    cur["updated_at"] = now_iso()
    cur["elapsed_sec"] = round(time.time() - T0, 1)
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cur, f, indent=2, ensure_ascii=False)
    os.replace(tmp, STATUS)


class GpuMonitor(threading.Thread):
    """Sample device memory with nvidia-smi (whole GPU, incl. co-tenants)."""

    def __init__(self, interval=5.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_evt = threading.Event()
        self.peak_mb = 0

    def run(self):
        while not self.stop_evt.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"], text=True, timeout=10)
                val = max(int(x.strip()) for x in out.strip().splitlines() if x.strip())
                self.peak_mb = max(self.peak_mb, val)
            except Exception:
                pass
            self.stop_evt.wait(self.interval)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=15)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--n_iter", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--learning_rate", type=float, default=1e-4,
                   help="unused by the param-group loop; kept for the NTF arg namespace")
    p.add_argument("--enc_lr", type=float, default=1e-2)
    p.add_argument("--net_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--milestones", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--n_levels", type=int, default=13)
    p.add_argument("--log2_hashmap_size", type=int, default=20)
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--check_interval", type=int, default=200)
    p.add_argument("--weight_expr", type=float, default=1.0)
    p.add_argument("--sample_chunk", type=int, default=65536)
    p.add_argument("--pilot_points", type=int, default=2048)
    p.add_argument("--in_sample_eval", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def pearson_cols(a, b):
    """Per-column Pearson correlation, NaN when a column is constant."""
    import numpy as np
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    sa = np.sqrt((a * a).sum(0))
    sb = np.sqrt((b * b).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b).sum(0) / (sa * sb)
    r[(sa == 0) | (sb == 0)] = np.nan
    return r


def main():
    args_cli = parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.chdir(OUT)

    import numpy as np
    import torch
    import anndata as ad
    import pandas as pd

    import NTF.data as ndmod
    import NTF.models as ntfm
    from NTF.config import add_shared_args, process_args
    from NTF.data import SpatialOmicsDataset
    from NTF.sample import sample_points

    # drop stale state from a previous invocation
    if os.path.exists(STATUS):
        os.remove(STATUS)
    write_status(status="RUNNING", phase="setup", host=os.uname().nodename,
                 pid=os.getpid(), pilot=args_cli.pilot,
                 command=" ".join([sys.executable] + sys.argv))

    # ---------------- data ----------------
    t_load = time.time()
    grid = np.load(GRID)
    xc, yc, zc = grid["x_centers"], grid["y_centers"], grid["z_centers"]
    inside, obs_mask, query_mask = grid["inside"], grid["obs_mask"], grid["query_mask"]
    bs = float(grid["bs"])

    adata = ad.read_h5ad(OBS)
    adata.X = np.ascontiguousarray(adata.X, dtype=np.float32)
    adata.obsm["spatial"] = np.ascontiguousarray(adata.obsm["X_aligned"], dtype=np.float32)
    assert adata.obsm["spatial"].shape == (adata.n_obs, 3)
    assert "slice" in adata.obs

    load_sec = time.time() - t_load

    # ---------------- geometry: explicit tight bounding box ----------------
    # NTF normalises per axis via (x-bb0)/(bb1-bb0), so an explicit bbox that
    # matches the 50um grid keeps z (span 0.837 mm) at full relative resolution
    # instead of being padded by the data-derived margin (resolution.max()*2).
    xyz = adata.obsm["spatial"]
    margin = 0.5 * bs
    bb = np.stack([xyz.min(0) - margin, xyz.max(0) + margin]).astype(np.float32)
    bb[:, 0] = [xc.min() - bs, xc.max() + bs]
    bb[:, 1] = [yc.min() - bs, yc.max() + bs]
    bb[:, 2] = [zc.min() - bs, zc.max() + bs]
    BB = torch.tensor(bb, dtype=torch.float32)

    ndmod.SpatialOmicsDataset.bounding_box = property(lambda self: BB)
    # isotropic 50um PSF instead of the per-axis min-gap estimate (z gap is
    # 0.19-0.25 mm between slices, which would smear z by ~6 voxels)
    ndmod.SpatialOmicsDataset._estimate_resolution = staticmethod(
        lambda xyz, n_samples=2000: torch.tensor([bs, bs, bs], dtype=torch.float32)
    )

    parser = add_shared_args(description="NTF 50um MERFISH")
    ntf_args = parser.parse_args([
        "--input_data", OBS,
        "--output_dir", OUT,
        "--slice_id", "slice",
        "--log_dir", os.path.join(OUT, "runs"),
        "--device", "cuda:0",
        "--n_iter", str(args_cli.n_iter),
        "--batch_size", str(args_cli.batch_size),
        "--learning_rate", str(args_cli.learning_rate),
        "--n_levels", str(args_cli.n_levels),
        "--log2_hashmap_size", str(args_cli.log2_hashmap_size),
        "--n_samples", str(args_cli.n_samples),
        "--weight_expr", str(args_cli.weight_expr),
        "--early_stopping_check_interval", str(args_cli.check_interval),
        "--single_precision",
        "--no_dropout",
    ])
    ntf_args = process_args(ntf_args)

    ntf_args.early_stopping_check_interval = args_cli.check_interval
    if args_cli.pilot:
        ntf_args.early_stopping_patience = 10 ** 9

    cfg = {
        "n_iter": args_cli.n_iter,
        "batch_size": args_cli.batch_size,
        "learning_rate": args_cli.learning_rate,
        "enc_lr": args_cli.enc_lr,
        "net_lr": args_cli.net_lr,
        "weight_decay": args_cli.weight_decay,
        "milestones": args_cli.milestones,
        "gamma": args_cli.gamma,
        "optimizer_note": "param-group Adam (hash grid 1e-2, rest 1e-3); NTF default single lr=1e-4 does not move the hash grid",
        "n_levels": args_cli.n_levels,
        "log2_hashmap_size": args_cli.log2_hashmap_size,
        "level_scale": ntf_args.level_scale,
        "base_resolution": ntf_args.base_resolution,
        "n_features_per_level": ntf_args.n_features_per_level,
        "width": ntf_args.width,
        "depth": ntf_args.depth,
        "n_samples": args_cli.n_samples,
        "weight_expr": args_cli.weight_expr,
        "single_precision": True,
        "no_dropout": True,
        "bounding_box_mm": bb.tolist(),
        "psf_resolution_mm": [bs, bs, bs],
        "bbox_note": "explicit tight grid bbox; per-axis normalisation -> z keeps full [0,1] range (not squashed)",
    }
    print("[cfg]", json.dumps(cfg, indent=2), flush=True)
    write_status(phase="data_loaded", config=cfg, load_sec=round(load_sec, 1),
                 n_obs=int(adata.n_obs), n_genes=int(adata.n_vars),
                 n_slices=int(adata.obs["slice"].nunique()))

    mon = GpuMonitor()
    mon.start()

    # ---------------- train ----------------
    # NTF's own train() puts every parameter (hash grid + MLP) in one Adam with
    # lr=1e-4. With tcnn's hash grid initialised at U(-1e-4,1e-4) that leaves the
    # output pinned at softplus(0)=0.693 for >10k iterations (verified: loss and
    # predictions flat for lr 1e-4/1e-3/5e-3). We therefore re-use the library
    # model/loss but drive the loop with two param groups (standard instant-ngp
    # practice): hash grid lr 1e-2, MLP/heads lr 1e-3, no weight decay.
    # Evidence: in-sample r_all 0.878 @2k, 0.903 @6k vs 0.000 with lr=1e-4.
    write_status(phase="training")
    t_train = time.time()

    dataset = SpatialOmicsDataset(adata, "slice")
    model = ntfm.NeuralTranscriptomicField(
        n_genes=dataset.n_genes, n_slices=dataset.n_slices,
        resolution=dataset.resolution, bounding_box=dataset.bounding_box,
        args=ntf_args, pos_weight=dataset.nonzero_to_zero_ratio,
    )
    enc_ids = {id(p) for p in model.inr.encoding.parameters()}
    enc_pars = [p for p in model.parameters() if id(p) in enc_ids]
    net_pars = [p for p in model.parameters() if id(p) not in enc_ids]
    optimizer = torch.optim.Adam(
        [{"params": enc_pars, "lr": args_cli.enc_lr},
         {"params": net_pars, "lr": args_cli.net_lr}],
        betas=(0.9, 0.99), eps=1e-15, weight_decay=args_cli.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(m * args_cli.n_iter) for m in args_cli.milestones],
        gamma=args_cli.gamma,
    )

    model.train()
    avg = {"loss": 0.0, "mse": 0.0, "logvar": 0.0, "exprreg": 0.0}
    alpha = 0.999
    n_seen = 0
    for i in range(1, args_cli.n_iter + 1):
        batch = dataset.get_batch(args_cli.batch_size, "cuda:0")
        losses = model(**batch)
        loss = losses[ntfm.DS_LOSS] + args_cli.weight_expr * losses.get(ntfm.E_REG, 0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        cur = {"loss": float(loss), "mse": float(losses[ntfm.D_LOSS]),
               "logvar": float(losses[ntfm.S_LOSS]),
               "exprreg": float(losses.get(ntfm.E_REG, 0))}
        if n_seen == 0:
            avg = {k: v for k, v in cur.items()}
        else:
            for k, v in cur.items():
                avg[k] = avg[k] * alpha + v * (1 - alpha)
        n_seen += 1
        if i % 500 == 0 or i == args_cli.n_iter:
            print(f"[iter {i}/{args_cli.n_iter}] " + json.dumps(
                {k: round(v, 6) for k, v in avg.items()}) +
                f" lr_enc={optimizer.param_groups[0]['lr']:.2e}"
                f" lr_net={optimizer.param_groups[1]['lr']:.2e}", flush=True)
            write_status(phase="training", iters_done=i,
                         train_sec_so_far=round(time.time() - t_train, 1),
                         loss={k: round(v, 5) for k, v in avg.items()})

    train_sec = time.time() - t_train
    torch.cuda.synchronize()
    gpu_alloc_peak = torch.cuda.max_memory_allocated() / 2**20
    gpu_resv_peak = torch.cuda.max_memory_reserved() / 2**20
    model.eval()
    print(f"[train] {train_sec:.1f}s  alloc_peak={gpu_alloc_peak:.0f}MB resv_peak={gpu_resv_peak:.0f}MB",
          flush=True)
    write_status(phase="trained", train_sec=round(train_sec, 1),
                 gpu_alloc_peak_mb=round(gpu_alloc_peak, 1),
                 gpu_reserved_peak_mb=round(gpu_resv_peak, 1),
                 gpu_peak_mb=int(max(mon.peak_mb, gpu_resv_peak)),
                 final_loss={k: round(v, 6) for k, v in avg.items()})

    if args_cli.pilot:
        # small inference to validate the sampling path
        t_s = time.time()
        zz, yy, xx = np.nonzero(inside)
        idx = np.linspace(0, len(zz) - 1, min(args_cli.pilot_points, len(zz))).astype(np.int64)
        coords = np.stack([xc[xx[idx]], yc[yy[idx]], zc[zz[idx]]], 1).astype(np.float32)
        with torch.no_grad():
            res = sample_points(model.inr, torch.from_numpy(coords).to("cuda:0"))
        pred = res["expression"].float().cpu().numpy()
        pilot_sec = time.time() - t_s
        mon.stop()
        print(f"[pilot] {pred.shape} finite={np.isfinite(pred).mean():.4f} "
              f"min={pred.min():.4f} max={pred.max():.4f} {pilot_sec:.1f}s", flush=True)
        write_status(status="RUNNING", phase="pilot_done",
                     pilot_infer_sec=round(pilot_sec, 2),
                     pilot_finite=float(np.isfinite(pred).mean()),
                     gpu_peak_mb=int(max(mon.peak_mb, gpu_resv_peak)),
                     gpu_alloc_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                     train_sec=round(train_sec, 1))
        return

    # ---------------- sample full query volume ----------------
    write_status(phase="sampling")
    # Unified grid: exactly the `inside` voxels (523,163), matching the other
    # lanes (isoST ships the same 523,163). 183 observed voxels fall outside
    # `inside` (SPEC inconsistency: query∪obs = 523,346); they are counted and
    # reported, not added as extra rows.
    uni = inside
    iz, iy, ix = np.nonzero(uni)
    n = len(iz)
    h, w = uni.shape[1], uni.shape[2]
    coords = np.stack([xc[ix], yc[iy], zc[iz]], 1).astype(np.float32)
    # For a held-out run the input observed matrix contains only the visible
    # sections; the grid's obs_mask still covers all five.  Use the input
    # observed voxels as the observed set so held-out target voxels are sampled.
    is_obs = np.zeros(n, dtype=bool)

    X_out = np.empty((n, adata.n_vars), dtype=np.float32)
    # fill observed rows first (exact measured values)
    oz, oy, ox = adata.obs["iz"].to_numpy(), adata.obs["iy"].to_numpy(), adata.obs["ix"].to_numpy()
    oflat = oz.astype(np.int64) * h * w + oy.astype(np.int64) * w + ox
    uflat = iz.astype(np.int64) * h * w + iy.astype(np.int64) * w + ix
    pos = np.searchsorted(uflat, oflat)  # uflat is sorted (np.nonzero order)
    pos_c = np.clip(pos, 0, n - 1)
    keep = uflat[pos_c] == oflat
    n_obs_outside = int((~keep).sum())
    is_obs[pos[keep]] = True
    assert keep.sum() + n_obs_outside == adata.n_obs, "observed voxel mapping broken"
    X_out[pos[keep]] = adata.X[keep]
    src = np.full(n, -1, dtype=np.int64)
    src[pos[keep]] = np.nonzero(keep)[0]

    qsel = np.nonzero(~is_obs)[0]
    t_samp = time.time()
    model.inr.eval()
    for s in range(0, len(qsel), args_cli.sample_chunk):
        sl = qsel[s:s + args_cli.sample_chunk]
        xb = torch.from_numpy(coords[sl]).to("cuda:0")
        with torch.no_grad():
            res = sample_points(model.inr, xb)
        X_out[sl] = res["expression"].float().cpu().numpy()
    torch.cuda.synchronize()
    sample_sec = time.time() - t_samp
    print(f"[sample] {len(qsel)} query points in {sample_sec:.1f}s", flush=True)

    finite_frac = float(np.isfinite(X_out).mean())
    assert finite_frac == 1.0, f"non-finite predictions: {finite_frac}"
    assert np.isfinite(X_out[qsel]).all()
    print(f"[sample] X {X_out.shape} finite={finite_frac} min={X_out.min():.4f} "
          f"max={X_out.max():.4f} query_mean={X_out[qsel].mean():.4f}", flush=True)
    write_status(phase="sampled", sample_sec=round(sample_sec, 1),
                 n_out=int(n), n_query=int(len(qsel)), n_observed=int(is_obs.sum()),
                 query_finite=float(np.isfinite(X_out[qsel]).mean()),
                 gpu_peak_mb=int(max(mon.peak_mb, torch.cuda.max_memory_reserved() / 2**20)))

    # ---------------- in-sample reconstruction check (sanity, not the target) ----------------
    in_sample = {}
    if args_cli.in_sample_eval:
        t_e = time.time()
        oc = adata.obsm["spatial"]
        pred_obs = np.empty((len(oc), adata.n_vars), dtype=np.float32)
        for s in range(0, len(oc), args_cli.sample_chunk):
            with torch.no_grad():
                res = sample_points(model.inr, torch.from_numpy(oc[s:s + args_cli.sample_chunk]).to("cuda:0"))
            pred_obs[s:s + args_cli.sample_chunk] = res["expression"].float().cpu().numpy()
        G = adata.X
        nz = G > 0
        r_all = pearson_cols(pred_obs, G)
        r_nz = np.full(adata.n_vars, np.nan)
        for g in range(adata.n_vars):
            m = nz[:, g]
            if m.sum() >= 5 and pred_obs[m, g].std() > 0:
                r_nz[g] = np.corrcoef(pred_obs[m, g], G[m, g])[0, 1]
        in_sample = {
            "pearson_all_mean": float(np.nanmean(r_all)),
            "pearson_nonzero_mean": float(np.nanmean(r_nz)),
            "pearson_nonzero_median": float(np.nanmedian(r_nz)),
            "mse_all": float(((pred_obs - G) ** 2).mean()),
            "rmse_all": float(np.sqrt(((pred_obs - G) ** 2).mean())),
            "pred_mean_at_obs": float(pred_obs.mean()),
            "gt_mean": float(G.mean()),
            "n_genes_with_corr": int(np.isfinite(r_nz).sum()),
            "eval_sec": round(time.time() - t_e, 1),
        }
        print("[in_sample]", json.dumps(in_sample), flush=True)
        write_status(phase="sampled", in_sample=in_sample)

    # ---------------- write h5ad ----------------
    write_status(phase="writing")
    # slice label: observed voxels keep their measured slice; query voxels get
    # the label of the nearest observed z layer.
    obs_z = np.array(sorted(set(oz.tolist())), dtype=np.int64)
    z_of_slice = {}
    for z in obs_z:
        lab = adata.obs.loc[adata.obs["iz"] == z, "slice"]
        z_of_slice[z] = str(lab.iloc[0])
    zc_obs = zc[obs_z]
    nearest = obs_z[np.abs(zc[iz][:, None] - zc_obs[None, :]).argmin(1)]
    slice_lab = np.array([z_of_slice[z] for z in nearest], dtype=object)
    slice_lab[pos[keep]] = adata.obs["slice"].astype(str).to_numpy()[keep]

    out = ad.AnnData(
        X=X_out,
        obs=pd.DataFrame({
            "x_mm": coords[:, 0],
            "y_mm": coords[:, 1],
            "z_mm": coords[:, 2],
            "ix": ix.astype(np.int32),
            "iy": iy.astype(np.int32),
            "iz": iz.astype(np.int32),
            "is_observed": is_obs,
            "slice": pd.Categorical(slice_lab, categories=list(adata.obs["slice"].cat.categories)),
        }, index=[f"v{ix[i]}_{iy[i]}_{iz[i]}" for i in range(n)]),
        var=adata.var[["gene_name"]].copy(),
    )
    out.var_names = adata.var_names.copy()
    out.obsm["spatial"] = coords
    out.uns["ntf"] = {
        "voxel_um": int(round(bs * 1000)),
        "n_query": int(len(qsel)),
        "n_observed": int(is_obs.sum()),
        "source_observed": OBS,
        "model": "NTF GeneINR hash-grid INR, 5 slices",
        **{k: v for k, v in cfg.items() if k != "bounding_box_mm"},
        "bounding_box_mm": bb.tolist(),
    }
    t_w = time.time()
    out.write_h5ad(os.path.join(OUT, "NTF_50um_uniform.h5ad"), compression=None)
    write_sec = time.time() - t_w

    mon.stop()
    gpu_peak = int(max(mon.peak_mb, torch.cuda.max_memory_reserved() / 2**20))
    total = time.time() - T0
    metrics = {
        "method": "NTF",
        "dataset": "merfish_3d_recon_50um",
        "voxel_um": int(round(bs * 1000)),
        "train_sec": round(train_sec, 1),
        "sample_sec": round(sample_sec, 1),
        "write_sec": round(write_sec, 1),
        "load_sec": round(load_sec, 1),
        "total_sec": round(total, 1),
        "gpu_peak_mb": gpu_peak,
        "gpu_alloc_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "gpu_reserved_peak_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "gpu_peak_note": "max device-wide nvidia-smi used incl. co-tenant xpy job; torch peaks are this process",
        "n_iter": args_cli.n_iter,
        "n_query": int(len(qsel)),
        "n_observed": int(is_obs.sum()),
        "n_out": int(n),
        "n_genes": int(adata.n_vars),
        "query_finite_frac": float(np.isfinite(X_out[qsel]).mean()),
        "all_finite_frac": finite_frac,
        "query_pred_min": float(X_out[qsel].min()),
        "query_pred_max": float(X_out[qsel].max()),
        "query_pred_mean": float(X_out[qsel].mean()),
        "obs_match_input": bool(np.array_equal(X_out[pos[keep]], adata.X[keep])),
        "n_observed_input": int(len(oflat)),
        "n_observed_outside_inside": n_obs_outside,
        "note_grid": "rows are the grid `inside` mask (523,163); 183 observed voxels lie outside it and are dropped (reported)",
        "in_sample": in_sample,
        "config": cfg,
        "host": os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "finished_at": now_iso(),
    }
    with open(METRICS, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("[metrics]", json.dumps(metrics, ensure_ascii=False), flush=True)
    write_status(status="DONE", phase="done", gpu_peak_mb=gpu_peak,
                 metrics_path=METRICS,
                 output=os.path.join(OUT, "NTF_50um_uniform.h5ad"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        try:
            write_status(status="FAILED", phase="error", error=tb[-3000:])
        except Exception:
            pass
        sys.exit(1)

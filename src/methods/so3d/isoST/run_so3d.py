#!/usr/bin/env python3
"""isoST on the So3D 46-dataset held-out-middle-slice benchmark.

Per dataset: train the official isoST dual ODE on the VISIBLE slices only, then
integrate f from the lower visible slice and g from the upper visible slice to
the held-out slice depth and map the generated points to the target spots by 2D
NN in normalised xy.  Only target rows are written; no volume is materialised.

Usage (inside the granted srun step):
  python run_so3d.py --pilot --datasets Brain__Mouse__3Ddataset_1
  python run_so3d.py --all --skip-existing
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _so3d_common as C  # noqa: E402

ISOST_DIR = "/storage/yuhongtaoLab/linpeiran/xpy/lab/baseline/methods/isoST"
sys.path.insert(0, ISOST_DIR)

CUDA_HOME = "/soft/devtools/cuda/cuda-12.9"
os.environ.setdefault("CUDA_HOME", CUDA_HOME)
os.environ.setdefault("CUDA_PATH", CUDA_HOME)
os.environ["PATH"] = f"{CUDA_HOME}/bin:" + os.environ.get("PATH", "")
os.environ.setdefault("LIBRARY_PATH", f"{CUDA_HOME}/lib64:")
os.environ.setdefault("LD_LIBRARY_PATH", f"{CUDA_HOME}/lib64:")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

METHOD = "isoST"
OUT = HERE
LANE = {"job": "9383993", "node": "gvna03"}

# isoST reference inter-slice gap in model units; one Euler step = gap/3.
REFERENCE_GAP = 0.928329
STEPS_PER_GAP = 3.0


def make_params(gene_dim, delta_d):
    return dict(
        gene_dim=int(gene_dim), hidden_dim=64, head_num=1, K=8,
        method="euler", delta_d=delta_d, stride=1, lr=0.001,
        optimizer_name="NAdam", weight_decay=1e-8, warm_up_rate=1.0,
        std_x=0.01, std_y=0.01, std_z=0.1, std_seq=0.1, alpha=0.1, dual=True,
        beta_start_value=1.0, beta_end_value=0.05, beta_start_iteration=50,
        beta_n_iterations=50,
    )


class Fitter:
    """Visible-only geometry + feature normalisation."""

    def __init__(self, vis_c, scale_z):
        self.xy_all = vis_c[:, :2]
        self.xmin = float(self.xy_all[:, 0].min())
        self.ymin = float(self.xy_all[:, 1].min())
        self.W = float(max(self.xy_all[:, 0].max() - self.xmin,
                           self.xy_all[:, 1].max() - self.ymin))
        self.zmin = float(vis_c[:, 2].min())
        self.scale_z = scale_z

    def xy(self, x, y):
        return np.stack([(x - self.xmin) / self.W, (y - self.ymin) / self.W],
                        axis=1).astype(np.float32)

    def zmodel(self, z):
        return (np.asarray(z, dtype=np.float64) - self.zmin) * self.scale_z


def build_training(fit, vis, vis_c, order):
    Xv = vis["X"]
    fmin = Xv.min(axis=0)
    frange = (Xv.max(axis=0) - fmin).astype(np.float32)
    const = frange <= 1e-12
    Fn = (Xv - fmin) / np.where(const, 1.0, frange)
    Fn[:, const] = 0.0
    Fn = np.clip(Fn, 0.0, 1.0).astype(np.float32)

    tensors, zs = [], []
    for s in order:
        rows = np.flatnonzero(vis["slice"] == s)
        zc = float(np.median(vis["z"][rows]))
        zmod = float(fit.zmodel(zc))
        t = np.column_stack([fit.xy(vis_c[rows, 0], vis_c[rows, 1]),
                             np.full(rows.size, zmod, np.float32), Fn[rows]]).astype(np.float32)
        tensors.append(t)
        zs.append(zmod)
    return dict(tensors=tensors, z_model=zs, fmin=fmin, frange=frange, const=const,
                n_visible_spots=int(Xv.shape[0]))


def inv_features(fn, td):
    fn = np.clip(np.asarray(fn, dtype=np.float32), 0.0, 1.0)
    out = fn * td["frange"] + td["fmin"]
    out[:, td["const"]] = 0.0
    return np.clip(out, 0.0, None).astype(np.float32)


def run_dataset(ds, args, epochs, vis=None, tgt=None):
    import torch
    from model.base_model import IsoST as TrainerClass
    from utils.train_ode import train
    from torchdiffeq import odeint
    from scipy.spatial import cKDTree

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if vis is None:
        vis = C.load_visible(ds)
    if tgt is None:
        tgt = C.load_target(ds)
    vis_c, tgt_c, diag = C.normalize_geometry(vis, tgt)

    order, zs_native = C.slice_order(vis)
    z_target = float(np.median(tgt["z"]))
    lo, hi, _, _ = C.bracket_slices(vis, z_target)
    if lo is None or hi is None:
        raise Unsupported(f"{ds}: no visible bracket around target z={z_target}")

    gaps = np.diff(np.array([zs_native[s] for s in order], dtype=np.float64))
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    ok_gaps = [g for g in gaps if g > 0]
    if not ok_gaps:
        raise Unsupported(f"{ds}: cannot estimate inter-slice z spacing")
    z_gap_model = REFERENCE_GAP  # geometry is normalised so the median gap is 100
    scale_z = z_gap_model / 100.0
    delta_d = z_gap_model / STEPS_PER_GAP

    fit = Fitter(vis_c, scale_z)
    td = build_training(fit, vis, vis_c, order)
    slices_t = [torch.from_numpy(t).cuda() for t in td["tensors"]]

    exp_dir = os.path.join(OUT, "experiments", ds)
    os.makedirs(exp_dir, exist_ok=True)
    trainer = TrainerClass(device="cuda:0", **make_params(vis["n_genes"], delta_d)).cuda()
    t0 = time.time()
    train(trainer, exp_dir, [slices_t], batch_num=1, epochs_list=epochs,
          checkpoint_every=10 ** 9, backup_every=50)
    train_sec = time.time() - t0
    trainer.eval()

    z_list = td["z_model"]
    li, hj = order.index(lo), order.index(hi)
    z_lo, z_hi = z_list[li], z_list[hj]
    z_t = float(fit.zmodel(z_target))
    tinf = time.time()
    with torch.no_grad():
        depths = torch.tensor([z_lo, z_t, z_hi], dtype=torch.float32, device="cuda:0")
        traj_f = odeint(trainer.f_joint, slices_t[li], depths, method="euler")
        traj_g = odeint(trainer.g_joint, slices_t[hj], depths.flip(0), method="euler")
        f_t = traj_f[1].detach().cpu().numpy()
        g_t = traj_g[1].detach().cpu().numpy()
        del traj_f, traj_g
    gen = np.vstack([f_t, g_t])
    del f_t, g_t
    tree = cKDTree(gen[:, :2])
    qxy = fit.xy(tgt_c[:, 0], tgt_c[:, 1])
    dist, nn = tree.query(qxy, k=1)
    pred = inv_features(gen[nn, 3:], td)
    infer_sec = time.time() - tinf

    stats = dict(
        bracket=[int(lo), int(hi)],
        n_visible_slices=len(order), n_visible_spots=td["n_visible_spots"],
        n_generated=int(gen.shape[0]), n_target=tgt["n"], n_genes=vis["n_genes"],
        z_target_native=z_target,
        z_lo_model=float(z_lo), z_t_model=z_t, z_hi_model=float(z_hi),
        nn_um_median=float(np.median(dist) * 100.0),
        nn_um_p90=float(np.percentile(dist, 90) * 100.0),
        nn_um_max=float(dist.max() * 100.0),
        geometry=diag,
        train_sec=round(train_sec, 1), infer_sec=round(infer_sec, 1),
        n_params=int(sum(p.numel() for p in trainer.parameters())),
        pred_min=float(pred.min()), pred_max=float(pred.max()),
        pred_mean=float(pred.mean()), pred_zero_frac=float((pred == 0).mean()),
        per_gene_std_mean=float(pred.std(axis=0).mean()),
        const_genes=int((pred.std(axis=0) == 0).sum()),
    )
    return pred, vis["genes"], tgt, stats


class Unsupported(Exception):
    pass


def write_ds_status(ds, status, reason, extra=None):
    """Per-dataset status contract: predictions/<ds>/<method>.status.json."""
    d = {"method": METHOD, "dataset": ds, "status": status, "reason": reason,
         "updated_at": C.now()}
    if extra:
        d.update(extra)
    os.makedirs(os.path.join(C.PRED_ROOT, ds), exist_ok=True)
    C.write_json(os.path.join(C.PRED_ROOT, ds, f"{METHOD}.status.json"), d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--datasets", default=None, help="comma separated dataset names")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--epochs", default="40,40,40")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    ap.add_argument("--max-datasets", type=int, default=0)
    args = ap.parse_args()

    gpu_idx = str(args.gpu).split(",")[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    epochs = [int(x) for x in args.epochs.split(",")]
    if args.pilot:
        epochs = [int(min(e, 2)) for e in epochs]

    st = C.Status(METHOD, OUT, extra=dict(
        lane=LANE, pilot=bool(args.pilot), epochs_list=epochs,
        reference_gap=REFERENCE_GAP, steps_per_gap=STEPS_PER_GAP,
        feature_norm="per-gene min-max fit on visible spots only (no log1p/PCA)",
        mapping="2D NN in normalised xy over f(lower)+g(upper) virtual points",
        target_only=True))
    C.wait_ready(log=st.log)
    import _runner_c_agg as AGG

    names = C.dataset_dirs()
    if args.datasets:
        keep = set(args.datasets.split(","))
        names = [n for n in names if n in keep]
    if args.pilot:
        names = names[:1]
    if args.max_datasets:
        names = names[:args.max_datasets]

    st.write(stage="ready", datasets_total=len(names))
    st.log(f"START pilot={args.pilot} datasets={len(names)} epochs={epochs} gpu={gpu_idx}")

    mon = C.GpuSampler(); mon.start()
    res = C.ResourceMonitor(); res.start()

    failures, unsupported, metrics = [], [], []
    t_all = time.time()
    for i, ds in enumerate(names):
        path = C.pred_path(METHOD, ds)
        if args.skip_existing and os.path.exists(path):
            try:
                genes = np.asarray(C.load_meta(ds)["genes"]).astype(str)
                if C.existing_ok(METHOD, ds, genes, C.load_target(ds)):
                    st.log(f"[{ds}] SKIP existing valid prediction")
                    st.record(ds, {"status": "DONE", "skipped": True,
                                   "prediction": path})
                    write_ds_status(ds, "DONE", "existing verified prediction kept")
                    continue
            except Exception:
                pass
        st.write(stage=f"run:{ds}", current=ds, dataset_index=i)
        t0 = time.time()
        try:
            pred, genes, tgt, stats = run_dataset(ds, args, epochs)
            C.write_prediction(METHOD, ds, pred, tgt, genes,
                               extra=dict(method=METHOD, dataset=ds))
            chk = C.verify_prediction(METHOD, ds, genes, tgt)
            if not chk["pass"]:
                raise RuntimeError(f"prediction verification failed: {chk}")
            rec = dict(status="DONE", total_sec=round(time.time() - t0, 1),
                       bytes=chk["bytes"], prediction=path, **stats)
            metrics.append(dict(dataset=ds, **rec))
            st.log(f"[{ds}] DONE target={tgt['n']} genes={genes.size} "
                   f"train={stats['train_sec']}s infer={stats['infer_sec']}s "
                   f"nn100_med={stats['nn_um_median']:.1f} mean={chk['pred_mean']:.4f}")
            write_ds_status(ds, "DONE", "prediction written and verified target-only",
                            extra={"bytes": chk["bytes"],
                                   "n_target": int(tgt["n"]),
                                   "n_genes": int(genes.size),
                                   "train_sec": stats["train_sec"],
                                   "infer_sec": stats["infer_sec"]})
        except Unsupported as e:
            rec = dict(status="UNSUPPORTED", evidence=str(e),
                       total_sec=round(time.time() - t0, 1))
            unsupported.append(dict(dataset=ds, evidence=str(e)))
            st.log(f"[{ds}] UNSUPPORTED {e}")
            write_ds_status(ds, "UNSUPPORTED", str(e))
        except Exception as e:
            tb = traceback.format_exc()
            rec = dict(status="FAILED", error=repr(e), traceback=tb[-3000:],
                       total_sec=round(time.time() - t0, 1))
            failures.append(dict(dataset=ds, error=repr(e), traceback=tb[-3000:]))
            st.log(f"[{ds}] FAILED {e!r}\n{tb}")
            write_ds_status(ds, "FAILED", repr(e), extra={"traceback": tb[-1500:]})
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        st.record(ds, rec)
        st.write(gpu_peak_mb=int(mon.peak), failures=len(failures),
                 unsupported=len(unsupported))
        AGG.update()

    mon.stop_evt.set(); mon.join(timeout=5)
    res.stop_evt.set(); res.join(timeout=5)
    total_sec = round(time.time() - t_all, 1)
    cm = dict(
        method=METHOD, dataset_scope="so3d_46_heldout_middle_slice",
        config=vars(args), epochs_list=epochs, reference_gap=REFERENCE_GAP,
        delta_d=REFERENCE_GAP / STEPS_PER_GAP,
        feature_norm="per-gene min-max fit on visible spots only",
        mapping="2D NN in normalised xy over f(lower)+g(upper) virtual points",
        n_done=sum(1 for r in metrics if r["status"] == "DONE"),
        n_failed=len(failures), n_unsupported=len(unsupported),
        total_sec=total_sec,
        mean_sec_per_dataset=round(total_sec / max(1, len(names)), 1),
        gpu_peak_mb=int(mon.peak), gpu_peak_util=int(mon.peak_util),
        failures=failures, unsupported=unsupported, datasets=metrics)
    cm.update(res.summary())
    try:
        import torch
        cm["torch_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except Exception:
        cm["torch_peak_mb"] = None
    C.write_json(os.path.join(OUT, "compute_metrics.json"), cm)
    ok = not failures and not unsupported
    st.finish("DONE" if ok else "PARTIAL",
              n_done=cm["n_done"], n_failed=len(failures),
              n_unsupported=len(unsupported), gpu_peak_mb=int(mon.peak),
              compute_metrics=os.path.join(OUT, "compute_metrics.json"))
    st.log(f"FINISHED done={cm['n_done']} failed={len(failures)} "
           f"unsupported={len(unsupported)} total={total_sec}s gpu_peak={mon.peak}MiB")
    AGG.update()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

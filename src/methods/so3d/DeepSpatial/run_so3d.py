#!/usr/bin/env python3
"""DeepSpatial on the So3D 46-dataset held-out-middle-slice benchmark.

Per dataset: fit the official DeepSpatial UOT + flow-matching model on the
VISIBLE slices only, then run the official reconstruct_between_slices between
the two visible slices bracketing the held-out slice and map the generated
points to the target spots (3D NN).  The generated segment stays in memory; the
output file holds target rows exclusively.

Usage (inside the granted srun step):
  python run_so3d.py --pilot --datasets Brain__Mouse__3Ddataset_1
  python run_so3d.py --all --skip-existing
"""
import argparse
import os
import sys
import time
import traceback

import numpy as np
import anndata as ad
import pandas as pd
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _so3d_common as C  # noqa: E402

METHOD_DIR = "/storage/yuhongtaoLab/linpeiran/xpy/lab/baseline/methods/DeepSpatial"
sys.path.insert(0, METHOD_DIR)
import deepspatial as ds  # noqa: E402

METHOD = "DeepSpatial"
OUT = HERE
LANE = {"job": "9383993", "node": "gvna03"}


class Unsupported(Exception):
    pass


def write_ds_status(dsname, status, reason, extra=None):
    """Per-dataset status contract: predictions/<ds>/<method>.status.json."""
    d = {"method": METHOD, "dataset": dsname, "status": status, "reason": reason,
         "updated_at": C.now()}
    if extra:
        d.update(extra)
    os.makedirs(os.path.join(C.PRED_ROOT, dsname), exist_ok=True)
    C.write_json(os.path.join(C.PRED_ROOT, dsname, f"{METHOD}.status.json"), d)


def gpu_mb():
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2**20, 1)


def make_frames(vis, order):
    frames = {}
    for s in order:
        rows = np.flatnonzero(vis["slice"] == s)
        zc = float(np.median(vis["z"][rows]))
        a = ad.AnnData(
            X=vis["X"][rows].copy(),
            obs=pd.DataFrame(
                {"cell_class": ["all"] * rows.size,
                 "z_coord": np.full(rows.size, zc, dtype=np.float64),
                 "section": np.full(rows.size, int(s), dtype=np.int64)},
                index=[f"{s}_{i}" for i in range(rows.size)]),
            var=pd.DataFrame(index=np.asarray(vis["genes"]).astype(str)))
        a.obsm["spatial_xy"] = np.stack(
            [vis["coords"][rows, 0], vis["coords"][rows, 1]], axis=1).astype(np.float64)
        frames[int(s)] = a
    return frames


def run_dataset(dsname, args, rng_seed, vis=None, tgt=None):
    if vis is None:
        vis = C.load_visible(dsname)
    if tgt is None:
        tgt = C.load_target(dsname)
    vis_c, tgt_c, diag = C.normalize_geometry(vis, tgt)

    order, zs_native = C.slice_order(vis)
    if len(order) < 2:
        raise Unsupported(f"{dsname}: only {len(order)} visible slice(s)")
    z_target = float(np.median(tgt["z"]))
    lo, hi, _, _ = C.bracket_slices(vis, z_target)
    if lo is None or hi is None:
        raise Unsupported(f"{dsname}: no visible bracket around target z={z_target}")
    gap = float(zs_native[hi] - zs_native[lo])
    if not np.isfinite(gap) or gap <= 0:
        raise Unsupported(f"{dsname}: non-positive bracket gap ({gap})")
    thickness = gap / args.density_factor

    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    torch.cuda.manual_seed_all(rng_seed)
    torch.cuda.reset_peak_memory_stats()

    frames = make_frames(vis, order)
    model = ds.DeepSpatial()

    t0 = time.time()
    model.setup_data(adata_list=[frames[s] for s in order],
                     spatial_key="spatial_xy", z_key="z_coord",
                     label_key="cell_class", batch_size=args.batch_size,
                     num_workers=0, n_samples_base=args.n_samples)
    setup_sec = time.time() - t0

    patch_size = args.patch_size
    if patch_size <= 0:
        # Keep the gene-token budget bounded: official config uses 238/8 ~= 30
        # tokens.  Wide panels (2k-4k genes) would otherwise produce 250-500
        # tokens and make the patch transformer's attention quadratic in G.
        patch_size = max(8, int(np.ceil(vis["n_genes"] / float(args.gene_tokens))))
    model.build_model(patch_size=patch_size, hidden_size=args.hidden_size,
                      depth=args.depth, num_heads=args.num_heads, lr=2e-4,
                      sampling_method="dopri5", atol=1e-5, rtol=1e-5)
    t0 = time.time()
    ckpt_dir = os.path.join(OUT, "checkpoints", dsname)
    os.makedirs(ckpt_dir, exist_ok=True)
    model.fit(max_epochs=args.epochs, save_dir=ckpt_dir,
              accelerator="gpu", devices=[0], save_ckpt=False)
    torch.cuda.synchronize()
    train_sec = time.time() - t0
    peak_train = gpu_mb()

    tinf = time.time()
    ad0, ad1 = frames[int(lo)], frames[int(hi)]
    seg = model.reconstruct_between_slices(
        adata0=ad0, adata1=ad1, thickness=thickness,
        steps=args.steps, chunk_size=args.chunk_size, device="cuda")
    torch.cuda.synchronize()
    gen_sec = time.time() - tinf

    z_gen = np.asarray(seg.obs["z_coord"], dtype=np.float64)
    xy_gen = np.asarray(seg.obsm["spatial_xy"], dtype=np.float64)[:, :2]
    Xg = seg.X.toarray() if hasattr(seg.X, "toarray") else np.asarray(seg.X)
    Xg = np.asarray(Xg, dtype=np.float32)
    n_neg = int((Xg < 0).sum())
    np.maximum(Xg, 0.0, out=Xg)
    if not np.isfinite(Xg).all():
        raise RuntimeError(f"{dsname}: non-finite DeepSpatial segment values")

    s_xy = diag["scale_xy"]
    s_z = diag["scale_z"]
    z0 = diag["z0_native"]
    pts = np.column_stack([xy_gen[:, 0] * s_xy, xy_gen[:, 1] * s_xy,
                           (z_gen - z0) * s_z])
    tree = cKDTree(pts)
    n_generated = int(tree.n)
    dist, j = tree.query(tgt_c, k=1)
    pred = Xg[j]
    del seg, Xg, pts
    torch.cuda.empty_cache()
    infer_sec = time.time() - tinf

    stats = dict(
        bracket=[int(lo), int(hi)], bracket_gap_native=gap, thickness=thickness,
        n_visible_slices=len(order), n_visible_spots=int(vis["X"].shape[0]),
        n_generated=n_generated, n_target=tgt["n"], n_genes=vis["n_genes"],
        patch_size=int(patch_size), gene_tokens=int(args.gene_tokens),
        z_target_native=z_target,
        nn_um_median=float(np.median(dist)),
        nn_um_p90=float(np.percentile(dist, 90)),
        nn_um_max=float(dist.max()),
        n_negative_clipped=n_neg, geometry=diag,
        setup_sec=round(setup_sec, 1), train_sec=round(train_sec, 1),
        recon_sec=round(gen_sec, 1), infer_sec=round(infer_sec, 1),
        torch_peak_mb=peak_train,
        pred_min=float(pred.min()), pred_max=float(pred.max()),
        pred_mean=float(pred.mean()), pred_zero_frac=float((pred == 0).mean()),
        per_gene_std_mean=float(pred.std(axis=0).mean()),
        const_genes=int((pred.std(axis=0) == 0).sum()),
    )
    return pred, vis["genes"], tgt, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-samples", type=int, default=15000)
    ap.add_argument("--patch-size", type=int, default=0,
                    help="<=0: auto = ceil(n_genes/gene_tokens), min 8")
    ap.add_argument("--gene-tokens", type=int, default=32,
                    help="target token count for auto patch size")
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--chunk-size", type=int, default=2048)
    ap.add_argument("--density-factor", type=float, default=3.0,
                    help="generated points per observed spot = gap/thickness")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    ap.add_argument("--max-datasets", type=int, default=0)
    args = ap.parse_args()

    gpu_idx = str(args.gpu).split(",")[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    if args.pilot:
        args.epochs = min(args.epochs, 1)
        args.n_samples = min(args.n_samples, 2000)
        args.batch_size = min(args.batch_size, 256)
        args.steps = min(args.steps, 8)

    st = C.Status(METHOD, OUT, extra=dict(
        lane=LANE, pilot=bool(args.pilot), config=vars(args),
        mapping="3D NN in visible-normalised geometry over the official "
                "reconstruct_between_slices segment (memory only)",
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
    st.log(f"START pilot={args.pilot} datasets={len(names)} cfg={vars(args)}")

    mon = C.GpuSampler(); mon.start()
    res = C.ResourceMonitor(); res.start()

    failures, unsupported, metrics = [], [], []
    t_all = time.time()
    for i, dsname in enumerate(names):
        path = C.pred_path(METHOD, dsname)
        if args.skip_existing and os.path.exists(path):
            try:
                genes = np.asarray(C.load_meta(dsname)["genes"]).astype(str)
                if C.existing_ok(METHOD, dsname, genes, C.load_target(dsname)):
                    st.log(f"[{dsname}] SKIP existing valid prediction")
                    st.record(dsname, {"status": "DONE", "skipped": True,
                                       "prediction": path})
                    write_ds_status(dsname, "DONE", "existing verified prediction kept")
                    continue
            except Exception:
                pass
        st.write(stage=f"run:{dsname}", current=dsname, dataset_index=i)
        t0 = time.time()
        try:
            pred, genes, tgt, stats = run_dataset(dsname, args, args.seed)
            C.write_prediction(METHOD, dsname, pred, tgt, genes,
                               extra=dict(method=METHOD, dataset=dsname))
            chk = C.verify_prediction(METHOD, dsname, genes, tgt)
            if not chk["pass"]:
                raise RuntimeError(f"prediction verification failed: {chk}")
            rec = dict(status="DONE", total_sec=round(time.time() - t0, 1),
                       bytes=chk["bytes"], prediction=path, **stats)
            metrics.append(dict(dataset=dsname, **rec))
            st.log(f"[{dsname}] DONE target={tgt['n']} genes={genes.size} "
                   f"train={stats['train_sec']}s recon={stats['recon_sec']}s "
                   f"nn_med={stats['nn_um_median']:.1f} mean={chk['pred_mean']:.4f}")
            write_ds_status(dsname, "DONE",
                            "prediction written and verified target-only",
                            extra={"bytes": chk["bytes"],
                                   "n_target": int(tgt["n"]),
                                   "n_genes": int(genes.size),
                                   "train_sec": stats["train_sec"],
                                   "recon_sec": stats["recon_sec"]})
        except Unsupported as e:
            rec = dict(status="UNSUPPORTED", evidence=str(e),
                       total_sec=round(time.time() - t0, 1))
            unsupported.append(dict(dataset=dsname, evidence=str(e)))
            st.log(f"[{dsname}] UNSUPPORTED {e}")
            write_ds_status(dsname, "UNSUPPORTED", str(e))
        except Exception as e:
            tb = traceback.format_exc()
            rec = dict(status="FAILED", error=repr(e), traceback=tb[-3000:],
                       total_sec=round(time.time() - t0, 1))
            failures.append(dict(dataset=dsname, error=repr(e), traceback=tb[-3000:]))
            st.log(f"[{dsname}] FAILED {e!r}\n{tb}")
            write_ds_status(dsname, "FAILED", repr(e), extra={"traceback": tb[-1500:]})
        torch.cuda.empty_cache()
        st.record(dsname, rec)
        st.write(gpu_peak_mb=int(mon.peak), failures=len(failures),
                 unsupported=len(unsupported))
        AGG.update()

    mon.stop_evt.set(); mon.join(timeout=5)
    res.stop_evt.set(); res.join(timeout=5)
    total_sec = round(time.time() - t_all, 1)
    cm = dict(
        method=METHOD, dataset_scope="so3d_46_heldout_middle_slice",
        config=vars(args),
        mapping="official reconstruct_between_slices per bracket; 3D NN in "
                "visible-normalised geometry; segment never written",
        n_done=sum(1 for r in metrics if r["status"] == "DONE"),
        n_failed=len(failures), n_unsupported=len(unsupported),
        total_sec=total_sec,
        mean_sec_per_dataset=round(total_sec / max(1, len(names)), 1),
        gpu_peak_mb=int(mon.peak), gpu_peak_util=int(mon.peak_util),
        torch_peak_mb=max([m.get("torch_peak_mb") or 0 for m in metrics] or [0]),
        failures=failures, unsupported=unsupported, datasets=metrics)
    cm.update(res.summary())
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

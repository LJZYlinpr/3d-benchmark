#!/usr/bin/env python
"""SINTER3D on the So3D 46-dataset held-out-middle-slice benchmark (runner B).

Adapted from Workspace/merfish_test/SINTER3D/run_sinter3d_50um.py and
Workspace/merfish_test/heart_10slice/methods/SINTER3D/run_sinter3d_heart.py:
same official `SINTER3D.model_spatial.Model`, loss, optimiser and
hyper-parameters. Documented adaptations:

  * Input = canonical `visible.h5ad` (all visible slices of one dataset).  Its X
    is already normalised by the dataset producer and is used directly as the
    count-like target (`obsm['count']`), library_size = row sums.  The decoder
    therefore outputs canonical-X units directly; no renormalisation, no expm1
    back-scale.
  * Canonical spots carry no target labels for the held-out slice, and the
    deconv basis must not come from target expression, so the basis is built
    from visible bins only: KMeans (k<=8, fit on a <=30k visible subsample) on
    unit-normalised visible profiles.  The decoder prediction path is
    coordinate-only; slice embeddings only feed the deconvolution branch.
  * Coordinates: canonical coordinates are in per-dataset source units that
    differ by orders of magnitude.  `_so3d_common.normalize_geometry` rescales
    xy so the median within-slice NN distance of *visible* spots is 100 (the
    official SINTER3D c2c_dist / `normalize`) and z so the median inter-slice
    gap is also 100, keeping the official µm-frame behaviour.
  * One held-out middle slice per dataset: training uses visible slices only;
    the target coordinates come from `target.npz` geometry (never its X).
  * Throughput: the official trainer is full-batch.  With 2k-4k gene panels and
    up to 60k visible spots a single full-batch step costs ~0.42 s (measured),
    i.e. >9 h for 46 datasets x 6000 steps.  We therefore use plain mini-batch
    SGD (batch <=16384, epoch-wise reshuffled permutation, same loss, optimiser,
    lr and step budget of 6000).  Measured 50.6 ms/step at batch 16384 with the
    largest dataset; datasets with fewer spots than the batch are full-batch.

Outputs: predictions/<dataset>/SINTER3D.npz, methods/SINTER3D/*.

Usage (inside the granted srun step on gvna03, job 9383992):
  python run_sinter3d.py --all
  python run_sinter3d.py --dataset Brain__Mouse__3Ddataset_1 --steps 500
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS_DIR = os.path.dirname(HERE)
if METHODS_DIR not in sys.path:
    sys.path.insert(0, METHODS_DIR)
import _so3d_common as C  # noqa: E402

SINTER3D_DIR = ("/storage/yuhongtaoLab/linpeiran/xpy/lab/baseline/methods/"
                "SINTER3D/SINTER3D-master")
if SINTER3D_DIR not in sys.path:
    sys.path.insert(0, SINTER3D_DIR)

METHOD = "SINTER3D"
OUTDIR = os.path.join(C.METHODS_ROOT, METHOD)

HIDDEN_DIMS = [512, 128]
SLICE_EMB_DIM = 16
MID_CHANNEL = 200
LR = 1e-3
SEED = 2024
PATIENCE = 10 ** 9
NORMALIZE = C.C2C_UM      # coordinates are already in the 100-unit frame
ALPHA_POISSON = 5
LAMBDA_FEATURE = 4
GAMMA_FEATURE = 3
NUM_FREQS = 6
COORD_EMB_DIM = 64
N_BASIS = 8
MAX_KMEANS_ROWS = 30000
MIN_VISIBLE = 100
MIN_GENES = 10


def build_tensors(vis, coords_um, log):
    import anndata as ad
    import pandas as pd
    from sklearn.cluster import KMeans

    X = vis["X"]
    order, zs = C.slice_order(vis)
    sid = {int(s): i for i, s in enumerate(order)}
    slice_ids = np.array([sid[int(s)] for s in vis["slice"]], dtype=np.int64)
    lib = X.sum(axis=1).astype(np.float32)
    log(f"[info] slices={len(order)} library_size min/mean/max = "
        f"{lib.min():.3f}/{lib.mean():.3f}/{lib.max():.3f} "
        f"(zero rows {int((lib <= 0).sum())})")
    lib_safe = np.clip(lib, 1e-6, None)

    unit = X / lib_safe[:, None]
    rng = np.random.RandomState(SEED)
    idx = np.arange(X.shape[0])
    if X.shape[0] > MAX_KMEANS_ROWS:
        idx = np.sort(rng.choice(X.shape[0], MAX_KMEANS_ROWS, replace=False))
    k = int(min(N_BASIS, max(2, int(idx.size // 500))))
    km = KMeans(n_clusters=k, n_init=4, random_state=SEED).fit(X[idx])
    lab = km.predict(X)
    basis = np.zeros((k, X.shape[1]), dtype=np.float32)
    for c in range(k):
        m = lab == c
        basis[c] = unit[m].mean(0) if m.sum() else unit.mean(0)
    keep = basis.sum(1) > 0
    basis = basis[keep]
    basis = basis / basis.sum(1, keepdims=True)
    ct_list = [f"vis_cluster{c}" for c in np.nonzero(keep)[0]]
    log(f"[info] basis {basis.shape} from visible-only KMeans k={k} "
        f"(fit on {idx.size} rows), sizes={np.bincount(lab)[np.nonzero(keep)[0]].tolist()}")

    genes = vis["genes"]
    obs = pd.DataFrame(
        {"slice": slice_ids, "library_size": lib, "cell_type": "bin"},
        index=[f"b{i}-s{slice_ids[i]}" for i in range(X.shape[0])])
    var = pd.DataFrame({"gene_name": genes}, index=pd.Index(genes, name="gene"))
    adata_st = ad.AnnData(X=X, obs=obs, var=var)
    adata_st.obsm["count"] = X
    adata_st.obsm["3D_coor"] = coords_um.astype(np.float32)
    adata_st.obsm["spatial"] = coords_um[:, :2].astype(np.float32)

    adata_basis = ad.AnnData(X=basis, var=var.copy())
    adata_basis.obs = pd.DataFrame({"celltype": ct_list},
                                   index=pd.Index(ct_list, name="celltype"))
    return dict(adata_st=adata_st, adata_basis=adata_basis, X=X,
                coords_um=coords_um, lib_mean=float(lib.mean()),
                slice_labels=[int(s) for s in order])


def decode_at(model, coords_um, batch=60000):
    import torch
    import anndata as ad
    import scipy.sparse as sp

    model.net.eval()
    out = np.empty((coords_um.shape[0], model.adata_st.n_vars), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, coords_um.shape[0], batch):
            ii = np.arange(s, min(s + batch, coords_um.shape[0]))
            q = ad.AnnData(
                X=sp.csr_matrix((ii.size, model.adata_st.n_vars), dtype=np.float32),
                var=model.adata_st.var.copy(),
                obsm={"3D_coor": coords_um[ii].astype(np.float32)})
            q.obs["slice"] = "target"
            model.inference_latent(q, coord_key="3D_coor", slice_key="slice",
                                   method="distance", decode=True, deconv=False)
            out[ii] = np.asarray(q.obsm["X_pred"], dtype=np.float32)
    return out


def run_dataset(ds, args, status, mon, sampler):
    import torch
    from SINTER3D.model_spatial import Model

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.manual_seed(SEED)
    np.random.seed(SEED)

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
        raise RuntimeError(f"target slice {sorted(tgt_slices)} overlaps visible "
                           f"{sorted(vis_slices)}")
    if vis["n"] < MIN_VISIBLE or vis["n_genes"] < MIN_GENES:
        return dict(status="UNSUPPORTED",
                    reason=f"too small: n_visible={vis['n']}, n_genes={vis['n_genes']} "
                           f"(min {MIN_VISIBLE}/{MIN_GENES})",
                    n_visible=int(vis["n"]), n_target=int(tgt["n"]),
                    n_genes=int(vis["n_genes"]), sec=round(time.time() - t0, 1))

    coords_tr, coords_te, gdiag = C.normalize_geometry(vis, tgt)
    status.log(f"[{ds}] visible={vis['n']} ({len(vis_slices)} slices) "
               f"target={tgt['n']} genes={vis['n_genes']} "
               f"nn_xy={gdiag['nn_xy_native']:.4g} z_gap={gdiag['z_gap_native']:.4g} "
               f"target_slice={sorted(tgt_slices)}")

    tensors = build_tensors(vis, coords_tr, status.log)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    status.write(stage="train", current_dataset=ds)

    t_build = time.time() - t0
    model = Model([], tensors["adata_st"], tensors["adata_basis"],
                  slice_idx=list(range(len(tensors["slice_labels"]))),
                  hidden_dims=HIDDEN_DIMS, slice_emb_dim=SLICE_EMB_DIM,
                  training_steps=args.steps, lr=LR, seed=SEED, patience=PATIENCE,
                  mid_channel=MID_CHANNEL, save_path=os.path.join(OUTDIR, "runs"),
                  use_type="train", normalize=NORMALIZE, coord_emb_dim=COORD_EMB_DIM,
                  num_freqs=NUM_FREQS, alpha_poisson=ALPHA_POISSON,
                  lambda_feature=LAMBDA_FEATURE, gamma_feature=GAMMA_FEATURE)
    model.net.c = model.normalize

    net, opt = model.net, model.optimizer
    net.train()
    best_loss, best_state = float("inf"), None
    n_rows = int(model.X.shape[0])
    bs = int(min(args.batch, n_rows))
    perm = np.random.RandomState(SEED).permutation(n_rows)
    pos = 0
    dev = model.X.device
    t_train = time.time()
    for step in range(args.steps):
        if bs < n_rows:
            if pos + bs > n_rows:
                perm = np.random.permutation(n_rows)
                pos = 0
            ii = torch.from_numpy(np.sort(perm[pos:pos + bs])).to(dev)
            pos += bs
        else:
            ii = None
        if ii is None:
            loss, _, _, _, _, _ = net(
                coord=model.coord, node_feats=model.X, count_matrix=model.Y,
                library_size=model.lY, slice_label=model.slice, basis=model.basis,
                c=model.normalize, step=step)
        else:
            loss, _, _, _, _, _ = net(
                coord=model.coord[ii], node_feats=model.X[ii],
                count_matrix=model.Y[ii], library_size=model.lY[ii],
                slice_label=model.slice[ii], basis=model.basis,
                c=model.normalize, step=step)
        opt.zero_grad()
        loss.backward()
        opt.step()
        li = float(loss.item())
        if best_loss - li > 1e-4:
            best_loss, best_state = li, {"model": net.state_dict()}
        if step % 200 == 0 or step == args.steps - 1:
            status.log(f"  [{ds}] step {step:5d}/{args.steps} loss={li:.4f} "
                       f"best={best_loss:.4f} "
                       f"{(time.time() - t_train) / (step + 1) * 1000:.1f} ms/step")
            status.write(stage="train", current_dataset=ds, step=step,
                         max_steps=args.steps, train_loss=round(li, 6),
                         best_loss=round(best_loss, 6),
                         gpu_peak_mb=int(sampler.peak))
    if best_state is not None:
        net.load_state_dict(best_state["model"])
    train_sec = time.time() - t_train
    alloc = (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0.0
    model.net.c = model.normalize

    t1 = time.time()
    pred = decode_at(model, coords_te.astype(np.float32))
    infer_sec = time.time() - t1
    if not np.isfinite(pred).all():
        raise RuntimeError(f"{ds}: non-finite model output")

    ins_rmse = None
    if args.in_sample:
        t2 = time.time()
        rng = np.random.RandomState(SEED)
        ii = np.arange(tensors["X"].shape[0])
        if ii.size > args.in_sample_rows:
            ii = np.sort(rng.choice(ii, args.in_sample_rows, replace=False))
        rec = decode_at(model, tensors["coords_um"][ii].astype(np.float32))
        ins_rmse = float(np.sqrt(np.mean((rec - tensors["X"][ii]) ** 2)))
        status.log(f"[{ds}] in-sample visible RMSE {ins_rmse:.5f} "
                   f"on {ii.size} rows ({time.time() - t2:.1f}s)")
        del rec
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    C.write_prediction(METHOD, ds, pred, tgt, vis["genes"])
    ver = C.verify_prediction(METHOD, ds, vis["genes"], tgt)
    if not ver["pass"]:
        raise RuntimeError(f"{ds}: prediction verification failed: {ver}")
    rec = dict(status="DONE",
               n_visible=int(vis["n"]), n_target=int(tgt["n"]),
               n_genes=int(vis["n_genes"]),
               n_visible_slices=len(vis_slices),
               target_slice=sorted(tgt_slices),
               visible_slices=sorted(vis_slices),
               steps=int(args.steps), batch=int(min(args.batch, vis["n"])),
               best_train_loss=round(float(best_loss), 6),
               train_sec=round(train_sec, 1),
               build_sec=round(t_build, 1),
               infer_sec=round(infer_sec, 1),
               total_sec=round(time.time() - t0, 1),
               ms_per_step=round(train_sec / max(1, args.steps) * 1000, 1),
               torch_alloc_peak_mb=round(alloc, 1),
               in_sample_visible_rmse=ins_rmse,
               n_basis=int(tensors["adata_basis"].shape[0]),
               geometry=gdiag,
               verify=ver,
               output=C.pred_path(METHOD, ds))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--datasets", default=None, help="comma separated list")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--in-sample", action="store_true", default=True)
    ap.add_argument("--no-in-sample", dest="in_sample", action="store_false")
    ap.add_argument("--in-sample-rows", type=int, default=8000)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    status = C.Status(METHOD, OUTDIR, dict(command=" ".join([sys.executable] + sys.argv),
                                           config=vars(args)))
    mon = C.ResourceMonitor()
    mon.start()
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
                   f"steps={args.steps}")
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
                rec = run_dataset(ds, args, status, mon, sampler)
            except Exception as e:
                tb = traceback.format_exc()
                status.log(f"[{ds}] FAILED {e!r}\n{tb}")
                rec = dict(status="FAILED", error=repr(e), traceback=tb[-2500:])
                try:
                    import torch
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

        mon.stop_evt.set()
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
            "input_mode": "canonical visible.h5ad (all visible slices), canonical X used "
                          "as-is (no renormalisation); target.npz X never read",
            "output": "predictions/<dataset>/SINTER3D.npz (target rows only, no volume)",
            "config": dict(steps=args.steps, batch=args.batch,
                           batch_note="mini-batch SGD (official is full-batch); "
                                      "same loss/optimiser/lr/step budget",
                           hidden_dims=HIDDEN_DIMS,
                           slice_emb_dim=SLICE_EMB_DIM, mid_channel=MID_CHANNEL,
                           lr=LR, seed=SEED, normalize=NORMALIZE,
                           alpha_poisson=ALPHA_POISSON,
                           lambda_feature=LAMBDA_FEATURE,
                           gamma_feature=GAMMA_FEATURE, num_freqs=NUM_FREQS,
                           coord_emb_dim=COORD_EMB_DIM, n_basis_cap=N_BASIS,
                           coord_frame=gdiag_frame(),
                           basis="visible-only KMeans <=8, unit-profile mean"),
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
            "resources": mon.summary(),
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
                      gpu_peak_mb=int(sampler.peak), metrics="methods/SINTER3D/compute_metrics.json")
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
            "resources": mon.summary(),
            "config": metrics["config"],
            "metrics": "methods/SINTER3D/compute_metrics.json",
            "status_file": "methods/SINTER3D/status.json",
            "log": "methods/SINTER3D/run.log",
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


def gdiag_frame():
    return (f"visible-only: xy median within-slice NN -> {C.C2C_UM} units, "
            f"z median inter-slice gap -> {C.C2C_UM} units")


if __name__ == "__main__":
    sys.exit(main())

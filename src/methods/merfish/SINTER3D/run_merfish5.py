#!/usr/bin/env python
"""SINTER3D for the MERFISH 5-slice target-only benchmark.

Per scenario: train on visible-section bins only, decode the hidden target bins,
write predictions/<scenario>/SINTER3D.npz aligned to canonical/target_gt.npz.

Adapted from Workspace/merfish_test/SINTER3D/run_sinter3d_50um.py (same official
model_spatial.Model, loss, optimiser and hyper-parameters). Two documented
adaptations for this dataset:
  * canonical X is already normalised, so it is used directly as the count-like
    target (`count`), with library_size = row sums; no renormalisation. The
    decoder therefore outputs canonical-X units directly (no expm1 back-scale).
  * canonical bins carry no cell-type labels (leakage guard), so the deconv
    basis is built from KMeans clusters of the *visible* bins only
    (unit-normalised mean profiles). Target expression never enters the basis.
The decoder path (X_pred) is coordinate-only; slice embeddings only affect the
deconvolution loss, so hidden sections need no slice label.
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

METHOD = "SINTER3D"
OUTDIR = os.path.join(hc.METHODS_ROOT, "SINTER3D")
SINTER3D_DIR = os.path.join(os.environ["MERFISH5_UPSTREAM"], "SINTER3D", "SINTER3D-master")
if SINTER3D_DIR not in sys.path:
    sys.path.insert(0, SINTER3D_DIR)

HIDDEN_DIMS = [512, 128]
SLICE_EMB_DIM = 16
MID_CHANNEL = 200
LR = 1e-3
SEED = 2024
PATIENCE = 10 ** 9
NORMALIZE = 100.0
ALPHA_POISSON = 5
LAMBDA_FEATURE = 4
GAMMA_FEATURE = 3
NUM_FREQS = 6
COORD_EMB_DIM = 64
N_BASIS = 8


def build_tensors(prob, bins, log):
    """Training tensors from visible bins only (canonical X, no renormalisation)."""
    import anndata as ad
    import pandas as pd

    X = np.ascontiguousarray(prob["expr_tr"], dtype=np.float32)
    coords_um = prob["coords_tr"].astype(np.float32)
    sec = bins["section"][prob["tr_idx"]]
    sec_labels = [str(s) for s in prob["visible"]]
    sid = {s: i for i, s in enumerate(sec_labels)}
    slice_ids = np.array([sid[s] for s in sec], dtype=np.int64)

    lib = X.sum(axis=1).astype(np.float32)
    log(f"[info] library_size (rowsum canonical X) min/mean/max = "
        f"{lib.min():.3f}/{lib.mean():.3f}/{lib.max():.3f}; zero rows {int((lib <= 0).sum())}")
    lib_safe = np.clip(lib, 1e-6, None)

    # deconv basis: KMeans over visible bins, unit-normalised cluster mean profiles
    from sklearn.cluster import KMeans
    k = int(min(N_BASIS, max(2, X.shape[0] // 500)))
    km = KMeans(n_clusters=k, n_init=4, random_state=SEED).fit(X)
    lab = km.labels_
    unit = X / lib_safe[:, None]
    basis = np.zeros((k, X.shape[1]), dtype=np.float32)
    for c in range(k):
        m = lab == c
        if m.sum() == 0:
            basis[c] = unit.mean(0)
        else:
            basis[c] = unit[m].mean(0)
    keep = basis.sum(1) > 0
    basis = basis[keep]
    ct_list = [f"vis_cluster{c}" for c in np.nonzero(keep)[0]]
    basis = basis / basis.sum(1, keepdims=True)
    log(f"[info] basis: {basis.shape} from visible-only KMeans (k={k}), "
        f"sizes={np.bincount(lab)[np.nonzero(keep)[0]].tolist()}")

    obs = pd.DataFrame(
        {"slice": slice_ids, "library_size": lib, "cell_type": "bin"},
        index=[f"b{i}-s{slice_ids[i]}" for i in range(X.shape[0])])
    genes = bins["genes"]
    var = pd.DataFrame({"gene_name": genes}, index=pd.Index(genes, name="gene"))
    adata_st = ad.AnnData(X=X, obs=obs, var=var)
    adata_st.obsm["count"] = X
    adata_st.obsm["3D_coor"] = coords_um
    adata_st.obsm["spatial"] = coords_um[:, :2]

    adata_basis = ad.AnnData(X=basis, var=var.copy())
    adata_basis.obs = pd.DataFrame({"celltype": ct_list},
                                   index=pd.Index(ct_list, name="celltype"))
    return dict(adata_st=adata_st, adata_basis=adata_basis, X=X,
                slice_labels=sec_labels, lib_mean=float(lib.mean()), coords_um=coords_um)


def decode_at(model, coords_um, batch=60000):
    import torch
    import scipy.sparse as sp

    model.net.eval()
    out = np.empty((coords_um.shape[0], model.adata_st.n_vars), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, coords_um.shape[0], batch):
            ii = np.arange(s, min(s + batch, coords_um.shape[0]))
            import anndata as ad
            q = ad.AnnData(
                X=sp.csr_matrix((ii.size, model.adata_st.n_vars), dtype=np.float32),
                var=model.adata_st.var.copy(),
                obsm={"3D_coor": coords_um[ii].astype(np.float32)})
            q.obs["slice"] = "target"
            model.inference_latent(q, coord_key="3D_coor", slice_key="slice",
                                   method="distance", decode=True, deconv=False)
            out[ii] = np.asarray(q.obsm["X_pred"], dtype=np.float32)
    return out


def run_scenario(scen, bins, tmeta, args, status, mon):
    import torch
    from SINTER3D.model_spatial import Model

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    prob = hc.build_problem(scen, bins, tmeta)
    assert not (set(prob["visible"]) & set(prob["target"]))
    assert not np.isin(bins["section"][prob["tr_idx"]], prob["target"]).any()

    status.log(f"[{prob['name']}] visible={prob['visible']} target={prob['target']} "
               f"train={prob['expr_tr'].shape} predict={prob['coords_te'].shape}")
    t0 = time.time()
    tensors = build_tensors(prob, bins, status.log)
    torch.cuda.reset_peak_memory_stats()

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
    t_train = time.time()
    for step in range(args.steps):
        loss, _, _, _, _, _ = net(
            coord=model.coord, node_feats=model.X, count_matrix=model.Y,
            library_size=model.lY, slice_label=model.slice, basis=model.basis,
            c=model.normalize, step=step)
        opt.zero_grad()
        loss.backward()
        opt.step()
        li = float(loss.item())
        if best_loss - li > 1e-4:
            best_loss, best_state = li, {"model": net.state_dict(), "optimizer": opt.state_dict()}
        if step % 200 == 0 or step == args.steps - 1:
            status.log(f"  step {step:5d}/{args.steps} loss={li:.4f} best={best_loss:.4f} "
                       f"{(time.time()-t_train)/(step+1)*1000:.1f} ms/step")
            status.write(stage="train", current_scenario=prob["name"], step=step,
                         max_iter=args.steps, train_loss=round(li, 4))
    if best_state is not None:
        net.load_state_dict(best_state["model"])
    train_sec = time.time() - t_train
    alloc = torch.cuda.max_memory_allocated() / 2**20
    model.net.c = model.normalize

    t1 = time.time()
    pred = decode_at(model, prob["coords_te"].astype(np.float32))
    infer_sec = time.time() - t1

    ins_rmse = None
    if args.in_sample:
        t2 = time.time()
        rec = decode_at(model, tensors["coords_um"])
        ins_rmse = float(np.sqrt(np.mean((rec - tensors["X"]) ** 2)))
        status.log(f"[{prob['name']}] in-sample visible RMSE {ins_rmse:.5f} "
                   f"({time.time()-t2:.1f}s)")
        del rec
    del model
    torch.cuda.empty_cache()

    path = hc.write_prediction(METHOD, prob, pred)
    res = hc.verify_prediction(path, METHOD, prob)
    res.update({
        "scenario": prob["name"], "k": prob["k"],
        "visible": prob["visible"], "target": prob["target"],
        "n_train_bins": int(tensors["X"].shape[0]),
        "n_target_bins": int(prob["row_index"].size),
        "train_sec": round(train_sec, 1), "infer_sec": round(infer_sec, 1),
        "scenario_sec": round(time.time() - t0, 1),
        "torch_alloc_peak_mb": round(alloc, 1),
        "best_train_loss": round(best_loss, 5),
        "in_sample_visible_rmse": ins_rmse,
        "n_basis": int(tensors["adata_basis"].shape[0]),
    })
    cfg = dict(steps=args.steps, hidden_dims=HIDDEN_DIMS, slice_emb_dim=SLICE_EMB_DIM,
               mid_channel=MID_CHANNEL, lr=LR, seed=SEED, normalize=NORMALIZE,
               alpha_poisson=ALPHA_POISSON, lambda_feature=LAMBDA_FEATURE,
               gamma_feature=GAMMA_FEATURE, num_freqs=NUM_FREQS,
               coord_emb_dim=COORD_EMB_DIM, coords_unit="um",
               output_space="canonical X (no renormalisation, no expm1 back-scale)",
               basis="visible-only KMeans, k<=8, unit-profile mean")
    return res, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--steps", type=int, default=6000)
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
            "method": METHOD, "dataset": "merfish_5slice_all",
            "model": "SINTER3D model_spatial (official), decoder output in canonical X space",
            "input_mode": "canonical 50um bins, visible sections only",
            "output": "predictions/<scenario>/SINTER3D.npz (target bins only, no volume)",
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
                      coverage=1.0, metrics="methods/SINTER3D/compute_metrics.json",
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

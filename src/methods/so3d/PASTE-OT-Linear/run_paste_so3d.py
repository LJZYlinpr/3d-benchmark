#!/usr/bin/env python
"""PASTE-OT-Linear for the So3D 46 held-out-slice benchmark.

Adapted from Workspace/merfish_test/missing_combos_50um/all_methods/methods/
PASTE-OT-Linear/run_paste_heart.py:
  * entropic fused Gromov-Wasserstein OT (POT: KL expression dissimilarity,
    square_loss spatial cost, alpha=0.1, epsilon=0.05, PPA) between the two
    visible slices bracketing the held-out slice;
  * each visible slice's cloud is transported into the other frame and placed
    at z-interpolated xy with z-linear blended expression;
  * held-out target spots take inverse-distance (k=4) interpolation of the
    aligned visible clouds.

Only visible-slice expression participates in OT / blending.  Target
expression is never read.  Target-only output (no volume).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
import _so3d_common as C  # noqa: E402

METHOD = "PASTE-OT-Linear"


def kl_cost_gpu(X_a, X_b, device):
    import torch
    eps = 1e-12
    p = X_a / (X_a.sum(1, keepdims=True) + eps)
    q = X_b / (X_b.sum(1, keepdims=True) + eps)
    pt = torch.from_numpy(p).to(device)
    qt = torch.from_numpy(q).to(device)
    lp = torch.from_numpy(np.log(p + eps)).to(device)
    lq = torch.from_numpy(np.log(q + eps)).to(device)
    h = (pt * lp).sum(1)
    M = h[:, None] - pt @ lq.T
    return M / M.clamp_min(1e-12).max()


def solve_pair(Xl, xl, Xu, xu, alpha, epsilon, max_iter, device, log, tol=1e-7):
    import torch
    from ot.gromov import entropic_fused_gromov_wasserstein as efgw

    t0 = time.time()
    M = kl_cost_gpu(Xl, Xu, device)
    xlt = torch.from_numpy(np.ascontiguousarray(xl, dtype=np.float32)).to(device)
    xut = torch.from_numpy(np.ascontiguousarray(xu, dtype=np.float32)).to(device)
    C1 = torch.cdist(xlt, xlt)
    C1 = C1 / C1.clamp_min(1e-12).max()
    C2 = torch.cdist(xut, xut)
    C2 = C2 / C2.clamp_min(1e-12).max()
    a = torch.ones(len(xl), device=device, dtype=torch.float32) / len(xl)
    b = torch.ones(len(xu), device=device, dtype=torch.float32) / len(xu)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    pi, lg = efgw(M, C1, C2, a, b, loss_fun="square_loss", epsilon=epsilon,
                  alpha=alpha, max_iter=max_iter, tol=tol, solver="PPA", log=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ot_sec = time.time() - t1
    pi = pi.float()
    P = pi / pi.sum(1, keepdim=True).clamp_min(1e-30)
    Q = pi / pi.sum(0, keepdim=True).clamp_min(1e-30)
    xlt_f = torch.from_numpy(np.ascontiguousarray(xl, dtype=np.float32)).to(device)
    xut_f = torch.from_numpy(np.ascontiguousarray(xu, dtype=np.float32)).to(device)
    Xlt = torch.from_numpy(np.ascontiguousarray(Xl, dtype=np.float32)).to(device)
    Xut = torch.from_numpy(np.ascontiguousarray(Xu, dtype=np.float32)).to(device)
    out = dict(
        cy=(P @ xut_f).cpu().numpy().astype(np.float32),
        cx=(Q.T @ xlt_f).cpu().numpy().astype(np.float32),
        Yu=(P @ Xut).cpu().numpy().astype(np.float32),
        Xl=(Q.T @ Xlt).cpu().numpy().astype(np.float32),
    )
    plan = (pi / pi.sum()).clamp_min(1e-30)
    ent = float(-(plan * plan.log()).sum())
    disp_l = np.linalg.norm(out["cy"] - xl, axis=1)
    disp_u = np.linalg.norm(out["cx"] - xu, axis=1)
    meta = dict(ot_sec=round(ot_sec, 2), pair_sec=round(time.time() - t0, 2),
                n_l=int(len(xl)), n_u=int(len(xu)), niter=lg.get("niter"),
                solver="PPA", epsilon=epsilon, alpha=alpha, tol=tol, max_iter=max_iter,
                fgw_dist=float(lg["fgw_dist"]) if "fgw_dist" in lg else None,
                plan_entropy=round(ent, 3),
                plan_entropy_max=round(float(np.log(len(xl) * len(xu))), 3),
                disp=dict(lower=dict(median=float(np.median(disp_l)),
                                     p95=float(np.percentile(disp_l, 95))),
                          upper=dict(median=float(np.median(disp_u)),
                                     p95=float(np.percentile(disp_u, 95)))),
                rowsum_err=float((pi.sum(1) - a).abs().max()),
                colsum_err=float((pi.sum(0) - b).abs().max()),
                gpu_alloc_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1)
                if torch.cuda.is_available() else 0.0)
    del M, C1, C2, pi, P, Q
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"[ot] n={len(xl)}x{len(xu)} iters={meta['niter']} ot={ot_sec:.1f}s "
        f"gpu={meta['gpu_alloc_peak_mb']:.0f}MB")
    return out, meta


def predict_points(t, ot, xl, xu, Xl, Xu, qxy, knn):
    wl, wu = 1.0 - t, t
    n_l, n_u = len(xl), len(xu)
    pos = np.empty((n_l + n_u, 2), dtype=np.float32)
    val = np.empty((n_l + n_u, Xl.shape[1]), dtype=np.float32)
    np.multiply(xl, wl, out=pos[:n_l])
    pos[:n_l] += wu * ot["cy"]
    np.multiply(xu, wu, out=pos[n_l:])
    pos[n_l:] += wl * ot["cx"]
    np.multiply(Xl, wl, out=val[:n_l])
    val[:n_l] += wu * ot["Yu"]
    np.multiply(Xu, wu, out=val[n_l:])
    val[n_l:] += wl * ot["Xl"]
    tree = cKDTree(pos.astype(np.float64))
    dd, ii = tree.query(qxy, k=knn, workers=4)
    dd = dd.astype(np.float32)
    w = 1.0 / (dd + 1e-4) ** 2
    w /= w.sum(1, keepdims=True)
    out = np.zeros((len(qxy), val.shape[1]), dtype=np.float32)
    for k in range(knn):
        out += w[:, k:k + 1] * val[ii[:, k]]
    return out, dd[:, 0]


def predict_dataset(vis, tgt, args, dev, log):
    order, z_of = C.slice_order(vis)
    tz = float(np.median(tgt["coords"][:, 2]))
    lows = [s for s in order if z_of[s] <= tz]
    highs = [s for s in order if z_of[s] >= tz]
    if lows and highs and lows[-1] != highs[0]:
        lo, hi = lows[-1], highs[0]
        bracket = "bracketing"
    elif lows and highs and lows[-1] == highs[0]:
        # target z coincides with a visible slice z: use its two neighbours
        i = order.index(lows[-1])
        lo = order[max(0, i - 1)]
        hi = order[min(len(order) - 1, i + 1)]
        bracket = "coinincident-visible-neighbours"
    elif len(order) >= 2:
        # edge case: take the two nearest visible slices (extrapolate, clipped)
        near = sorted(order, key=lambda s: abs(z_of[s] - tz))[:2]
        near.sort(key=lambda s: z_of[s])
        lo, hi = near[0], near[1]
        bracket = "nearest-two-visible (edge extrapolation)"
    else:
        raise RuntimeError("need >=2 visible slices for PASTE bracket")
    if lo == hi or lo in set(tgt["slice"].tolist()) or hi in set(tgt["slice"].tolist()):
        raise RuntimeError(f"invalid bracket {lo}->{hi}")
    t = (tz - z_of[lo]) / (z_of[hi] - z_of[lo]) if z_of[hi] != z_of[lo] else 0.5
    t = float(np.clip(t, 0.0, 1.0))

    def rows_of(s):
        return np.flatnonzero(vis["slice"] == s)

    lo_rows, hi_rows = rows_of(lo), rows_of(hi)
    rng = np.random.default_rng(2024)
    if args.max_anchors and lo_rows.size > args.max_anchors:
        lo_rows = np.sort(rng.choice(lo_rows, args.max_anchors, replace=False))
    if args.max_anchors and hi_rows.size > args.max_anchors:
        hi_rows = np.sort(rng.choice(hi_rows, args.max_anchors, replace=False))

    Xl = np.ascontiguousarray(vis["X"][lo_rows])
    Xu = np.ascontiguousarray(vis["X"][hi_rows])
    xl = np.ascontiguousarray(vis["coords"][lo_rows, :2], dtype=np.float32)
    xu = np.ascontiguousarray(vis["coords"][hi_rows, :2], dtype=np.float32)
    ot, otmeta = solve_pair(Xl, xl, Xu, xu, args.alpha, args.epsilon,
                            args.max_iter, dev, log, args.tol)
    pred, d0 = predict_points(t, ot, xl, xu, Xl, Xu,
                              tgt["coords"][:, :2].astype(np.float32), args.knn)
    meta = {
        "bracket": [lo, hi], "bracket_kind": bracket,
        "bracket_z": [z_of[lo], z_of[hi]], "target_z": tz, "t": t,
        "n_lower": int(lo_rows.size), "n_upper": int(hi_rows.size),
        "ot": otmeta,
        "map_dist": dict(mean=float(d0.mean()), median=float(np.median(d0)),
                         p95=float(np.percentile(d0, 95)), max=float(d0.max())),
    }
    return pred, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--knn", type=int, default=4)
    ap.add_argument("--max-anchors", type=int, default=20000,
                    help="cap visible spots per bracket for the OT cost matrix")
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

    def body():
        t0 = time.time()
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        dev = args.device if torch.cuda.is_available() else "cpu"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        log(f"[env] host={os.uname().nodename} torch={torch.__version__} cuda={torch.cuda.is_available()} "
            f"device={dev} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
        dss = C.select_datasets(args.datasets)
        if args.pilot:
            dss = dss[:1]
        log(f"[data] {len(dss)} datasets, pilot={args.pilot}")
        status.update(stage="predict", datasets_total=len(dss))
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
                pred, meta = predict_dataset(vis, tgt, args, dev, log)
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
                log(f"[pred] {ds} n_t={tgt['n']} g={vis['n_vars']} bracket={meta['bracket']} "
                    f"t={meta['t']:.3f} mapmed={meta['map_dist']['median']:.3g} "
                    f"{time.time()-ts:.2f}s pass={v['pass']}")
            except Exception:
                import traceback
                tb = traceback.format_exc()
                failed[ds] = {"status": "FAILED", "error": tb[-3000:]}
                C.write_ds_status(ds, METHOD, "FAILED",
                                  tb.strip().splitlines()[-1] if tb.strip() else "error",
                                  error=tb[-3000:])
                log(f"[fail] {ds}\n{tb}")
            status.update(stage="predict", datasets_done=i + 1, last_dataset=ds,
                          failures=len(failed), gpu_peak_mb=mon.peak_mb)

        mon.stop()
        total = time.time() - t0
        all_pass = all(v["pass"] for v in ver) and not failed
        metrics = dict(
            method=METHOD,
            method_full="PASTE entropic FGW (POT PPA) between visible bracket slices + "
                        "z-linear blend + IDW(k=4) on aligned clouds",
            status="DONE" if all_pass else "FAILED",
            lane=C.LANE, host=os.uname().nodename,
            slurm_job=os.environ.get("SLURM_JOB_ID"), pilot=args.pilot,
            n_datasets=len(dss), datasets=dss, n_done=len(per), n_failed=len(failed),
            coverage=round(len(per) / len(dss), 4) if dss else 0.0,
            total_target_rows=int(sum(p["n_target"] for p in per.values())),
            config=dict(alpha=args.alpha, epsilon=args.epsilon,
                        dissimilarity="KL", loss_fun="square_loss",
                        solver="POT entropic_fused_gromov_wasserstein (PPA)",
                        max_iter=args.max_iter, tol=args.tol, knn=args.knn,
                        knn_power=2, max_anchors=args.max_anchors,
                        blend="z-linear between bracketing visible slices",
                        device=dev),
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

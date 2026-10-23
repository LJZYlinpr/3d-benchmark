#!/usr/bin/env python
"""PASTE-OT-Linear for the MERFISH 5-slice target-only benchmark.

Adapted from Workspace/merfish_test/PASTE-OT-Linear/run_paste_ot_linear.py:
  * entropic fused Gromov-Wasserstein OT (POT: KL expression dissimilarity,
    square_loss spatial cost, alpha=0.1, epsilon=0.05, PGD) between the two
    visible slices bracketing a hidden block;
  * each slice's point cloud is transported into the other frame and placed at
    its z-interpolated xy with z-linear blended expression;
  * hidden target bins take inverse-distance (k=4) interpolation of the
    aligned visible clouds.

Only visible sections are used for OT and blending; target expression is
never read.  Output is target bins only (no volume).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _rb_common as C  # noqa: E402

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
    return M / M.max()


def solve_pair(Xl, xl, Xu, xu, alpha, epsilon, max_iter, device, log, tol=1e-7):
    import torch
    from ot.gromov import entropic_fused_gromov_wasserstein as efgw

    t0 = time.time()
    M = kl_cost_gpu(Xl, Xu, device)
    xlt = torch.from_numpy(np.ascontiguousarray(xl, dtype=np.float32)).to(device)
    xut = torch.from_numpy(np.ascontiguousarray(xu, dtype=np.float32)).to(device)
    C1 = torch.cdist(xlt, xlt)
    C1 = C1 / C1.max()
    C2 = torch.cdist(xut, xut)
    C2 = C2 / C2.max()
    a = torch.ones(len(xl), device=device, dtype=torch.float32) / len(xl)
    b = torch.ones(len(xu), device=device, dtype=torch.float32) / len(xu)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    # The frozen PASTE config (alpha=0.1, epsilon=0.05) is unchanged; the
    # reference used solver="PGD", which here collapses to the uniform plan
    # (POT warns "numerical errors at iteration 0"; plan entropy == log(n*m)).
    # PPA solves the same entropic FGW objective to convergence.
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
        cy=(P @ xut_f).cpu().numpy().astype(np.float32),      # lower -> expected upper xy
        cx=(Q.T @ xlt_f).cpu().numpy().astype(np.float32),    # upper -> expected lower xy
        Yu=(P @ Xut).cpu().numpy().astype(np.float32),        # lower -> expected upper expr
        Xl=(Q.T @ Xlt).cpu().numpy().astype(np.float32),      # upper -> expected lower expr
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
                disp_um=dict(lower=dict(median=float(np.median(disp_l)),
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


def predict_section(t, ot, xl, xu, Xl, Xu, qxy, knn):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--knn", type=int, default=4)
    ap.add_argument("--max-anchors", type=int, default=6000,
                    help="0 = full slice; otherwise deterministic random anchor subset per bracket")
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

    def body():
        t0 = time.time()
        import torch
        import os as _os
        if not _os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
            import torch as _torch
            _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                "expandable_segments:True"
                if int(_torch.__version__.split(".")[0]) >= 2
                else "max_split_size_mb:512")
        dev = args.device if torch.cuda.is_available() else "cpu"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.empty_cache()
        log(f"[env] anchors_max={args.max_anchors} host={__import__('os').uname().nodename} torch={torch.__version__} "
            f"cuda={torch.cuda.is_available()} device={dev} "
            f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

        bins = C.load_bins()
        tg = C.load_target_gt()
        all_sc = C.load_scenarios()
        scs = C.select_scenarios(all_sc, "M1a" if args.pilot else args.scenarios)
        sec_rows = {s: np.flatnonzero(bins["section"] == s) for s in C.SECTIONS}
        z_of = {s: float(bins["z"][sec_rows[s]].mean()) for s in C.SECTIONS}
        xy_of = {s: np.ascontiguousarray(
            np.stack([bins["x"][sec_rows[s]], bins["y"][sec_rows[s]]], 1), dtype=np.float32)
            for s in C.SECTIONS}
        log(f"[data] bins={bins['X'].shape} scenarios={len(scs)}")

        per = {}
        ver = []
        for i, sc in enumerate(scs):
            ts = time.time()
            tgt = sorted(sc["target"], key=int)
            vis = [s for s in sc["visible"] if s in sec_rows]
            lo = max((v for v in vis if int(v) < int(tgt[0])), key=int)
            hi = min((v for v in vis if int(v) > int(tgt[-1])), key=int)
            log(f"[{sc['name']}] bracket {lo}->{hi} hidden={tgt}")
            lo_rows, hi_rows = sec_rows[lo], sec_rows[hi]
            if args.max_anchors and lo_rows.size > args.max_anchors:
                rng = np.random.default_rng(2024)
                lo_rows = np.sort(rng.choice(lo_rows, args.max_anchors, replace=False))
            if args.max_anchors and hi_rows.size > args.max_anchors:
                rng = np.random.default_rng(2025)
                hi_rows = np.sort(rng.choice(hi_rows, args.max_anchors, replace=False))
            Xl_an = np.ascontiguousarray(bins["X"][lo_rows])
            Xu_an = np.ascontiguousarray(bins["X"][hi_rows])
            xl_an = np.ascontiguousarray(np.stack([bins["x"][lo_rows], bins["y"][lo_rows]], 1),
                                         dtype=np.float32)
            xu_an = np.ascontiguousarray(np.stack([bins["x"][hi_rows], bins["y"][hi_rows]], 1),
                                         dtype=np.float32)
            log(f"[{sc['name']}] anchors {lo}:{lo_rows.size}/{sec_rows[lo].size} "
                f"{hi}:{hi_rows.size}/{sec_rows[hi].size}")
            ot, meta = solve_pair(
                Xl_an, xl_an, Xu_an, xu_an,
                args.alpha, args.epsilon, args.max_iter, dev, log, args.tol)
            rows = C.scenario_rows(tg, sc)
            pred = np.empty((rows.size, C.N_GENES), dtype=np.float32)
            tgt_section = tg["section"][rows]
            tgt_bin = tg["bin_id"][rows]
            dists = {}
            for s in tgt:
                m = tgt_section == s
                b = sec_rows[s]
                bid = {x: k for k, x in enumerate(bins["bin_id"][b])}
                missing = [x for x in tgt_bin[m] if x not in bid]
                if missing:
                    raise RuntimeError(f"target bin missing from bins_all: {missing[:3]}")
                sel = np.array([bid[x] for x in tgt_bin[m]], dtype=np.int64)
                qxy = np.stack([bins["x"][b][sel], bins["y"][b][sel]], 1).astype(np.float32)
                tt = (z_of[s] - z_of[lo]) / (z_of[hi] - z_of[lo])
                tt = float(np.clip(tt, 0.0, 1.0))
                p, d0 = predict_section(tt, ot, xl_an, xu_an, Xl_an, Xu_an,
                                        qxy, args.knn)
                pred[m] = p
                dists[s] = dict(t=round(tt, 4), n=int(m.sum()),
                                map_dist_um=dict(mean=float(d0.mean()),
                                                 median=float(np.median(d0)),
                                                 p95=float(np.percentile(d0, 95)),
                                                 max=float(d0.max())))
                log(f"[{sc['name']}] section {s} bins={int(m.sum())} t={tt:.3f} "
                    f"mapdist med={np.median(d0):.1f}um")
            path = C.save_pred(METHOD, sc, tg, rows, pred)
            v = C.verify_pred(path, tg, rows, sc, METHOD)
            ver.append(v)
            per[sc["name"]] = dict(n_rows=int(rows.size), bracket=[lo, hi],
                                   target=tgt, ot=meta, sections=dists,
                                   sec=round(time.time() - ts, 2))
            log(f"[pred] {sc['name']} rows={rows.size} {time.time() - ts:.1f}s pass={v['pass']}")
            status.update(stage="predict", scenarios_done=i + 1, last_scenario=sc["name"],
                          scenarios=list(per), gpu_peak_mb=mon.peak_mb)
        mon.stop()
        total = time.time() - t0
        ok = all(v["pass"] for v in ver)
        metrics = dict(
            method=METHOD, status="DONE" if ok else "FAILED_VERIFY", pilot=args.pilot,
            lane=C.LANE, host=__import__("os").uname().nodename,
            slurm_job=__import__("os").environ.get("SLURM_JOB_ID"),
            n_scenarios=len(scs), scenarios=[s["name"] for s in scs],
            n_target_rows=int(sum(p["n_rows"] for p in per.values())),
            config=dict(alpha=args.alpha, epsilon=args.epsilon,
                        dissimilarity="KL", loss_fun="square_loss",
                        solver="POT entropic_fused_gromov_wasserstein (PPA)",
                        solver_note="reference used PGD; PGD degenerates to the "
                                    "uniform plan on these 1122-gene bins, PPA "
                                    "solves the identical objective",
                        max_iter=args.max_iter, knn=args.knn, knn_power=2,
                        anchors=f"deterministic random subset, max {args.max_anchors} bins per bracket (0=full)",
                        blend="z-linear between bracketing visible slices",
                        device=dev),
            compute=dict(cpu_cores=int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", 1)),
                         gpu_peak_mb=mon.peak_mb, gpu_util_peak_pct=mon.peak_util,
                         ram_peak_mb=round(C.ram_peak_mb(), 1),
                         total_sec=round(total, 2)),
            per_scenario=per, verification=ver, finished_at=C.now())
        C.write_json(outdir / ("compute_metrics_pilot.json" if args.pilot else "compute_metrics.json"),
                     metrics)
        status.update(status="DONE" if ok else "FAILED_VERIFY", stage="done",
                      n_scenarios=len(scs), total_sec=round(total, 2),
                      n_target_rows=metrics["n_target_rows"], gpu_peak_mb=mon.peak_mb,
                      verification_all_pass=ok)
        C.update_runner_status(METHOD, dict(
            status="DONE" if ok else "FAILED_VERIFY", n_scenarios=len(scs),
            n_target_rows=metrics["n_target_rows"], total_sec=round(total, 2),
            gpu_peak_mb=mon.peak_mb, metrics=str(outdir / "compute_metrics.json")))
        log(f"[done] {len(scs)} scenarios rows={metrics['n_target_rows']} "
            f"total={total:.1f}s gpu_peak={mon.peak_mb}MB all_pass={ok}")

    C.run_guard(METHOD, status, log, body)


if __name__ == "__main__":
    main()

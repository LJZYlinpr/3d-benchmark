#!/usr/bin/env python
"""ReHo for the So3D 46 held-out-slice benchmark.

Repeat-Holding: copy expression from a truly visible source slice.
  * source slice = visible slice whose median z is closest to the target slice
    (ties -> lower z, matching ReHo's "copy preceding" spirit);
  * within that visible slice copy the expression of the nearest spot in xy.
Target expression is never read; target coordinates are used as query points
only.  Target-only output (no volume).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
import _so3d_common as C  # noqa: E402

METHOD = "ReHo"


def predict_dataset(vis, tgt, log):
    order, z_of = C.slice_order(vis)
    vz = vis["coords"][:, 2]
    vxy = vis["coords"][:, :2].astype(np.float64)
    tz = float(np.median(tgt["coords"][:, 2])) if tgt["n"] else 0.0

    vis = dict(vis)
    vis["_order"] = order
    if len(order) == 0:
        raise RuntimeError("no visible slices")
    src = min(order, key=lambda s: (abs(z_of[s] - tz), z_of[s]))
    src_mask = vis["slice"] == src
    src_idx = np.flatnonzero(src_mask)
    if src_idx.size == 0:
        raise RuntimeError(f"chosen source slice {src} is empty")
    if src in set(tgt["slice"].tolist()):
        raise RuntimeError(f"source slice {src} is a target slice (leakage guard)")

    tree = cKDTree(vxy[src_idx])
    dist, nn = tree.query(tgt["coords"][:, :2].astype(np.float64), k=1, workers=4)
    pred = vis["X"][src_idx][nn].astype(np.float32)
    meta = {
        "source_slice": src,
        "source_z": z_of[src],
        "source_rule": "nearest visible median z; tie -> lower z",
        "target_z": tz,
        "z_gap": float(abs(z_of[src] - tz)),
        "visible_slices": order,
        "visible_slice_z": {k: round(v, 6) for k, v in z_of.items()},
        "n_source_spots": int(src_idx.size),
        "n_target_spots": int(tgt["n"]),
        "xy_dist_mean": float(dist.mean()),
        "xy_dist_median": float(np.median(dist)),
        "xy_dist_p95": float(np.percentile(dist, 95)),
        "xy_dist_max": float(dist.max()),
    }
    return pred, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    outdir = C.METHODS_ROOT / METHOD
    outdir.mkdir(parents=True, exist_ok=True)
    log = C.RunLog(outdir / "run.log")
    status = C.Status(outdir / "status.json", METHOD)
    status.update(status="RUNNING", stage="load", pilot=args.pilot,
                  command=" ".join([sys.executable] + sys.argv))
    C.set_method(METHOD, "RUNNING", stage="load", datasets_done=0, coverage=0.0,
                 metrics=str(outdir / "compute_metrics.json"))

    def body():
        t0 = time.time()
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
                pred, meta = predict_dataset(vis, tgt, log)
                C.write_prediction(ds, METHOD, pred, tgt,
                                   extra={"genes": vis["genes"].astype(str)})
                v = C.verify_prediction(ds, METHOD, tgt)
                ver.append(v)
                per[ds] = {"sec": round(time.time() - ts, 3),
                           "n_target": int(tgt["n"]), "n_genes": int(vis["n_vars"]),
                           "n_visible_spots": int(vis["n_obs"]),
                           "target_slices": sorted(set(tgt["slice"].tolist())),
                           "visible_slices": sorted(set(vis["slice"].tolist())),
                           **meta, "verification": v}
                C.write_ds_status(ds, METHOD, "DONE", "",
                                  n_target=int(tgt["n"]), n_genes=int(vis["n_vars"]),
                                  runtime_sec=round(time.time() - ts, 3),
                                  source_slice=meta["source_slice"],
                                  z_gap=meta["z_gap"])
                log(f"[pred] {ds} n_t={tgt['n']} g={vis['n_vars']} src={meta['source_slice']} "
                    f"z_gap={meta['z_gap']:.4g} xy_med={meta['xy_dist_median']:.3g} "
                    f"{time.time()-ts:.2f}s pass={v['pass']}")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                failed[ds] = {"status": "FAILED", "error": tb[-3000:]}
                C.write_ds_status(ds, METHOD, "FAILED",
                                  tb.strip().splitlines()[-1] if tb.strip() else "error",
                                  error=tb[-3000:])
                log(f"[fail] {ds}\n{tb}")
            status.update(stage="predict", datasets_done=i + 1, last_dataset=ds,
                          failures=len(failed), gpu="none (CPU-only)")

        total = time.time() - t0
        all_pass = all(v["pass"] for v in ver) and not failed
        metrics = dict(
            method=METHOD,
            method_full="Repeat Holding: copy nearest visible slice at nearest xy spot",
            status="DONE" if all_pass else "FAILED",
            lane=C.LANE, host=__import__("os").uname().nodename,
            slurm_job=__import__("os").environ.get("SLURM_JOB_ID"),
            pilot=args.pilot,
            n_datasets=len(dss), datasets=dss,
            n_done=len(per), n_failed=len(failed),
            coverage=round(len(per) / len(dss), 4) if dss else 0.0,
            total_target_rows=int(sum(p["n_target"] for p in per.values())),
            compute=dict(cpu_cores=int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", 1)),
                         gpu="none (CPU-only)", gpu_peak_mb=0,
                         ram_peak_mb=round(C.ram_peak_mb(), 1),
                         total_sec=round(total, 2)),
            per_dataset=per, failures=failed, finished_at=C.now(),
        )
        C.write_json(outdir / ("compute_metrics_pilot.json" if args.pilot else "compute_metrics.json"),
                     metrics)
        status.update(status=metrics["status"], stage="done", n_datasets=len(dss),
                      n_done=len(per), n_failed=len(failed),
                      coverage=metrics["coverage"], total_sec=round(total, 2),
                      total_target_rows=metrics["total_target_rows"])
        C.set_method(METHOD, metrics["status"], datasets_total=len(dss),
                     datasets_done=len(per), coverage=metrics["coverage"],
                     n_failed=len(failed), total_sec=round(total, 2),
                     total_target_rows=metrics["total_target_rows"],
                     gpu="none (CPU-only)", metrics=str(outdir / "compute_metrics.json"),
                     failures=list(failed))
        log(f"[done] {len(per)}/{len(dss)} datasets rows={metrics['total_target_rows']} "
            f"total={total:.1f}s all_pass={all_pass}")

    C.run_guard(METHOD, status, log, body)


if __name__ == "__main__":
    main()

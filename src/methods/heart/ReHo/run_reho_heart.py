#!/usr/bin/env python
"""ReHo (Repeat Holding) for the heart 10-slice target-only benchmark.

CPU-only sanity baseline, adapted from MERFISH ReHo/run_reho.py:
  * source slice for a hidden section = the visible slice whose z is nearest
    (ties -> the preceding/lower visible slice, matching ReHo's "copy
    preceding" spirit);
  * within that visible slice copy the expression of the bin at the same
    (ix, iy); if the lattice bin is absent there, copy the Euclidean-nearest
    bin in x/y of the same visible slice.

Only visible-section measured expression is copied; target expression is
never read, and no interpolation/blending/model is applied (observed-like
values by construction).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _rb_common as C  # noqa: E402

METHOD = "ReHo"


def predict_scenario(sc, bins, tg, log):
    """Return (rows, pred, meta) for one scenario."""
    rows = C.scenario_rows(tg, sc)
    sec_rows = {s: np.flatnonzero(bins["section"] == s) for s in C.SECTIONS}
    z_of = {s: float(bins["z"][sec_rows[s]].mean()) for s in C.SECTIONS}
    vis = [s for s in sc["visible"] if s in sec_rows]
    meta = {"target_sections": {}, "source_rule": "nearest visible z; tie -> lower z"}
    pred = np.empty((rows.size, C.N_GENES), dtype=np.float32)
    tgt_section = tg["section"][rows]
    tgt_bin = tg["bin_id"][rows]
    tgt_ix = np.empty(rows.size, dtype=np.int64)
    tgt_iy = np.empty(rows.size, dtype=np.int64)
    for s in sc["target"]:
        b = sec_rows[s]
        m = bins["bin_id"][b]
        pos = {bid: i for i, bid in enumerate(m)}
        want = np.flatnonzero(tgt_section == s)
        ids = tgt_bin[want]
        miss = [x for x in ids if x not in pos]
        if miss:
            raise RuntimeError(f"target bins missing from bins_all: {miss[:3]}")
        idx = np.array([pos[x] for x in ids], dtype=np.int64)
        tgt_ix[want] = bins["ix"][b][idx]
        tgt_iy[want] = bins["iy"][b][idx]

    for s in sc["target"]:
        want = np.flatnonzero(tgt_section == s)
        zq = z_of[s]
        order = sorted(vis, key=lambda v: (abs(z_of[v] - zq), z_of[v]))
        src = order[0]
        sb = sec_rows[src]
        lookup = {(int(i), int(j)): k for k, (i, j) in
                  enumerate(zip(bins["ix"][sb], bins["iy"][sb]))}
        xy_src = np.stack([bins["x"][sb], bins["y"][sb]], 1)
        tree = cKDTree(xy_src)
        nx = np.array([lookup.get((int(i), int(j)), -1)
                       for i, j in zip(tgt_ix[want], tgt_iy[want])])
        hit = nx >= 0
        pred[want[hit]] = bins["X"][sb][nx[hit]]
        n_miss = int((~hit).sum())
        if n_miss:
            tsel = sec_rows[s]
            bid_t = {bid: k for k, bid in enumerate(bins["bin_id"][tsel])}
            miss_rows = want[~hit]
            qxy = np.stack([bins["x"][tsel][[bid_t[x] for x in tgt_bin[miss_rows]]],
                            bins["y"][tsel][[bid_t[x] for x in tgt_bin[miss_rows]]]], 1)
            dist, nn = tree.query(qxy, k=1, workers=4)
            pred[miss_rows] = bins["X"][sb][nn]
            meta["target_sections"][s] = dict(
                src=src, n=len(want), same_bin=int(hit.sum()), knn_xy=n_miss,
                z_gap_um=float(abs(z_of[src] - zq)),
                xy_dist_um=dict(mean=float(dist.mean()), max=float(dist.max())))
        else:
            meta["target_sections"][s] = dict(
                src=src, n=len(want), same_bin=int(hit.sum()), knn_xy=0,
                z_gap_um=float(abs(z_of[src] - zq)))
    return rows, pred, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    outdir = C.METHODS_ROOT / METHOD
    outdir.mkdir(parents=True, exist_ok=True)
    log = C.RunLog(outdir / "run.log")
    status = C.Status(outdir / "status.json", METHOD)
    status.update(status="RUNNING", stage="load", pilot=args.pilot,
                  command=" ".join([sys.executable] + sys.argv))

    def body():
        t0 = time.time()
        bins = C.load_bins()
        tg = C.load_target_gt()
        all_sc = C.load_scenarios()
        scs = C.select_scenarios(all_sc, "k1_s25" if args.pilot else args.scenarios)
        log(f"[data] bins={bins['X'].shape} sections={sorted(set(bins['section'].tolist()))}")
        status.update(stage="predict", n_scenarios=len(scs))
        per, ver = {}, []
        for i, sc in enumerate(scs):
            ts = time.time()
            rows, pred, meta = predict_scenario(sc, bins, tg, log)
            p = C.save_pred(METHOD, sc, tg, rows, pred)
            v = C.verify_pred(p, tg, rows, sc, METHOD)
            ver.append(v)
            per[sc["name"]] = dict(n_rows=int(rows.size),
                                   sec=round(time.time() - ts, 3),
                                   target=sc["target"],
                                   n_target_bins={k: x["n"] for k, x in meta["target_sections"].items()},
                                   sources={k: x["src"] for k, x in meta["target_sections"].items()},
                                   same_bin={k: x["same_bin"] for k, x in meta["target_sections"].items()},
                                   knn_xy={k: x["knn_xy"] for k, x in meta["target_sections"].items()})
            log(f"[pred] {sc['name']} rows={rows.size} target={sc['target']} "
                f"src={ {k: x['src'] for k, x in meta['target_sections'].items()} } "
                f"{time.time() - ts:.2f}s")
            status.update(stage="predict", scenarios_done=i + 1,
                          last_scenario=sc["name"], scenarios=list(per))
        total = time.time() - t0
        all_pass = all(v["pass"] for v in ver)
        metrics = dict(
            method=METHOD,
            method_full="Repeat Holding: copy nearest visible slice at same (ix,iy) bin, "
                        "xy-nearest fallback",
            status="DONE" if all_pass else "FAILED_VERIFY",
            lane=C.LANE, host=__import__("os").uname().nodename,
            slurm_job=__import__("os").environ.get("SLURM_JOB_ID"),
            n_scenarios=len(scs), scenarios=[s["name"] for s in scs],
            n_target_rows=int(sum(p["n_rows"] for p in per.values())),
            pilot=args.pilot,
            compute=dict(cpu_cores=int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", 1)),
                         gpu="none (CPU-only)", gpu_peak_mb=0,
                         ram_peak_mb=round(C.ram_peak_mb(), 1),
                         total_sec=round(total, 2)),
            per_scenario=per, verification=ver, finished_at=C.now(),
        )
        C.write_json(outdir / ("compute_metrics_pilot.json" if args.pilot else "compute_metrics.json"),
                     metrics)
        status.update(status="DONE" if all_pass else "FAILED_VERIFY", stage="done",
                      n_scenarios=len(scs), total_sec=round(total, 2),
                      n_target_rows=metrics["n_target_rows"], verification_all_pass=all_pass)
        C.update_runner_status(METHOD, dict(
            status="DONE" if all_pass else "FAILED_VERIFY",
            n_scenarios=len(scs), n_target_rows=metrics["n_target_rows"],
            total_sec=round(total, 2), gpu="none (CPU-only)",
            metrics=str(outdir / "compute_metrics.json")))
        log(f"[done] {len(scs)} scenarios rows={metrics['n_target_rows']} total={total:.1f}s "
            f"all_pass={all_pass}")

    C.run_guard(METHOD, status, log, body)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""UniST (SUICA GAE + INR) on the MERFISH 5-slice (.006-.010) target-only benchmark.

Per scenario: run the official SUICA 3D-sparse pipeline (GAE embedder + INR +
decoder, code unmodified) on the VISIBLE sections only, and predict at the
hidden target bin coordinates only. Output = target rows in target_gt order.

Note: SUICA internally fits a 3-component PCA on the expression purely for its
diagnostic plot (datasets.py raw_pca); the model input is the full 238 genes and
the prediction is the decoder reconstruction. No 50PC/PCA features are used.

Usage (inside the granted srun step):
  python run_heart.py --pilot [--scenario <name>]
  python run_heart.py
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import anndata as ad
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

UPSTREAM = Path(os.environ["MERFISH5_UPSTREAM"])
SUICA_DIR = UPSTREAM / "UniST" / "external" / "SUICA_pro"
sys.path.insert(0, str(HERE))
from pipeline_script import PIPELINE_SCRIPT  # noqa: E402

PY = os.environ.get("UNIST_PY", sys.executable)

METHOD = "UniST"
OUT = C.METHODS_ROOT / METHOD
LOG = OUT / "run.log"

PRINT_PATCH = '''
import time as _tt, builtins as _bi
_T0 = _tt.time()
_op = _bi.print
def _print(*a, **k):
    _op("[t=%9.1fs]" % (_tt.time() - _T0), *a, **k)
_bi.print = _print
'''


def log(msg):
    line = f"[{C.now()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def build_inputs(bins, tg, rows, sc, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    vis_rows = np.flatnonzero(np.isin(bins["section"], np.asarray(sc["visible"], dtype=object)))
    zsec = C.section_z(bins["z"], bins["section"])
    vis = ad.AnnData(
        X=bins["X"][vis_rows].copy(),
        obs=pd.DataFrame({"section": bins["section"][vis_rows].astype(str),
                          "z_um": bins["z"][vis_rows]},
                         index=[f"v{i}" for i in range(vis_rows.size)]),
        var=pd.DataFrame({"gene_name": bins["genes"]}, index=bins["genes"]),
    )
    vis.obsm["spatial"] = np.stack([bins["x"][vis_rows], bins["y"][vis_rows],
                                    bins["z"][vis_rows]], axis=1) / 1000.0
    vis_path = run_dir / "visible_data.h5ad"
    vis.write_h5ad(vis_path)

    q = np.stack([tg["x"][rows], tg["y"][rows],
                  np.array([zsec[s] for s in tg["section"][rows]], dtype=np.float64)],
                 axis=1) / 1000.0
    q_path = run_dir / "query_coords.npy"
    np.save(q_path, q.astype(np.float64))
    return vis_path, q_path, int(vis_rows.size), int(rows.size)


def run_scenario(bins, tg, rows, sc, args):
    run_dir = OUT / "run" / sc["name"]
    vis_path, q_path, n_vis, n_q = build_inputs(bins, tg, rows, sc, run_dir)
    script = PRINT_PATCH + PIPELINE_SCRIPT.format(
        SUICA_DIR=SUICA_DIR, visible_data=vis_path, query_coords_file=str(q_path),
        output_dir=run_dir, gae_epochs=args.gae_epochs, inr_epochs=args.inr_epochs)
    script_path = run_dir / "_run_suica.py"
    script_path.write_text(script)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu).split(",")[0],
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    t0 = time.time()
    proc = subprocess.run([PY, str(script_path)], cwd=str(SUICA_DIR), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sec = time.time() - t0
    (run_dir / "pipeline.log").write_text(proc.stdout or "")
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-40:])
        raise RuntimeError(f"SUICA pipeline rc={proc.returncode}\n{tail}")
    X = np.load(run_dir / "X_pred.npy").astype(np.float32)
    if X.shape != (n_q, bins["genes"].size):
        raise RuntimeError(f"X_pred {X.shape} != ({n_q},{bins['genes'].size})")
    if not np.isfinite(X).all():
        raise RuntimeError(f"{int((~np.isfinite(X)).sum())} non-finite UniST predictions")
    np.clip(X, 0.0, None, out=X)
    stage = {}
    for line in (proc.stdout or "").splitlines():
        if "[t=" in line and ("Stage 1" in line or "Stage 2" in line or "Stage 3" in line
                              or "DONE!" in line):
            stage.setdefault(line.split("]", 1)[1].strip()[:40], float(line.split("[t=")[1].split("s]")[0]))
    return X, dict(pipeline_sec=round(sec, 1), n_visible_bins=n_vis, n_query=n_q,
                   stage_marks=stage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--scenarios", default=None)
    ap.add_argument("--gae-epochs", type=int, default=300)
    ap.add_argument("--inr-epochs", type=int, default=400)
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).split(",")[0]
    if args.pilot:
        args.gae_epochs = min(args.gae_epochs, 2)
        args.inr_epochs = min(args.inr_epochs, 2)

    status = C.Status(OUT / "status.json", METHOD,
                      dict(pilot=bool(args.pilot), config=vars(args), log=str(LOG)))
    mon = C.GpuSampler(); mon.start()
    log(f"START pilot={args.pilot} gae={args.gae_epochs} inr={args.inr_epochs}")

    C.wait_ready()
    bins = C.load_bins()
    scs = C.load_scenarios()
    tg = C.load_target_gt()
    log(f"loaded bins X={bins['X'].shape} scenarios={len(scs)}")
    if args.scenario:
        scs = [s for s in scs if s["name"] == args.scenario]
    if args.scenarios:
        keep = set(args.scenarios.split(","))
        scs = [s for s in scs if s["name"] in keep]
    if args.pilot:
        scs = scs[:1]
    status.update(status="RUNNING", stage="loaded", scenarios_total=len(scs))

    metrics = {"method": METHOD, "dataset": "merfish_5slice_50um", "voxel_um": 50,
               "pilot": bool(args.pilot), "config": vars(args),
               "genes": "full 1122 input genes (no PC reduction for the model)",
               "scenarios": []}
    failures = []
    for i, sc in enumerate(scs):
        t0 = time.time()
        status.update(stage=f"suica:{sc['name']}", current=sc["name"], scenario_index=i)
        try:
            rows = C.scenario_rows(tg, sc)
            X, info = run_scenario(bins, tg, rows, sc, args)
            path = C.save_pred(METHOD, sc, tg, rows, X, extra=dict(genes=bins["genes"]))
            rec = dict(scenario=sc["name"], k=len(sc["target"]), visible=sc["visible"],
                       target=sc["target"], n_rows=int(rows.size),
                       total_sec=round(time.time() - t0, 1),
                       pred_min=float(X.min()), pred_max=float(X.max()),
                       pred_mean=float(X.mean()), pred_zero_frac=float((X == 0).mean()),
                       output=str(path), **info)
            metrics["scenarios"].append(rec)
            log(f"[{sc['name']}] DONE rows={rows.size} {rec['total_sec']}s "
                f"pipeline={info['pipeline_sec']}s mean={rec['pred_mean']:.4f}")
            del X
        except Exception as e:
            tb = traceback.format_exc()
            log(f"[{sc['name']}] FAILED {e!r}\n{tb}")
            failures.append(dict(scenario=sc["name"], error=repr(e), traceback=tb[-2500:]))
            status.update(stage=f"failed:{sc['name']}", last_error=repr(e), failures=failures)
        status.update(scenarios_done=i + 1, gpu_peak_mb=int(mon.peak), failures=len(failures))

    mon.stop.set(); mon.join(timeout=5)
    metrics.update(gpu_peak_mb=int(mon.peak), n_done=len(metrics["scenarios"]),
                   n_failed=len(failures), failures=failures)
    C.write_json(OUT / "compute_metrics.json", metrics)
    ok = not failures
    status.update(status="DONE" if ok else "PARTIAL", stage="finished",
                  gpu_peak_mb=int(mon.peak), n_done=len(metrics["scenarios"]),
                  n_failed=len(failures), metrics=str(OUT / "compute_metrics.json"))
    log(f"FINISHED done={metrics['n_done']} failed={len(failures)} gpu_peak={mon.peak}MiB")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

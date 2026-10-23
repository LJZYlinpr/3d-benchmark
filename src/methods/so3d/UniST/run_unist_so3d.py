#!/usr/bin/env python3
"""UniST (SUICA GAE embedder + INR + decoder) on the So3D 46 held-out-middle-slice set.

Target-only: the official SUICA 3D-sparse pipeline is fit on canonical visible
spots only; predictions are produced at the held-out target spot coordinates and
saved as predictions/<dataset>/UniST.npz. No full volume is written, and
target.npz['X'] (GT) is never read.

Adapted from Workspace/merfish_test/UniST/run_unist_50um.py and the heart
target-only adapter (methods/UniST/run_heart.py): the official PIPELINE_SCRIPT
is reused verbatim.

Usage (inside the granted srun step):
  python run_unist_so3d.py --pilot
  python run_unist_so3d.py --datasets A,B
  python run_unist_so3d.py            # all datasets, skip DONE
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANON = ROOT / "canonical"
READY = ROOT / "PREP_READY"
PRED_ROOT = Path(os.environ.get("SO3D_PRED_ROOT", str(ROOT / "predictions")))
STATUS_ROOT = Path(os.environ.get("SO3D_RUNNER_ROOT", str(ROOT)))
LOG = HERE / "run.log"

BENCH = Path("/storage/yuhongtaoLab/linpeiran/xpy/lab/baseline")
sys.path.insert(0, str(BENCH))
from benchmark.scripts.rerun_unist_official import PIPELINE_SCRIPT  # noqa: E402

SUICA_DIR = BENCH / "methods/UniST/external/SUICA_pro"
PY = "/storage/yuhongtaoLab/linpeiran/env/miniconda/envs/unist/bin/python"
METHOD = "UniST"

PRINT_PATCH = '''
import time as _tt, builtins as _bi
_T0 = _tt.time()
_op = _bi.print
def _print(*a, **k):
    _op("[t=%9.1fs]" % (_tt.time() - _T0), *a, **k)
_bi.print = _print
'''


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path, d):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(d, indent=2, default=jdefault))
    tmp.replace(path)


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


class StatusWriter:
    """Atomic heartbeat status.json (re-created so partial progress is visible)."""

    def __init__(self, path, method, extra=None):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.d = {"method": method, "status": "RUNNING", "stage": "start",
                  "host": os.uname().nodename,
                  "slurm_job": os.environ.get("SLURM_JOB_ID"),
                  "pid": os.getpid(), "started_at": now(), "updated_at": now(),
                  "elapsed_sec": 0.0, "gpu_peak_mb": 0}
        if extra:
            self.d.update(extra)
        self.update()

    def update(self, **kw):
        with self.lock:
            self.d.update(kw)
            self.d["elapsed_sec"] = round(time.time() - self.t0, 1)
            self.d["updated_at"] = now()
            write_json(self.path, self.d)


class GpuSampler(threading.Thread):
    def __init__(self, period=5.0):
        super().__init__(daemon=True)
        self.period = period
        self.peak = 0
        self.idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--id={self.idx}", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                if out:
                    self.peak = max(self.peak, int(out.splitlines()[0]))
            except Exception:
                pass
            self.stop.wait(self.period)


def wait_ready(timeout_sec=4 * 3600, poll=20):
    t0 = time.time()
    while not READY.exists():
        blocked = ROOT / "BLOCKED"
        if blocked.exists():
            raise RuntimeError(f"prep BLOCKED marker present: {blocked.read_text()[:800]}")
        if time.time() - t0 > timeout_sec:
            raise RuntimeError(f"PREP_READY not found after {timeout_sec}s")
        log(f"waiting for {READY} ({int(time.time() - t0)}s)")
        time.sleep(poll)


def list_datasets():
    out = []
    for d in sorted(CANON.iterdir()):
        if d.is_dir() and (d / "meta.json").exists() and (d / "visible.h5ad").exists() \
                and (d / "target.npz").exists():
            out.append(d.name)
    return out


def load_target(meta):
    """Metadata-only view of target.npz; GT ('X') is deliberately never accessed."""
    p = CANON / meta["dataset"] / "target.npz"
    z = np.load(p, allow_pickle=False)
    tgt = {}
    for k in ("coords", "obs_names", "slice", "z"):
        if k not in z.files:
            raise RuntimeError(f"target.npz missing key {k!r}; keys={z.files}")
        tgt[k] = z[k]
    tgt["coords"] = np.asarray(tgt["coords"], dtype=np.float64)
    tgt["obs_names"] = np.asarray(tgt["obs_names"]).astype(str)
    tgt["slice"] = np.asarray(tgt["slice"]).astype(np.int64)
    tgt["z"] = np.asarray(tgt["z"], dtype=np.float64)
    n = meta["n_target"]
    for k in ("coords", "obs_names", "slice", "z"):
        if tgt[k].shape[0] != n:
            raise RuntimeError(f"target.{k} has {tgt[k].shape[0]} rows, meta.n_target={n}")
    if not np.isfinite(tgt["coords"]).all():
        raise RuntimeError("target coords non-finite")
    return tgt


def build_inputs(meta, tgt, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    vis = ad.read_h5ad(CANON / meta["dataset"] / "visible.h5ad")
    genes = [str(g) for g in vis.var_names]
    if genes != [str(g) for g in meta["genes"]]:
        raise RuntimeError("visible var_names != meta.json genes")
    coords = np.asarray(vis.obsm["spatial"])
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise RuntimeError(f"visible obsm['spatial'] shape {coords.shape} (need Nx3)")
    coords = coords[:, :3].astype(np.float64)
    if not np.isfinite(coords).all():
        raise RuntimeError("visible coords non-finite")
    rng = coords.max(axis=0) - coords.min(axis=0)
    if np.any(rng <= 0):
        raise Unsupported(f"degenerate visible coordinate range {rng.tolist()} "
                          f"(SUICA 3D normalization divides by each axis range)")

    sl = vis.obs["slice"].to_numpy().astype(np.int64)
    if set(np.unique(sl)) != set(int(s) for s in meta["visible_slices"]):
        raise RuntimeError("visible slice values != meta.visible_slices")

    obs = pd.DataFrame({"section": sl.astype(str),
                        "z_um": (vis.obs["z"].to_numpy().astype(np.float64) * 1000.0)},
                       index=[str(x) for x in vis.obs_names])
    v = ad.AnnData(X=vis.X, obs=obs,
                   var=pd.DataFrame({"gene_name": genes}, index=genes),
                   obsm={"spatial": coords})
    v.uns["coords_unit"] = "mm"
    vis_path = run_dir / "visible_data.h5ad"
    v.write_h5ad(vis_path)

    q = np.asarray(tgt["coords"], dtype=np.float64)
    q_path = run_dir / "query_coords.npy"
    np.save(q_path, q)
    return vis_path, q_path, v.n_obs, int(q.shape[0]), genes, coords


class Unsupported(Exception):
    pass


def run_suica(meta, tgt, args, status):
    ds = meta["dataset"]
    run_dir = HERE / "run" / ds
    vis_path, q_path, n_vis, n_q, genes, vcoords = build_inputs(meta, tgt, run_dir)

    script = PRINT_PATCH + PIPELINE_SCRIPT.format(
        SUICA_DIR=SUICA_DIR, visible_data=vis_path, query_coords_file=str(q_path),
        output_dir=run_dir, gae_epochs=args.gae_epochs, inr_epochs=args.inr_epochs)
    script_path = run_dir / "_run_suica.py"
    script_path.write_text(script)

    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads))
    t0 = time.time()
    try:
        proc = subprocess.run([PY, str(script_path)], cwd=str(SUICA_DIR), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              timeout=args.timeout)
    except subprocess.TimeoutExpired as e:
        (run_dir / "pipeline.log").write_text(str(e.stdout or "")[-2_000_000:])
        raise RuntimeError(f"SUICA pipeline timeout after {args.timeout}s")
    sec = time.time() - t0
    out = proc.stdout or ""
    (run_dir / "pipeline.log").write_text(out)
    if proc.returncode != 0:
        tail = "\n".join(out.splitlines()[-40:])
        raise RuntimeError(f"SUICA pipeline rc={proc.returncode}\n{tail}")

    xp = run_dir / "X_pred.npy"
    if not xp.exists():
        raise RuntimeError("pipeline finished but X_pred.npy is missing")
    X = np.load(xp).astype(np.float32)
    if X.shape != (n_q, len(genes)):
        raise RuntimeError(f"X_pred {X.shape} != ({n_q},{len(genes)})")
    nbad = int((~np.isfinite(X)).sum())
    if nbad:
        raise RuntimeError(f"{nbad} non-finite UniST predictions")
    np.clip(X, 0.0, None, out=X)

    stage = {}
    for line in out.splitlines():
        if "[t=" in line and any(s in line for s in ("Stage 1", "Stage 2", "Stage 3", "DONE!")):
            stage.setdefault(line.split("]", 1)[1].strip()[:40],
                             float(line.split("[t=")[1].split("s]")[0]))

    if not args.keep_intermediates:
        for sub in ("logs_gae", "logs_inr", "logs_pred"):
            shutil.rmtree(run_dir / sub, ignore_errors=True)
    return X, genes, dict(pipeline_sec=round(sec, 1), n_visible_spots=n_vis,
                          n_query=n_q, stage_marks=stage)


def save_pred(meta, tgt, X, genes, extra):
    ds = meta["dataset"]
    outdir = PRED_ROOT / ds
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{METHOD}.npz"
    tmp = outdir / f".{METHOD}.npz.tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh, pred=X.astype(np.float32), obs_names=tgt["obs_names"],
            coords=tgt["coords"].astype(np.float32), slice=tgt["slice"].astype(np.int64),
            genes=np.asarray(genes), dataset=np.asarray(ds), method=np.asarray(METHOD),
            row_index=np.arange(tgt["obs_names"].size, dtype=np.int64), **extra)
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--datasets", default=None, help="comma separated dataset names")
    ap.add_argument("--max-datasets", type=int, default=None)
    ap.add_argument("--gae-epochs", type=int, default=300)
    ap.add_argument("--inr-epochs", type=int, default=600)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=5400, help="per-dataset subprocess timeout (s)")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--keep-intermediates", action="store_true")
    ap.add_argument("--skip-ready", action="store_true",
                    help="pilot only: do not block on canonical/PREP_READY")
    args = ap.parse_args()

    if args.pilot:
        args.gae_epochs = min(args.gae_epochs, 2)
        args.inr_epochs = min(args.inr_epochs, 2)

    HERE.mkdir(parents=True, exist_ok=True)
    PRED_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"START pilot={args.pilot} gae={args.gae_epochs} inr={args.inr_epochs} "
        f"cuda={os.environ.get('CUDA_VISIBLE_DEVICES')} job={os.environ.get('SLURM_JOB_ID')}")

    if args.skip_ready:
        log("--skip-ready: bypassing PREP_READY wait (pilot only)")
        meta_ready = "(skipped)"
    else:
        wait_ready()
        meta_ready = READY.read_text()
    datasets = list_datasets()
    if args.datasets:
        keep = [x.strip() for x in args.datasets.split(",") if x.strip()]
        missing = [x for x in keep if x not in datasets]
        if missing:
            raise SystemExit(f"requested datasets not in canonical: {missing}")
        datasets = keep
    if args.pilot:
        datasets = datasets[:1]
    if args.max_datasets:
        datasets = datasets[:args.max_datasets]
    log(f"datasets={len(datasets)} prep_ready={meta_ready.strip().replace(chr(10), '; ')}")

    status = StatusWriter(HERE / "status.json", METHOD,
                          dict(pilot=bool(args.pilot), config=vars(args), log=str(LOG),
                               datasets_total=len(datasets), datasets_done=0))
    mon = GpuSampler()
    mon.start()

    recs, failures, unsupported = [], [], []
    t_all = time.time()
    for i, ds in enumerate(datasets):
        pred_path = PRED_ROOT / ds / f"{METHOD}.npz"
        st_path = PRED_ROOT / ds / f"{METHOD}.status.json"
        if args.skip_existing and pred_path.exists():
            try:
                old = json.loads(st_path.read_text())
            except Exception:
                old = {}
            if old.get("status") == "DONE":
                recs.append(dict(dataset=ds, status="DONE", skipped=True,
                                 seconds=0.0, output=str(pred_path)))
                log(f"[{ds}] SKIP already DONE")
                status.update(datasets_done=len(recs), current=ds, dataset_index=i)
                continue

        meta = json.loads((CANON / ds / "meta.json").read_text())
        t0 = time.time()
        status.update(stage=f"unist:{ds}", current=ds, dataset_index=i)
        try:
            tgt = load_target(meta)
            X, genes, info = run_suica(meta, tgt, args, status)
            path = save_pred(meta, tgt, X, genes,
                             dict(z=tgt["z"].astype(np.float32),
                                  target_slice=np.int64(meta["target_slice"])))
            sec = time.time() - t0
            rec = dict(dataset=ds, status="DONE", n_slices=meta["n_slices"],
                       target_slice=meta["target_slice"],
                       visible_slices=meta["visible_slices"],
                       n_visible=info["n_visible_spots"], n_target=info["n_query"],
                       n_genes=len(genes), seconds=round(sec, 1),
                       pipeline_sec=info["pipeline_sec"], stage_marks=info["stage_marks"],
                       pred_min=float(X.min()), pred_max=float(X.max()),
                       pred_mean=float(X.mean()), pred_zero_frac=float((X == 0).mean()),
                       gpu_peak_mb=int(mon.peak), output=str(path))
            recs.append(rec)
            write_json(st_path, {"status": "DONE", "reason": "ok", "method": METHOD,
                                 "dataset": ds, "updated_utc": now(),
                                 "rows": info["n_query"], "genes": len(genes),
                                 "seconds": round(sec, 1)})
            log(f"[{ds}] DONE rows={info['n_query']} genes={len(genes)} {sec:.1f}s "
                f"pipeline={info['pipeline_sec']}s mean={rec['pred_mean']:.4f} "
                f"zero={rec['pred_zero_frac']:.3f} gpu_peak={mon.peak}MiB")
            del X
        except Unsupported as e:
            sec = time.time() - t0
            rec = dict(dataset=ds, status="UNSUPPORTED", reason=str(e), seconds=round(sec, 1))
            recs.append(rec)
            unsupported.append(rec)
            write_json(st_path, {"status": "UNSUPPORTED", "reason": str(e),
                                 "method": METHOD, "dataset": ds, "updated_utc": now()})
            log(f"[{ds}] UNSUPPORTED {e}")
        except Exception as e:
            sec = time.time() - t0
            tb = traceback.format_exc()
            rec = dict(dataset=ds, status="FAILED", reason=repr(e), seconds=round(sec, 1),
                       traceback=tb[-3000:])
            recs.append(rec)
            failures.append(rec)
            write_json(st_path, {"status": "FAILED", "reason": repr(e), "method": METHOD,
                                 "dataset": ds, "updated_utc": now(),
                                 "traceback": tb[-3000:]})
            log(f"[{ds}] FAILED {e!r}\n{tb}")
        status.update(datasets_done=i + 1, gpu_peak_mb=int(mon.peak),
                      n_done=sum(r["status"] == "DONE" for r in recs),
                      n_failed=len(failures), n_unsupported=len(unsupported))

    mon.stop.set()
    mon.join(timeout=5)
    wall = round(time.time() - t_all, 1)
    n_done = sum(r["status"] == "DONE" for r in recs)
    n_fail = sum(r["status"] == "FAILED" for r in recs)
    n_unsup = sum(r["status"] == "UNSUPPORTED" for r in recs)
    metrics = {
        "method": METHOD, "benchmark": "so3d_46_heldout_middle_slice",
        "pilot": bool(args.pilot), "config": vars(args),
        "pipeline": "official SUICA_pro: GAE embedder (full genes) + INR + decoder",
        "genes": "all dataset genes in var order (no PC reduction for the model)",
        "epochs_note": (f"GAE={args.gae_epochs} INR={args.inr_epochs}; official defaults are "
                        "1000/2000, reduced here as a documented throughput deviation"),
        "target_only": True, "gt_read": False,
        "datasets_total": len(datasets), "n_done": n_done, "n_failed": n_fail,
        "n_unsupported": n_unsup, "coverage": round(n_done / max(len(datasets), 1), 4),
        "wall_sec": wall, "gpu_peak_mb": int(mon.peak),
        "per_dataset": recs, "failures": failures, "unsupported": unsupported,
    }
    write_json(HERE / "compute_metrics.json", metrics)

    runner = {
        "runner": "D", "method": METHOD, "job": os.environ.get("SLURM_JOB_ID"),
        "host": os.uname().nodename, "updated_utc": now(),
        "status": "DONE" if n_fail == 0 and n_unsup == 0 and n_done + n_unsup >= len(datasets)
        else ("PARTIAL" if n_done > 0 else "FAILED"),
        "datasets_total": len(datasets), "n_done": n_done, "n_failed": n_fail,
        "n_unsupported": n_unsup, "coverage_of_datasets": round(n_done / max(len(datasets), 1), 4),
        "wall_sec": wall, "gpu_peak_mb": int(mon.peak),
        "epochs": {"gae": args.gae_epochs, "inr": args.inr_epochs},
        "predictions_root": str(PRED_ROOT),
        "metrics": str(HERE / "compute_metrics.json"),
        "failures": [dict(dataset=r["dataset"], reason=r.get("reason")) for r in failures],
        "unsupported": [dict(dataset=r["dataset"], reason=r.get("reason")) for r in unsupported],
        "per_dataset": [dict(dataset=r["dataset"], status=r["status"], seconds=r.get("seconds"),
                             n_target=r.get("n_target"), n_genes=r.get("n_genes"),
                             reason=r.get("reason")) for r in recs],
    }
    write_json(STATUS_ROOT / "RUNNER_D_STATUS.json", runner)
    status.update(status=runner["status"], stage="finished", datasets_done=len(recs),
                  n_done=n_done, n_failed=n_fail, n_unsupported=n_unsup,
                  gpu_peak_mb=int(mon.peak), wall_sec=wall)
    log(f"FINISHED status={runner['status']} done={n_done} unsupported={n_unsup} "
        f"failed={n_fail} wall={wall}s gpu_peak={mon.peak}MiB")
    return 0 if runner["status"] == "DONE" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        log("FATAL\n" + traceback.format_exc())
        sys.exit(1)

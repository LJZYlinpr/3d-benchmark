#!/usr/bin/env python
"""Shared helpers for runner B (PASTE-OT-Linear / ReHo / Ours-EncMTP).

Reads only canonical geometry + visible-section expression.  The prediction
contract is predictions/<scenario>/<method>.npz with `pred` (target rows only,
target_gt order), `row_index`, and echoed bin_id/section/x/y.
"""
import json
import os
import subprocess
import threading
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path("/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/Workspace/merfish_test/missing_combos_50um/all_methods")
CANON = ROOT / "canonical"
PRED_ROOT = ROOT / "predictions"
METHODS_ROOT = ROOT / "methods"
N_GENES = 1122
SECTIONS = ["1", "2", "3", "4", "5"]
LANE = {"job": "9383994", "node": "gvna02"}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=jd, ensure_ascii=False))
    os.replace(tmp, path)


class RunLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg):
        line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg)
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


class Status:
    def __init__(self, path, method):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.d = {"method": method, "status": "RUNNING", "stage": "start",
                  "host": os.uname().nodename, "pid": os.getpid(),
                  "lane": LANE, "slurm_job": os.environ.get("SLURM_JOB_ID"),
                  "started_at": now(), "n_scenarios_total": 7,
                  "scenarios_done": 0, "gpu_peak_mb": 0}
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
        self.peak_mb = 0
        self.peak_util = 0
        self.stop_evt = threading.Event()

    def run(self):
        while not self.stop_evt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                for line in out.splitlines():
                    if not line.strip():
                        continue
                    mem, util = [x.strip() for x in line.split(",")]
                    self.peak_mb = max(self.peak_mb, int(float(mem)))
                    self.peak_util = max(self.peak_util, int(float(util)))
            except Exception:
                pass
            self.stop_evt.wait(self.period)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=15)


def ram_peak_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def load_bins():
    import anndata as ad
    a = ad.read_h5ad(CANON / "bins_all.h5ad")
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    X = np.ascontiguousarray(X, dtype=np.float32)
    obs = a.obs
    section = np.asarray(obs["section"]).astype(str)
    return dict(
        X=X, genes=np.asarray(a.var_names).astype(str), section=section,
        bin_id=np.asarray(a.obs_names).astype(str),
        x=np.asarray(obs["x"], dtype=np.float64),
        y=np.asarray(obs["y"], dtype=np.float64),
        z=np.asarray(obs["z"], dtype=np.float64),
        ix=np.asarray(obs["ix"], dtype=np.int64),
        iy=np.asarray(obs["iy"], dtype=np.int64),
        n_cells=np.asarray(obs["n_cells"], dtype=np.int64),
    )


def load_scenarios():
    d = json.loads((CANON / "scenarios.json").read_text())
    return [{**s, "visible": [str(v) for v in s["visible"]],
             "target": [str(v) for v in s["target"]]} for s in d["scenarios"]]


def load_target_gt(keys=("bin_id", "section", "x", "y", "scenario", "row_index")):
    z = np.load(CANON / "target_gt.npz", allow_pickle=False)
    out = {}
    for k in keys:
        if k in z.files:
            out[k] = z[k]
    z.close()
    out["bin_id"] = np.asarray(out["bin_id"]).astype(str)
    out["section"] = np.asarray(out["section"]).astype(str)
    out["x"] = np.asarray(out["x"], dtype=np.float64)
    out["y"] = np.asarray(out["y"], dtype=np.float64)
    out["scenario"] = np.asarray(out["scenario"]).astype(str)
    return out


def scenario_rows(tg, sc):
    m = tg["scenario"] == sc["name"]
    rows = np.flatnonzero(m)
    if rows.size == 0:
        raise ValueError(f"{sc['name']}: no target_gt rows")
    return rows


def save_pred(method, sc, tg, rows, pred):
    pred = np.ascontiguousarray(pred, dtype=np.float32)
    if pred.shape != (rows.size, N_GENES):
        raise ValueError(f"{method}/{sc['name']}: pred {pred.shape} != {(rows.size, N_GENES)}")
    if not np.isfinite(pred).all():
        raise ValueError(f"{method}/{sc['name']}: {int((~np.isfinite(pred)).sum())} non-finite")
    outdir = PRED_ROOT / sc["name"]
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{method}.npz"
    tmp = outdir / f".{method}.tmp"
    payload = dict(
        pred=pred, row_index=rows.astype(np.int64),
        bin_id=tg["bin_id"][rows], section=tg["section"][rows],
        x=tg["x"][rows], y=tg["y"][rows],
        method=np.array(method), scenario=np.array(sc["name"]),
    )
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    os.replace(tmp, path)
    return path


def verify_pred(path, tg, rows, sc, method):
    z = np.load(path, allow_pickle=False)
    pred = z["pred"]
    ok = dict(
        scenario=sc["name"], method=method, path=str(path),
        shape=list(pred.shape), dtype=str(pred.dtype),
        expected_shape=[int(rows.size), N_GENES],
        row_index_equal=bool(np.array_equal(z["row_index"], rows)),
        bin_id_equal=bool(np.array_equal(np.asarray(z["bin_id"]).astype(str),
                                         tg["bin_id"][rows])),
        finite=bool(np.isfinite(pred).all()),
        nonzero_frac=float((pred != 0).mean()),
        per_gene_std_mean=float(pred.std(0).mean()),
        min=float(pred.min()), max=float(pred.max()), mean=float(pred.mean()),
    )
    ok["pass"] = bool(ok["row_index_equal"] and ok["bin_id_equal"]
                      and ok["finite"] and ok["shape"] == ok["expected_shape"])
    return ok


def select_scenarios(all_sc, spec):
    if not spec or spec == "all":
        return all_sc
    names = [x.strip() for x in str(spec).split(",") if x.strip()]
    by = {s["name"]: s for s in all_sc}
    missing = [n for n in names if n not in by]
    if missing:
        raise SystemExit(f"unknown scenarios {missing}")
    return [by[n] for n in names]


def update_runner_status(method, payload):
    p = METHODS_ROOT / "RUNNER_B_STATUS.json"
    cur = {}
    if p.exists():
        try:
            cur = json.loads(p.read_text())
        except Exception:
            cur = {}
    methods = cur.get("methods", {})
    prev = methods.get(method, {})
    prev.update(payload)
    prev["updated_at"] = now()
    methods[method] = prev
    cur["methods"] = methods
    cur["runner"] = "heart-run-b"
    cur["lane"] = LANE
    cur["methods_assigned"] = ["PASTE-OT-Linear", "ReHo", "Ours-EncMTP"]
    cur["updated_at"] = now()
    write_json(p, cur)


def run_guard(method, status, log, fn):
    try:
        fn()
    except Exception:
        tb = traceback.format_exc()
        log("[fatal]\n" + tb)
        status.update(status="FAILED", stage="error", error=tb[-4000:])
        update_runner_status(method, dict(status="FAILED", error=tb.strip().splitlines()[-1]))
        raise

#!/usr/bin/env python
"""Shared helpers for the heart 10-slice target-only benchmark runners.

Contract (heart-infra, canonical/):
  bins_all.h5ad  X (61941,238) float32 already-normalised mean expression,
                 obs: section (str), x, y, z (um), ix, iy, n_cells,
                 index bin_id = s{section}_{ix}_{iy}
  scenarios.json 26 scenarios, k=1..4, contiguous hidden blocks, both sides visible
  target_gt.npz  shared superset with a `scenario` column; row order = scenario order,
                 target section asc, (ix,iy) asc. Keys include `gt` (never read here).

Prediction file (predictions/<scenario>/<method>.npz):
  pred       (n_rows_scenario, 238) float32, aligned to that scenario's target_gt rows
  row_index  (n_rows_scenario,) int64 into target_gt.npz
  bin_id, section, x, y   echoed from target_gt for verification

This module never loads `gt`. Only canonical geometry (bins/section/x/y/z) is used.
"""

import json
import os
import subprocess
import threading
import time

import numpy as np

ROOT = "/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/Workspace/merfish_test/missing_combos_50um/all_methods"
CANON = os.path.join(ROOT, "canonical")
PRED_ROOT = os.path.join(ROOT, "predictions")
METHODS_ROOT = os.path.join(ROOT, "methods")
N_GENES = 1122
ROW_STATUS = os.path.join(METHODS_ROOT, "RUNNER_A_STATUS.json")
LANE = {"job": "9383994", "node": "gvna02"}


# --------------------------------------------------------------------------- io
def load_scenarios():
    with open(os.path.join(CANON, "scenarios.json")) as f:
        obj = json.load(f)
    return obj["scenarios"]


def load_sections():
    with open(os.path.join(CANON, "sections.json")) as f:
        return json.load(f)


def load_bins(canon_dir=None):
    import anndata as ad

    canon = canon_dir or CANON
    path = os.path.join(canon, "bins_all.h5ad")
    a = ad.read_h5ad(path)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    genes = np.asarray(a.var_names).astype(str)
    out = {
        "X": np.ascontiguousarray(X, dtype=np.float32),
        "genes": genes,
        "section": a.obs["section"].astype(str).to_numpy(),
        "bin_id": a.obs_names.astype(str).to_numpy(),
        "x": a.obs["x"].to_numpy(np.float64),
        "y": a.obs["y"].to_numpy(np.float64),
        "z": a.obs["z"].to_numpy(np.float64),
        "ix": a.obs["ix"].to_numpy(np.int64),
        "iy": a.obs["iy"].to_numpy(np.int64),
        "n_cells": a.obs["n_cells"].to_numpy(np.int64),
    }
    assert out["X"].shape == (out["section"].size, N_GENES), out["X"].shape
    return out


def load_target_meta(canon_dir=None):
    """Load target_gt.npz metadata only. `gt` is deliberately not touched."""
    canon = canon_dir or CANON
    z = np.load(os.path.join(canon, "target_gt.npz"), allow_pickle=True)
    meta = {
        "scenario": z["scenario"].astype(str),
        "section": z["section"].astype(str),
        "bin_id": z["bin_id"].astype(str),
        "x": z["x"].astype(np.float64),
        "y": z["y"].astype(np.float64),
        "z": z["z"].astype(np.float64) if "z" in z.files else None,
        "n_rows": int(z["scenario"].size),
        "keys": list(z.files),
    }
    return meta


def build_problem(scen, bins, tmeta):
    """Visible training bins + hidden target bins for one scenario."""
    vis = [str(s) for s in scen["visible"]]
    tgt = [str(s) for s in scen["target"]]
    sec = bins["section"]
    vmask = np.isin(sec, vis)
    tr_idx = np.nonzero(vmask)[0]
    rows = np.nonzero(tmeta["scenario"] == scen["name"])[0]
    assert rows.size > 0, f"no target rows for scenario {scen['name']}"
    assert np.all(np.diff(rows) > 0), "scenario rows not ascending in target_gt"
    t_sec = tmeta["section"][rows]
    assert set(np.unique(t_sec)) == set(tgt), (
        f"{scen['name']}: target_gt sections {sorted(set(t_sec))} != scenario targets {sorted(tgt)}")
    # section-level order: target section asc, as produced by infra
    assert list(dict.fromkeys(t_sec.tolist())) == sorted(tgt), (
        f"{scen['name']}: target_gt row order not section-ascending")

    bid2z = dict(zip(bins["bin_id"].tolist(), bins["z"].tolist()))
    t_bin = tmeta["bin_id"][rows]
    missing = [b for b in t_bin.tolist() if b not in bid2z]
    assert not missing, f"{len(missing)} target bin_id not in bins_all, e.g. {missing[:3]}"

    coords_tr = np.stack([bins["x"][tr_idx], bins["y"][tr_idx], bins["z"][tr_idx]], 1)
    coords_te = np.stack(
        [tmeta["x"][rows], tmeta["y"][rows], np.array([bid2z[b] for b in t_bin.tolist()])], 1
    )
    return {
        "name": scen["name"],
        "k": int(scen["k"]),
        "visible": vis,
        "target": tgt,
        "tr_idx": tr_idx,
        "row_index": rows.astype(np.int64),
        "coords_tr": coords_tr.astype(np.float64),
        "coords_te": coords_te.astype(np.float64),
        "expr_tr": bins["X"][tr_idx],
        "target_section": t_sec,
        "target_bin_id": t_bin,
        "target_x": tmeta["x"][rows],
        "target_y": tmeta["y"][rows],
        "target_z": coords_te[:, 2],
    }


def pred_path(method, scenario):
    return os.path.join(PRED_ROOT, scenario, f"{method}.npz")


def write_prediction(method, prob, pred, extra=None):
    """Write pred aligned to the scenario's target_gt rows; verify before writing."""
    pred = np.asarray(pred)
    n, m = pred.shape
    assert m == N_GENES, pred.shape
    assert n == prob["row_index"].size, (pred.shape, prob["row_index"].size)
    assert np.isfinite(pred).all(), "non-finite prediction"
    pred = np.clip(pred, 0, None).astype(np.float32)

    path = pred_path(method, prob["name"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(
        pred=pred,
        row_index=prob["row_index"],
        bin_id=prob["target_bin_id"],
        section=prob["target_section"],
        x=prob["target_x"],
        y=prob["target_y"],
    )
    if extra:
        payload.update(extra)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)
    verify_prediction(path, method, prob)
    return path


def verify_prediction(path, method, prob):
    z = np.load(path, allow_pickle=True)
    for k in ["pred", "row_index", "bin_id", "section", "x", "y"]:
        assert k in z.files, (path, k, z.files)
    pred = z["pred"]
    assert pred.shape == (prob["row_index"].size, N_GENES), pred.shape
    assert pred.dtype == np.float32, pred.dtype
    assert np.isfinite(pred).all()
    assert (pred >= 0).all()
    assert np.array_equal(z["row_index"], prob["row_index"])
    assert np.array_equal(z["bin_id"].astype(str), prob["target_bin_id"].astype(str))
    assert np.array_equal(z["section"].astype(str), prob["target_section"].astype(str))
    assert np.allclose(z["x"], prob["target_x"]) and np.allclose(z["y"], prob["target_y"])
    return {
        "path": path,
        "shape": list(pred.shape),
        "bytes": os.path.getsize(path),
        "finite_frac": float(np.isfinite(pred).mean()),
        "coverage": 1.0,
        "pred_mean": float(pred.mean()),
        "pred_max": float(pred.max()),
        "const_genes": int((pred.std(0) == 0).sum()),
    }


# ------------------------------------------------------------------- monitoring
class ResourceMonitor(threading.Thread):
    """Device-wide GPU peak (nvidia-smi) + this process' RSS peak and mean CPU use."""

    def __init__(self, period=2.0, use_gpu=True):
        super().__init__(daemon=True)
        self.period = period
        self.use_gpu = use_gpu
        self.stop_evt = threading.Event()
        self.gpu_peak_mb = 0
        self.gpu_util_peak = 0
        self.rss_peak_mb = 0.0
        self._cpu_last = self._cpu_now()
        self._t_last = time.time()
        self.cpu_cores_sum = 0.0
        self.cpu_samples = 0

    @staticmethod
    def _cpu_now():
        try:
            with open("/proc/self/stat") as f:
                parts = f.read().split()
            return (int(parts[13]) + int(parts[14])) / os.sysconf("SC_CLK_TCK")
        except Exception:
            return 0.0

    def _rss_mb(self):
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
        return 0.0

    def run(self):
        while not self.stop_evt.is_set():
            if self.use_gpu:
                try:
                    r = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10)
                    for line in r.stdout.strip().splitlines():
                        mem, util = [x.strip() for x in line.split(",")[:2]]
                        self.gpu_peak_mb = max(self.gpu_peak_mb, int(float(mem)))
                        self.gpu_util_peak = max(self.gpu_util_peak, int(float(util)))
                except Exception:
                    pass
            self.rss_peak_mb = max(self.rss_peak_mb, self._rss_mb())
            cpu_now, t_now = self._cpu_now(), time.time()
            dt = t_now - self._t_last
            if dt > 0:
                self.cpu_cores_sum += (cpu_now - self._cpu_last) / dt
                self.cpu_samples += 1
            self._cpu_last, self._t_last = cpu_now, t_now
            self.stop_evt.wait(self.period)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=20)
        self.rss_peak_mb = max(self.rss_peak_mb, self._rss_mb())
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_alloc_peak_mb = torch.cuda.max_memory_allocated() / 2**20
                self.gpu_reserved_peak_mb = torch.cuda.max_memory_reserved() / 2**20
        except Exception:
            pass

    def summary(self):
        return {
            "gpu_peak_mb_device": int(self.gpu_peak_mb),
            "gpu_util_peak_pct": int(self.gpu_util_peak),
            "rss_peak_mb": round(self.rss_peak_mb, 1),
            "cpu_cores_mean": round(self.cpu_cores_sum / max(1, self.cpu_samples), 2),
            "gpu_peak_note": "device-wide nvidia-smi (co-tenants included); rss/cpu are this process",
        }


class Status:
    """Atomic status.json + run.log writer."""

    def __init__(self, method, outdir):
        self.method = method
        self.outdir = outdir
        self.path = os.path.join(outdir, "status.json")
        self.logpath = os.path.join(outdir, "run.log")
        self.t0 = time.time()
        self.state = {
            "method": method,
            "dataset": "merfish_5slice_all",
            "status": "RUNNING",
            "stage": "init",
            "pid": os.getpid(),
            "host": os.uname().nodename,
            "slurm_job": os.environ.get("SLURM_JOB_ID", LANE["job"]),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "scenarios_total": 7,
            "scenarios_done": 0,
            "scenarios": {},
        }
        self.write()

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')} +{time.time()-self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        with open(self.logpath, "a") as f:
            f.write(line + "\n")

    def write(self, **kw):
        self.state.update(kw)
        self.state["elapsed_sec"] = round(time.time() - self.t0, 1)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
        os.replace(tmp, self.path)

    def scenario_done(self, name, res):
        self.state["scenarios"][name] = res
        self.state["scenarios_done"] = len(self.state["scenarios"])
        self.write()

    def finish(self, status, **kw):
        self.write(status=status, stage=status.lower(), **kw)


def update_runner_status(**kw):
    """Merge-update methods/RUNNER_A_STATUS.json (all three methods)."""
    cur = {}
    if os.path.exists(ROW_STATUS):
        try:
            with open(ROW_STATUS) as f:
                cur = json.load(f)
        except Exception:
            cur = {}
    cur.update(kw)
    cur["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = ROW_STATUS + ".tmp"
    os.makedirs(os.path.dirname(ROW_STATUS), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(cur, f, indent=2, default=str)
    os.replace(tmp, ROW_STATUS)


def set_method(self_method, status, **extra):
    """Update one method's entry inside RUNNER_A_STATUS.json."""
    cur = {}
    if os.path.exists(ROW_STATUS):
        try:
            with open(ROW_STATUS) as f:
                cur = json.load(f)
        except Exception:
            cur = {}
    methods = cur.setdefault("methods", {})
    entry = methods.setdefault(self_method, {})
    entry.update(status=status, **extra)
    entry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    cur.setdefault("lane", LANE)
    cur["methods"] = methods
    cur["updated_at"] = entry["updated_at"]
    tmp = ROW_STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cur, f, indent=2, default=str)
    os.replace(tmp, ROW_STATUS)


def existing_ok(method, prob):
    p = pred_path(method, prob["name"])
    if not os.path.exists(p):
        return False
    try:
        z = np.load(p, allow_pickle=True)
        if z["pred"].shape != (prob["row_index"].size, N_GENES):
            return False
        return np.array_equal(z["row_index"], prob["row_index"]) and np.isfinite(z["pred"]).all()
    except Exception:
        return False

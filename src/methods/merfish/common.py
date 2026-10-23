#!/usr/bin/env python3
"""Shared helpers for the heart 10-slice (21-30) target-only runners.

Contract (read-only on canonical):
  canonical/bins_all.h5ad   10 sections, 50 um bins, mean normalised X
  canonical/scenarios.json  26 scenarios with visible / target section ids
  canonical/target_gt.npz   bin_id, section, x, y, gt (shared superset)

Per scenario a method writes predictions/<scenario>/<method>.npz with
  pred       (n_rows_scenario, n_genes) float32, aligned to target_gt order
  row_index  (n_rows_scenario,) int64 index into target_gt arrays
  bin_id, section, x, y   echoed from target_gt[row_index]
  method, scenario, genes

Only target bins are stored: no full 3D volume is ever written.
"""
import json
import os
import threading
import time
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("MERFISH5_ROOT", Path.cwd())).resolve()
CANON = ROOT / "canonical"
BINS_H5 = CANON / "bins_all.h5ad"
SCENARIOS_JSON = CANON / "scenarios.json"
TARGET_GT = CANON / "target_gt.npz"
READY = CANON / "READY"
BLOCKED = CANON / "BLOCKED"
PRED_ROOT = ROOT / "predictions"
METHODS_ROOT = ROOT / "methods"

SECTIONS = ["1", "2", "3", "4", "5"]


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


class Status:
    """Atomic heartbeat status.json for one method."""

    def __init__(self, path, method, extra=None):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.d = {"method": method, "status": "RUNNING", "stage": "start",
                  "host": os.uname().nodename,
                  "slurm_job": os.environ.get("SLURM_JOB_ID"),
                  "pid": os.getpid(), "started_at": now(),
                  "updated_at": now(), "elapsed_sec": 0.0,
                  "gpu_peak_mb": 0, "scenarios_done": 0, "scenarios_total": 7}
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
        self.stop = threading.Event()
        self.idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

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


class CpuSampler(threading.Thread):
    def __init__(self, period=5.0):
        super().__init__(daemon=True)
        self.period = period
        self.peak_rss = 0
        self.stop = threading.Event()

    def run(self):
        try:
            import resource
            while not self.stop.is_set():
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB
                self.peak_rss = max(self.peak_rss, int(rss))
                self.stop.wait(self.period)
        except Exception:
            pass


def wait_ready(timeout=None, poll=30):
    """Block until canonical/READY exists; stop on canonical/BLOCKED."""
    t0 = time.time()
    while True:
        if BLOCKED.exists():
            raise SystemExit(f"BLOCKED marker present: {BLOCKED.read_text(errors='replace')[:2000]}")
        if READY.exists():
            return True
        if timeout is not None and time.time() - t0 > timeout:
            return False
        time.sleep(poll)


def load_bins(sparse_to_dense=True):
    import anndata as ad
    a = ad.read_h5ad(BINS_H5)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    X = np.ascontiguousarray(X, dtype=np.float32)
    obs = a.obs
    def pick(names, default=None):
        for n in names:
            if n in obs.columns:
                return obs[n].to_numpy()
        return default
    section = pick(["section", "slice", "section_id"])
    if section is None:
        section = np.array([str(s) for s in range(21, 31)], dtype=object)
    z = pick(["z", "z_um", "z_mm"])
    if z is None and "spatial" in a.obsm:
        z = np.asarray(a.obsm["spatial"])[:, 2]
    x = pick(["x", "x_um", "x_mm"])
    y = pick(["y", "y_um", "y_mm"])
    if x is None and "spatial" in a.obsm:
        x = np.asarray(a.obsm["spatial"])[:, 0]
        y = np.asarray(a.obsm["spatial"])[:, 1]
    n_cells = pick(["n_cells", "ncells"])
    bin_id = pick(["bin_id"])
    if bin_id is None:
        bin_id = np.asarray(a.obs_names).astype(str)
    else:
        bin_id = np.asarray(bin_id).astype(str)
    return dict(
        adata=a, X=X, genes=np.asarray(a.var_names).astype(str),
        bin_id=bin_id,
        section=norm_section(section),
        z=np.asarray(z, dtype=np.float64),
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        n_cells=None if n_cells is None else np.asarray(n_cells),
    )


def align_prediction(tg, rows, section_pred, bins):
    """Place per-section predictions into target_gt row order.

    section_pred: {section: (bins_rows, pred[n_bins, n_genes])} where bins_rows
    index into the bins_all arrays in the same order as pred.
    """
    n_genes = next(iter(section_pred.values()))[1].shape[1]
    out = np.full((rows.size, n_genes), np.nan, dtype=np.float32)
    tgt_section = tg["section"][rows]
    tgt_bin = tg["bin_id"][rows]
    for s, (brow, P) in section_pred.items():
        m = tgt_section == s
        if not m.any():
            continue
        if P.shape[0] != brow.size:
            raise ValueError(f"section {s}: pred rows {P.shape[0]} != bins rows {brow.size}")
        local = {bins["bin_id"][brow[j]]: j for j in range(brow.size)}
        need = tgt_bin[m]
        missing = [b for b in need if b not in local]
        if missing:
            raise ValueError(f"section {s}: {len(missing)} target bin_id not in bins_all, e.g. {missing[:3]}")
        out[m] = P[[local[b] for b in need]]
    if not np.isfinite(out).all():
        raise ValueError(f"{int((~np.isfinite(out)).sum())} target rows unaligned")
    return out


def norm_section(v):
    out = []
    for s in np.asarray(v).astype(str):
        s = s.strip()
        if s.startswith("s") and s[1:].isdigit():
            s = s[1:]
        if s.isdigit():
            s = str(int(s))
        out.append(s)
    return np.array(out, dtype=object)


norm_sections = norm_section


def load_scenarios():
    d = json.loads(SCENARIOS_JSON.read_text())
    items = d["scenarios"] if isinstance(d, dict) and "scenarios" in d else d
    out = []
    for i, it in enumerate(items):
        name = str(it.get("name", it.get("id", it.get("scenario", f"scenario_{i:02d}"))))
        vis = it.get("visible", it.get("visible_sections", it.get("visible_ids")))
        tgt = it.get("target", it.get("targets", it.get("hidden",
              it.get("target_sections", it.get("hidden_sections", it.get("missing"))))))
        if vis is None or tgt is None:
            raise ValueError(f"scenario {name}: cannot find visible/target keys; got {list(it)}")
        out.append(dict(name=name, visible=norm_section(vis).tolist(),
                        target=norm_section(tgt).tolist(), raw=it))
    return out


def load_target_gt():
    """Metadata only: `gt` is deliberately never loaded by method runners."""
    z = np.load(TARGET_GT, allow_pickle=True)
    need = ["bin_id", "section", "x", "y"]
    missing = [k for k in need if k not in z.files]
    if missing:
        raise ValueError(f"target_gt.npz missing {missing}; has {z.files}")
    return dict(
        bin_id=np.asarray(z["bin_id"]).astype(str),
        section=norm_section(np.asarray(z["section"]).astype(str)),
        x=np.asarray(z["x"], dtype=np.float64),
        y=np.asarray(z["y"], dtype=np.float64),
        has_scenario="scenario" in z.files,
        scenario=norm_section(np.asarray(z["scenario"]).astype(str)) if "scenario" in z.files else None,
    )


def scenario_rows(tg, sc):
    """Rows of target_gt that belong to this scenario (in target_gt order)."""
    if tg["has_scenario"]:
        m = tg["scenario"] == sc["name"]
    else:
        m = np.isin(tg["section"], np.asarray(sc["target"], dtype=object))
    rows = np.flatnonzero(m)
    if rows.size == 0:
        raise ValueError(f"scenario {sc['name']}: no target_gt rows for {sc['target']}")
    return rows


def save_pred(method, sc, tg, rows, pred, extra=None):
    """Write predictions/<scenario>/<method>.npz (target rows only)."""
    pred = np.ascontiguousarray(pred, dtype=np.float32)
    if pred.shape[0] != rows.size:
        raise ValueError(f"{method}/{sc['name']}: pred rows {pred.shape[0]} != target rows {rows.size}")
    if not np.isfinite(pred).all():
        bad = int((~np.isfinite(pred)).sum())
        raise ValueError(f"{method}/{sc['name']}: {bad} non-finite predictions")
    outdir = PRED_ROOT / sc["name"]
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{method}.npz"
    tmp = outdir / f".{method}.npz.tmp"
    payload = dict(
        pred=pred,
        row_index=rows.astype(np.int64),
        bin_id=np.asarray(tg["bin_id"][rows], dtype=str),
        section=np.asarray(tg["section"][rows], dtype=str),
        x=tg["x"][rows],
        y=tg["y"][rows],
        method=np.array(method),
        scenario=np.array(sc["name"]),
    )
    if extra:
        payload.update(extra)
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    os.replace(tmp, path)
    return path


def block_bounds(sc):
    """Split a scenario's target sections into contiguous blocks.

    Returns list of (target_sections, lo_visible, hi_visible) in z order.
    """
    vis = sorted(set(sc["visible"]), key=int)
    tgt = sorted(set(sc["target"]), key=int)
    blocks = []
    cur = [tgt[0]]
    for s in tgt[1:]:
        if int(s) == int(cur[-1]) + 1:
            cur.append(s)
        else:
            blocks.append(cur)
            cur = [s]
    blocks.append(cur)
    out = []
    for b in blocks:
        lo = [v for v in vis if int(v) < int(b[0])]
        hi = [v for v in vis if int(v) > int(b[-1])]
        if not lo or not hi:
            raise ValueError(f"block {b} has no visible bracket: visible={vis}")
        out.append((b, lo[-1], hi[0]))
    return out


def section_z(z, section):
    """Map section id -> z (mean of its bins)."""
    out = {}
    for s in np.unique(section):
        m = section == s
        out[str(s)] = float(np.mean(z[m]))
    return out


def method_dir(method):
    return METHODS_ROOT / method

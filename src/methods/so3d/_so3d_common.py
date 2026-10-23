#!/usr/bin/env python
"""Shared helpers for the So3D 46-dataset held-out-middle-slice benchmark (runner B).

Canonical contract (built by the prep runner, never written here):
  canonical/<dataset>/visible.h5ad  AnnData, X (n_visible, G) sparse float32 in the
                                   source's provided normalised space (NEVER
                                   renormalised here), obs['slice'] (int), obs['z'],
                                   obsm['spatial'] == (x, y, z) source units
  canonical/<dataset>/target.npz   keys X (n_target, G; EVALUATION ONLY - never read),
                                   coords (n_target, 3), obs_names, slice, z
  canonical/<dataset>/meta.json    dataset / slices / target_slice / z_range / genes
  canonical/PREP_READY             gate: no data is read before it exists

Prediction contract (predictions/<dataset>/<method>.npz), row-for-row aligned to
target.npz:
  pred      (n_target, G) float32, non-negative, finite
  coords    (n_target, 3) float32, echoed from target
  obs_names (n_target,)
  slice     (n_target,) echoed
  row_index (n_target,) int64 == arange(n_target)
  genes     (G,) method-visible gene order (same as visible.h5ad var_names)

No target expression is ever loaded by this module.  `load_target` only touches
coords/obs_names/slice/z.
"""

import json
import os
import subprocess
import threading
import time

import numpy as np

ROOT = "/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/Workspace/so3d_bench"
CANON = os.path.join(ROOT, "canonical")
PRED_ROOT = os.path.join(ROOT, "predictions")
METHODS_ROOT = os.path.join(ROOT, "methods")
LOGS_ROOT = os.path.join(ROOT, "logs")
# The prep runner wrote the gate at <root>/PREP_READY (contract file); the task
# text names canonical/PREP_READY.  Accept either, prefer canonical/.
READY_CANDIDATES = [os.path.join(CANON, "PREP_READY"),
                    os.path.join(ROOT, "PREP_READY")]

LANE = {"job": "9383992", "node": "gvna03"}
N_DATASETS = 46

# Coordinate frame convention shared by both methods (only visible+target
# GEOMETRY is used; no expression):
#   xy is rescaled so that the median within-slice nearest-neighbour distance of
#   visible spots equals C2C_UM (the official SINTER3D c2c_dist convention).
#   z is rescaled so that the median gap between consecutive visible slices
#   equals C2C_UM as well (slice spacing is unknown per dataset; the native z
#   units differ wildly: 0..0.4 for embryo data vs 0..80 for BRCA, so a single
#   isotropic native scale is meaningless).  This makes SINTER3D's `normalize`
#   (=100) behave exactly as in the official µm frame.
C2C_UM = 100.0


# --------------------------------------------------------------------- basics
def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=jdefault)
    os.replace(tmp, path)


def prep_ready_path():
    for p in READY_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def wait_ready(timeout=7200, poll=20, log=print):
    """Block until a PREP_READY gate exists. Never builds canonical data."""
    t0 = time.time()
    warned = False
    while prep_ready_path() is None:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"no PREP_READY among {READY_CANDIDATES} after {timeout}s")
        if not warned:
            log(f"[wait] no PREP_READY ({READY_CANDIDATES}); polling every {poll}s "
                f"(not building data)")
            warned = True
        time.sleep(poll)
    log(f"[wait] PREP_READY at {prep_ready_path()} after {time.time() - t0:.0f}s")


def dataset_dirs(require_all=True):
    out = []
    if not os.path.isdir(CANON):
        return out
    for name in sorted(os.listdir(CANON)):
        d = os.path.join(CANON, name)
        if not os.path.isdir(d):
            continue
        need = ["visible.h5ad", "target.npz", "meta.json"]
        have = [os.path.exists(os.path.join(d, f)) for f in need]
        if require_all and not all(have):
            continue
        if not any(have):
            continue
        out.append(name)
    return out


def incomplete_datasets():
    out = {}
    if not os.path.isdir(CANON):
        return out
    for name in sorted(os.listdir(CANON)):
        d = os.path.join(CANON, name)
        if not os.path.isdir(d):
            continue
        missing = [f for f in ("visible.h5ad", "target.npz", "meta.json")
                   if not os.path.exists(os.path.join(d, f))]
        if missing:
            out[name] = missing
    return out


# ----------------------------------------------------------------------- io
def load_meta(ds):
    with open(os.path.join(CANON, ds, "meta.json")) as f:
        return json.load(f)


def load_visible(ds):
    import anndata as ad

    path = os.path.join(CANON, ds, "visible.h5ad")
    a = ad.read_h5ad(path)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    X = np.ascontiguousarray(X, dtype=np.float32)
    coords = np.asarray(a.obsm["spatial"], dtype=np.float64)
    if coords.shape[1] > 3:
        coords = coords[:, :3]
    sec = a.obs["slice"].to_numpy()
    sec = np.asarray(sec, dtype=np.int64) if sec.dtype.kind in "iu" else np.asarray(
        [int(str(s)) for s in sec], dtype=np.int64)
    z = (a.obs["z"].to_numpy(np.float64) if "z" in a.obs.columns
         else coords[:, 2].astype(np.float64))
    out = {
        "X": X,
        "genes": np.asarray(a.var_names).astype(str),
        "coords": coords,
        "slice": sec,
        "z": z,
        "obs_names": a.obs_names.astype(str).to_numpy(),
        "n": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
    }
    assert np.isfinite(X).all(), f"{ds}: non-finite visible X"
    assert np.isfinite(coords).all(), f"{ds}: non-finite visible coords"
    return out


def load_target(ds):
    """Target GEOMETRY only. `target.npz['X']` is evaluation-only and is never read."""
    p = os.path.join(CANON, ds, "target.npz")
    z = np.load(p, allow_pickle=True)
    coords = np.asarray(z["coords"], dtype=np.float64)
    obs_names = np.asarray(z["obs_names"]).astype(str)
    sec = np.asarray(z["slice"])
    if sec.dtype.kind not in "iu":
        sec = np.asarray([int(str(s)) for s in sec], dtype=np.int64)
    else:
        sec = sec.astype(np.int64)
    zz = (np.asarray(z["z"], dtype=np.float64) if "z" in z.files
          else coords[:, 2])
    out = {
        "coords": coords,
        "obs_names": obs_names,
        "slice": sec,
        "z": zz,
        "n": int(coords.shape[0]),
        "keys": list(z.files),
    }
    z.close()
    assert "X" not in out, "target expression must never be materialised"
    return out


# ------------------------------------------------------------------- geometry
def _med_nn_xy(xy):
    from scipy.spatial import cKDTree

    if xy.shape[0] < 3:
        return float("nan")
    d, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:, 1]))


def normalize_geometry(vis, tgt):
    """Visible-only scale estimate; returns rescaled coords + diagnostics.

    Target coordinates are transformed with the *visible-derived* scales, so no
    target expression or target-derived statistics enter.
    """
    xy = vis["coords"][:, :2]
    per_slice = []
    for s in np.unique(vis["slice"]):
        m = vis["slice"] == s
        if m.sum() >= 3:
            per_slice.append(_med_nn_xy(xy[m]))
    per_slice = [v for v in per_slice if np.isfinite(v) and v > 0]
    nn_xy = float(np.median(per_slice)) if per_slice else float("nan")
    if not np.isfinite(nn_xy) or nn_xy <= 0:
        raise ValueError("cannot estimate within-slice xy spacing")

    zs = np.array([float(np.median(vis["z"][vis["slice"] == s]))
                   for s in np.unique(vis["slice"])])
    zs = np.sort(zs)
    gaps = np.diff(zs)
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    z_gap = float(np.median(gaps)) if gaps.size else float("nan")
    if not np.isfinite(z_gap) or z_gap <= 0:
        raise ValueError("cannot estimate inter-slice z spacing")

    s_xy = C2C_UM / nn_xy
    s_z = C2C_UM / z_gap
    z0 = float(zs.min())

    def tf(c):
        o = np.empty_like(c, dtype=np.float64)
        o[:, 0] = c[:, 0] * s_xy
        o[:, 1] = c[:, 1] * s_xy
        o[:, 2] = (c[:, 2] - z0) * s_z
        return o

    diag = {
        "frame": (f"xy median within-slice NN -> {C2C_UM} um; median inter-slice "
                  f"z gap -> {C2C_UM} um (visible-only scales)"),
        "nn_xy_native": nn_xy,
        "z_gap_native": z_gap,
        "scale_xy": s_xy,
        "scale_z": s_z,
        "z0_native": z0,
        "native_bbox": {
            "x": [float(vis["coords"][:, 0].min()), float(vis["coords"][:, 0].max())],
            "y": [float(vis["coords"][:, 1].min()), float(vis["coords"][:, 1].max())],
            "z": [float(vis["coords"][:, 2].min()), float(vis["coords"][:, 2].max())],
        },
        "target_z_native": float(np.median(tgt["z"])) if tgt["n"] else None,
    }
    return tf(vis["coords"]), tf(tgt["coords"]), diag


def slice_order(vis):
    """Visible slice ids sorted by their median z position."""
    secs = np.unique(vis["slice"])
    zs = {int(s): float(np.median(vis["z"][vis["slice"] == s])) for s in secs}
    return [s for s in sorted(zs, key=lambda s: zs[s])], zs


def bracket_slices(vis, z_target):
    """Nearest visible slice below and above the target z (native z, ratio only)."""
    order, zs = slice_order(vis)
    below = [s for s in order if zs[s] < z_target]
    above = [s for s in order if zs[s] > z_target]
    lo = below[-1] if below else None
    hi = above[0] if above else None
    return lo, hi, zs, order


# ---------------------------------------------------------------- predictions
def pred_path(method, ds):
    return os.path.join(PRED_ROOT, ds, f"{method}.npz")


def write_prediction(method, ds, pred, tgt, genes, extra=None):
    pred = np.asarray(pred, dtype=np.float32)
    assert pred.shape == (tgt["n"], genes.size), (pred.shape, tgt["n"], genes.size)
    assert np.isfinite(pred).all(), f"{ds}: non-finite prediction"
    pred = np.clip(pred, 0, None)
    path = pred_path(method, ds)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(
        pred=pred,
        coords=np.asarray(tgt["coords"], dtype=np.float32),
        obs_names=np.asarray(tgt["obs_names"]).astype(str),
        slice=np.asarray(tgt["slice"], dtype=np.int64),
        row_index=np.arange(tgt["n"], dtype=np.int64),
        genes=np.asarray(genes).astype(str),
    )
    if extra:
        payload.update(extra)
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)
    return path


def verify_prediction(method, ds, genes, tgt):
    path = pred_path(method, ds)
    z = np.load(path, allow_pickle=True)
    for k in ("pred", "coords", "obs_names", "slice", "row_index", "genes"):
        assert k in z.files, (path, k, z.files)
    pred = z["pred"]
    checks = {
        "shape_ok": list(pred.shape) == [tgt["n"], genes.size],
        "dtype": str(pred.dtype),
        "finite": bool(np.isfinite(pred).all()),
        "nonneg": bool((pred >= 0).all()),
        "genes_ok": bool(np.array_equal(z["genes"].astype(str), np.asarray(genes).astype(str))),
        "obs_names_ok": bool(np.array_equal(z["obs_names"].astype(str),
                                            np.asarray(tgt["obs_names"]).astype(str))),
        "slice_ok": bool(np.array_equal(z["slice"].astype(np.int64),
                                        np.asarray(tgt["slice"]).astype(np.int64))),
        "coords_ok": bool(np.allclose(z["coords"], tgt["coords"], atol=1e-4)),
        "row_index_ok": bool(np.array_equal(z["row_index"], np.arange(tgt["n"]))),
    }
    std = pred.std(axis=0)
    checks.update({
        "bytes": os.path.getsize(path),
        "coverage": 1.0,
        "pred_mean": float(pred.mean()),
        "pred_min": float(pred.min()),
        "pred_max": float(pred.max()),
        "zero_frac": float((pred == 0).mean()),
        "per_gene_std_mean": float(std.mean()),
        "const_genes": int((std == 0).sum()),
        "pass": bool(all(checks[k] for k in (
            "shape_ok", "finite", "nonneg", "genes_ok", "obs_names_ok",
            "slice_ok", "coords_ok", "row_index_ok"))),
    })
    z.close()
    return checks


def dataset_status_path(method, ds):
    return os.path.join(PRED_ROOT, ds, f"{method}.status.json")


def write_dataset_status(method, ds, rec):
    """Per-dataset status file required by FORMAT_CONTRACT.md."""
    payload = {"method": method, "dataset": ds,
               "status": rec.get("status"),
               "reason": rec.get("reason") or rec.get("error"),
               "updated_at": now()}
    for k in ("n_visible", "n_target", "n_genes", "target_slice", "total_sec",
              "output", "error"):
        if k in rec:
            payload[k] = rec[k]
    os.makedirs(os.path.join(PRED_ROOT, ds), exist_ok=True)
    write_json(dataset_status_path(method, ds), payload)


def existing_ok(method, ds, genes, tgt):
    try:
        return bool(verify_prediction(method, ds, genes, tgt)["pass"])
    except Exception:
        return False


# ------------------------------------------------------------ status/monitor
class Status:
    """Per-method status.json with per-dataset records and live heartbeats."""

    def __init__(self, method, outdir, extra=None):
        self.method = method
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.path = os.path.join(outdir, "status.json")
        self.logpath = os.path.join(outdir, "run.log")
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.d = {
            "method": method,
            "status": "RUNNING",
            "stage": "start",
            "lane": LANE,
            "host": os.uname().nodename,
            "slurm_jobid": os.environ.get("SLURM_JOB_ID"),
            "pid": os.getpid(),
            "started_at": now(),
            "updated_at": now(),
            "elapsed_sec": 0.0,
            "datasets_total": N_DATASETS,
            "datasets_done": 0,
            "datasets_failed": 0,
            "datasets_unsupported": 0,
            "gpu_peak_mb": 0,
            "records": {},
            "log": self.logpath,
        }
        if extra:
            self.d.update(extra)
        self.write()
        with open(self.logpath, "a") as f:
            f.write(f"===== {method} start {now()} pid={os.getpid()} =====\n")

    def log(self, msg):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with open(self.logpath, "a") as f:
            f.write(line + "\n")

    def write(self, **kw):
        with self.lock:
            self.d.update(kw)
            self.d["elapsed_sec"] = round(time.time() - self.t0, 1)
            self.d["updated_at"] = now()
            write_json(self.path, self.d)

    def record(self, ds, rec):
        with self.lock:
            self.d["records"][ds] = rec
            self.d["datasets_done"] = sum(1 for v in self.d["records"].values()
                                          if v.get("status") == "DONE")
            self.d["datasets_failed"] = sum(1 for v in self.d["records"].values()
                                            if v.get("status") == "FAILED")
            self.d["datasets_unsupported"] = sum(1 for v in self.d["records"].values()
                                                 if v.get("status") == "UNSUPPORTED")
            self.d["updated_at"] = now()
            write_json(self.path, self.d)

    def finish(self, status, **kw):
        self.write(status=status, stage="finished", finished_at=now(), **kw)


class GpuSampler(threading.Thread):
    """Device-wide GPU peak from nvidia-smi (co-tenants share the card)."""

    def __init__(self, period=3.0):
        super().__init__(daemon=True)
        self.period = period
        self.peak = 0
        self.peak_util = 0
        self.stop_evt = threading.Event()

    def run(self):
        idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        while not self.stop_evt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--id={idx}", "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                if out:
                    mem, util = out.splitlines()[0].split(",")
                    self.peak = max(self.peak, int(mem))
                    self.peak_util = max(self.peak_util, int(util))
            except Exception:
                pass
            self.stop_evt.wait(self.period)


class ResourceMonitor(threading.Thread):
    """This process' peak RSS and mean CPU count."""

    def __init__(self, period=3.0):
        super().__init__(daemon=True)
        self.period = period
        self.peak_rss_mb = 0.0
        self.cpu_samples = []
        self.stop_evt = threading.Event()
        self._last = None

    def run(self):
        while not self.stop_evt.is_set():
            try:
                with open("/proc/self/statm") as f:
                    rss = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
                self.peak_rss_mb = max(self.peak_rss_mb, rss)
                with open("/proc/self/stat") as f:
                    parts = f.read().split()
                cpu = (int(parts[13]) + int(parts[14])) / os.sysconf("SC_CLK_TCK")
                if self._last is not None:
                    self.cpu_samples.append((cpu - self._last) / self.period)
                self._last = cpu
            except Exception:
                pass
            self.stop_evt.wait(self.period)

    def summary(self):
        return {
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "mean_cpu_cores": round(float(np.mean(self.cpu_samples)), 2) if self.cpu_samples else None,
            "n_cpu_samples": len(self.cpu_samples),
        }


def update_runner_status(method, summary):
    """Merge one method's summary into methods/RUNNER_B_STATUS.json."""
    path = os.path.join(METHODS_ROOT, "RUNNER_B_STATUS.json")
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            data = {}
    data.setdefault("runner", "B")
    data.setdefault("lane", LANE)
    data["methods"] = data.get("methods", {})
    data["methods"][method] = summary
    data["datasets_total"] = N_DATASETS
    data["coverage"] = {m: f"{v.get('n_done', 0)}/{N_DATASETS}"
                        for m, v in data["methods"].items()}
    data["updated_at"] = now()
    write_json(path, data)
    return path

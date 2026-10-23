#!/usr/bin/env python3
"""isoST on the MERFISH 5-slice (.006-.010) target-only benchmark.

Per scenario: train the isoST dual ODE (official model code, unmodified) on the
VISIBLE sections only, then integrate f from the lower visible slice and g from
the upper visible slice to every hidden target section depth and map the
virtual points to the target bins (per-layer 2D NN in normalised xy).

Only target bins are written (predictions/<scenario>/isoST.npz). No volume.

Usage (inside the granted srun step):
  python run_heart.py --pilot [--scenario <name>]
  python run_heart.py                 # all 26 scenarios
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

ISOST_DIR = os.path.join(os.environ["MERFISH5_UPSTREAM"], "isoST")
sys.path.insert(0, ISOST_DIR)

CUDA_HOME = "/soft/devtools/cuda/cuda-12.9"
os.environ.setdefault("CUDA_HOME", CUDA_HOME)
os.environ.setdefault("CUDA_PATH", CUDA_HOME)
os.environ["PATH"] = f"{CUDA_HOME}/bin:" + os.environ.get("PATH", "")
os.environ["LIBRARY_PATH"] = f"{CUDA_HOME}/lib64:" + os.environ.get("LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{CUDA_HOME}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

METHOD = "isoST"
OUT = C.METHODS_ROOT / METHOD
REFERENCE_MEDIAN_Z_GAP = 0.928329
SECTION_GAP_UM = 200.0
SCALE_Z = REFERENCE_MEDIAN_Z_GAP / SECTION_GAP_UM      # 0.0046416 model units per um
DELTA_D = REFERENCE_MEDIAN_Z_GAP / 4.0                 # 0.23208, ~50 um euler step

LOG = OUT / "run.log"


def log(msg):
    line = f"[{C.now()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def make_params(gene_dim):
    return dict(
        gene_dim=int(gene_dim), hidden_dim=64, head_num=1, K=8,
        method="euler", delta_d=DELTA_D, stride=1, lr=0.001,
        optimizer_name="NAdam", weight_decay=1e-8, warm_up_rate=1.0,
        std_x=0.01, std_y=0.01, std_z=0.1, std_seq=0.1, alpha=0.1, dual=True,
        beta_start_value=1.0, beta_end_value=0.05, beta_start_iteration=50,
        beta_n_iterations=50,
    )


class Fitter:
    def __init__(self, bins):
        self.bins = bins
        self.xmin = float(bins["x"].min())
        self.ymin = float(bins["y"].min())
        self.W = float(max(bins["x"].max() - self.xmin, bins["y"].max() - self.ymin))
        self.zmin = float(bins["z"].min())
        self.genes = bins["genes"]
        self.n_genes = len(self.genes)

    def xy(self, x, y):
        return np.stack([(x - self.xmin) / self.W, (y - self.ymin) / self.W], axis=1).astype(np.float32)

    def zmodel(self, z):
        return (np.asarray(z, dtype=np.float64) - self.zmin) * SCALE_Z


def build_training(fit, bins, visible):
    """Feature scaling + per-visible-section tensors (visible only)."""
    masks = {s: bins["section"] == s for s in visible}
    vis_rows = np.flatnonzero(np.isin(bins["section"], np.asarray(visible, dtype=object)))
    Xv = bins["X"][vis_rows]
    fmin = Xv.min(axis=0)
    frange = (Xv.max(axis=0) - fmin).astype(np.float32)
    const = frange <= 1e-12
    Fn = (Xv - fmin) / np.where(const, 1.0, frange)
    Fn[:, const] = 0.0
    Fn = np.clip(Fn, 0.0, 1.0).astype(np.float32)
    pos = {int(r): i for i, r in enumerate(vis_rows)}

    tensors, zs = [], []
    for s in visible:
        rows = np.flatnonzero(masks[s])
        idx = np.array([pos[int(r)] for r in rows], dtype=np.int64)
        zc = float(np.mean(bins["z"][rows]))
        zmod = float(fit.zmodel(zc))
        t = np.column_stack([fit.xy(bins["x"][rows], bins["y"][rows]),
                             np.full(rows.size, zmod, np.float32), Fn[idx]]).astype(np.float32)
        tensors.append(t)
        zs.append(zmod)
    return dict(tensors=tensors, z_model=zs, fmin=fmin, frange=frange, const=const,
                n_visible_bins=int(vis_rows.size))


def train_scenario(bins, sc, epochs, exp_dir):
    import torch
    from model.base_model import IsoST as TrainerClass
    from utils.train_ode import train

    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    fit = Fitter(bins)
    td = build_training(fit, bins, sc["visible"])
    slices_t = [torch.from_numpy(t).cuda() for t in td["tensors"]]
    trainer = TrainerClass(device="cuda:0", **make_params(fit.n_genes)).cuda()
    exp_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train(trainer, str(exp_dir), [slices_t], batch_num=1,
          epochs_list=epochs, checkpoint_every=10000, backup_every=10000)
    sec = time.time() - t0
    trainer.eval()
    return dict(trainer=trainer, slices=slices_t, fit=fit, td=td,
                train_sec=round(sec, 1), n_params=int(sum(p.numel() for p in trainer.parameters())))


def inv_features(fn, td):
    fn = np.clip(np.asarray(fn, dtype=np.float32), 0.0, 1.0)
    out = fn * td["frange"] + td["fmin"]
    out[:, td["const"]] = 0.0
    return np.clip(out, 0.0, None).astype(np.float32)


def predict(bins, sc, trained):
    import torch
    from torchdiffeq import odeint
    from scipy.spatial import cKDTree

    trainer, slices, fit, td = trained["trainer"], trained["slices"], trained["fit"], trained["td"]
    z_list = td["z_model"]
    section_pred = {}
    map_stats = {}
    infer_t0 = time.time()
    with torch.no_grad():
        for block, lo, hi in C.block_bounds(sc):
            li = sc["visible"].index(lo)
            hj = sc["visible"].index(hi)
            z_lo, z_hi = z_list[li], z_list[hj]
            for tsec in block:
                rows_all = np.flatnonzero(bins["section"] == tsec)
                z_t = float(fit.zmodel(np.mean(bins["z"][rows_all])))
                depths = torch.tensor([z_lo, z_t, z_hi], dtype=torch.float32, device="cuda:0")
                traj_f = odeint(trainer.f_joint, slices[li], depths, method="euler")
                traj_g = odeint(trainer.g_joint, slices[hj], depths.flip(0), method="euler")
                f_t = traj_f[1].detach().cpu().numpy()
                g_t = traj_g[1].detach().cpu().numpy()
                del traj_f, traj_g

                gen = np.vstack([f_t, g_t])
                tree = cKDTree(gen[:, :2])
                qxy = fit.xy(bins["x"][rows_all], bins["y"][rows_all])
                dist, nn = tree.query(qxy, k=1)
                P = inv_features(gen[nn, 3:], td)
                section_pred[tsec] = (rows_all, P)
                map_stats[tsec] = dict(n_bins=int(rows_all.size), n_gen=int(gen.shape[0]),
                                       nn_um_median=float(np.median(dist) * fit.W),
                                       nn_um_p90=float(np.percentile(dist, 90) * fit.W),
                                       nn_um_max=float(dist.max() * fit.W))
                log(f"  section {tsec}: bins={rows_all.size} gen={gen.shape[0]} "
                    f"nn um med={np.median(dist)*fit.W:.1f} p90={np.percentile(dist,90)*fit.W:.1f}")
    return section_pred, map_stats, round(time.time() - infer_t0, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--scenarios", default=None,
                    help="comma separated scenario names (default: all)")
    ap.add_argument("--epochs", default="60,60,60")
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    gpu_idx = str(args.gpu).split(",")[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    epochs = [int(x) for x in args.epochs.split(",")]
    if args.pilot:
        epochs = [int(min(e, 2)) for e in epochs]

    status = C.Status(OUT / "status.json", METHOD,
                      dict(pilot=bool(args.pilot), config=vars(args), epochs_list=epochs,
                           delta_d=DELTA_D, scale_z_per_um=SCALE_Z, log=str(LOG)))
    mon = C.GpuSampler(); mon.start()

    log(f"START pilot={args.pilot} epochs={epochs} gpu={gpu_idx}")
    C.wait_ready()
    bins = C.load_bins()
    scs = C.load_scenarios()
    tg = C.load_target_gt()
    log(f"loaded bins X={bins['X'].shape} sections={sorted(set(bins['section']))} "
        f"genes={bins['genes'].size} scenarios={len(scs)}")

    if args.scenario:
        scs = [s for s in scs if s["name"] == args.scenario]
    if args.scenarios:
        keep = set(args.scenarios.split(","))
        scs = [s for s in scs if s["name"] in keep]
    if args.pilot:
        scs = scs[:1]
    status.update(status="RUNNING", stage="loaded", scenarios_total=len(scs),
                  n_bins=int(bins["X"].shape[0]), n_genes=int(bins["X"].shape[1]))

    metrics = {"method": METHOD, "dataset": "merfish_5slice_50um", "voxel_um": 50,
               "pilot": bool(args.pilot), "epochs_list": epochs,
               "delta_d": DELTA_D, "scale_z_per_um": SCALE_Z,
               "feature_norm": "per-gene min-max fit on visible sections (no log1p, no PCA)",
               "mapping": "per hidden block: euler f from lower + g from upper visible slice; "
                          "per target section 2D NN in normalised xy",
               "scenarios": []}

    failures = []
    for i, sc in enumerate(scs):
        t0 = time.time()
        status.update(stage=f"train:{sc['name']}", current=sc["name"], scenario_index=i)
        exp_dir = OUT / "experiments" / sc["name"]
        try:
            trained = train_scenario(bins, sc, epochs, exp_dir)
            log(f"[{sc['name']}] trained visible={sc['visible']} target={sc['target']} "
                f"{trained['train_sec']}s params={trained['n_params']}")
            status.update(stage=f"predict:{sc['name']}")
            section_pred, map_stats, infer_sec = predict(bins, sc, trained)
            rows = C.scenario_rows(tg, sc)
            pred = C.align_prediction(tg, rows, section_pred, bins)
            path = C.save_pred(METHOD, sc, tg, rows, pred,
                               extra=dict(genes=bins["genes"]))
            rec = dict(scenario=sc["name"], k=len(sc["target"]), visible=sc["visible"],
                       target=sc["target"], n_rows=int(rows.size),
                       train_sec=trained["train_sec"], infer_sec=infer_sec,
                       total_sec=round(time.time() - t0, 1),
                       pred_min=float(pred.min()), pred_max=float(pred.max()),
                       pred_mean=float(pred.mean()), pred_zero_frac=float((pred == 0).mean()),
                       map=map_stats, output=str(path))
            metrics["scenarios"].append(rec)
            log(f"[{sc['name']}] DONE rows={rows.size} total={rec['total_sec']}s "
                f"min/max/mean={rec['pred_min']:.4f}/{rec['pred_max']:.4f}/{rec['pred_mean']:.4f}")
            del trained
            import torch
            torch.cuda.empty_cache()
        except Exception as e:
            tb = traceback.format_exc()
            log(f"[{sc['name']}] FAILED {e!r}\n{tb}")
            failures.append(dict(scenario=sc["name"], error=repr(e), traceback=tb[-2500:]))
            status.update(stage=f"failed:{sc['name']}", last_error=repr(e),
                          failures=failures)
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        status.update(scenarios_done=i + 1, gpu_peak_mb=int(mon.peak),
                      failures=len(failures))

    mon.stop.set(); mon.join(timeout=5)
    metrics["gpu_peak_mb"] = int(mon.peak)
    metrics["n_done"] = len(metrics["scenarios"])
    metrics["n_failed"] = len(failures)
    metrics["failures"] = failures
    C.write_json(OUT / "compute_metrics.json", metrics)
    ok = len(failures) == 0
    status.update(status="DONE" if ok else "PARTIAL", stage="finished",
                  gpu_peak_mb=int(mon.peak), n_done=len(metrics["scenarios"]),
                  n_failed=len(failures), metrics=str(OUT / "compute_metrics.json"))
    log(f"FINISHED done={metrics['n_done']} failed={len(failures)} gpu_peak={mon.peak}MiB")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

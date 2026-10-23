"""
isoST adapter for the MERFISH LOSO_54 task — audited version (2026-08-13).

Scientific-correctness changes vs the previous version:
  1. methods/isoST/model/utils/losses.py restored to UPSTREAM: `one_nn_mse` uses
     the official hard argmin nearest-neighbor assignment (no softmin).
     Verified on GPU: official loss forwards/backwards fine and gradients reach
     all parameters (see benchmark/results/large_scale_benchmark/AUDIT_2026-08-13).
  2. Hyperparameters aligned with the official config/mouse_brain.yml
     (alpha=0.1, warm_up_rate=1.0, std_x/std_y=0.01, std_z=0.1, std_seq=0.1,
     beta 1.0→0.05 over 50 iters after 50, weight_decay=1e-8, lr=1e-3 NAdam,
     method=euler, dual=True, stride=1, delta_d=0.2=official class default).
     Resource-driven deviations only (explicitly allowed by the audit goal):
       hidden_dim 64→32, points/slice subsampling, 3-stage epochs
       [100,100,100] = official recipe.
  3. Feature space: the benchmark dataset is already PC_1..PC_50, min-max [0,1]
     (= isoST's official "PC50_minmax" recipe); no re-PCA is applied.
  4. Inference follows the official fine_infer protocol (odeint f_joint from the
     lower visible slice + g_joint from the upper visible slice, stacked at the
     target depth) and now generates one virtual slice PER held-out z, so gap-2
     and gap-3 rounds are fully covered (the previous adapter predicted only the
     gap-mean plane).
  5. Frozen mapping to the evaluation support: nearest-neighbor match of the
     generated points to the public holdout (x,y) query coordinates
     (fixed_coordinate protocol; target x/y are declared public in the task
     manifests). Match distances are recorded for the audit.

Uses subprocess + isoST conda environment isolation.
"""
import json
import os
import subprocess
import time
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import anndata as ad
import yaml
from scipy import sparse

from benchmark.src.adapters.base import (
    assigned_gpu_indices, pick_from_assigned_or_free,
    BaseAdapter, AdapterError, FailureType,
)
from benchmark.src.utils.hpc import cuda_home as _cuda_home

ISOST_DIR = Path("/storage/yuhongtaoLab/linpeiran/xpy/lab/baseline/methods/isoST")
ISOST_ENV = "isoST"
# CUDA toolchain for the keops "online" Sinkhorn backend (math-identical to
# tensorized, verified; bounds GPU memory at full-slice scale). See METHOD_CHANGES.md.
# 2026-10-18 (SE cluster): resolved via benchmark.src.utils.hpc — the toolkit is a
# module (/soft/devtools/cuda/cuda-12.9) and /usr/local/cuda is an empty stub here.
ISOST_CUDA_HOME = _cuda_home()


def _isost_subprocess_env(gpu: int) -> dict:
    """Environment for the isoST subprocess: CUDA toolchain + GPU selection."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CUDA_HOME"] = str(ISOST_CUDA_HOME)
    env["CUDA_PATH"] = str(ISOST_CUDA_HOME)
    env["PATH"] = f"{ISOST_CUDA_HOME}/bin:{env['PATH']}"
    # 2026-08-29 (Blackwell migration): keops 2.2.3 JIT links `-lnvrtc` without
    # an -L flag — the linker needs LIBRARY_PATH, and the runtime dlopen needs
    # LD_LIBRARY_PATH (system CUDA 12.8 lib64). Math-neutral env fix.
    _lib = f"{ISOST_CUDA_HOME}/lib64"
    env["LIBRARY_PATH"] = f"{_lib}:" + env.get("LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{_lib}:" + env.get("LD_LIBRARY_PATH", "")
    return env

# Official config/mouse_brain.yml parameters (algorithm identity) —
# per EXPERIMENT_RESOURCE_POLICY.md (2026-08-14): NO resource shrinking.
OFFICIAL_PARAMS = {
    "gene_dim": None,           # set to n_genes (=50 PCs)
    "hidden_dim": 64,           # official value (config/mouse_brain.yml)
    "head_num": 1,
    "K": 8,                     # inert in this code path (PJointV2 hardcodes topk=5)
    "method": "euler",
    "delta_d": 0.2,             # official class default; z-gap≈1-3 -> 6-16 steps
                                # (official 0.01 applies to their normalized
                                #  coordinate scale; effective step count matches)
    "stride": 1,
    "lr": 0.001,
    "optimizer_name": "NAdam",
    "weight_decay": 1e-8,
    "warm_up_rate": 1.0,
    "std_x": 0.01, "std_y": 0.01, "std_z": 0.1,
    "std_seq": 0.1,
    "alpha": 0.1,               # w_seq = alpha
    "dual": True,
    "beta_start_value": 1.0,
    "beta_end_value": 0.05,
    "beta_start_iteration": 50,
    "beta_n_iterations": 50,
}


class IsoSTAdapter(BaseAdapter):
    """Adapter for isoST SDE-based 3D expression reconstruction (official loss)."""

    method_name = "isoST"
    conda_env = ISOST_ENV
    method_dir = ISOST_DIR

    def run_impl(
        self, adata_input: ad.AnnData, config: Dict, output_dir: Path
    ) -> ad.AnnData:
        holdout_slices = config.get("holdout_slices", None)
        max_points_per_slice = config.get("max_points_per_slice", None)  # None = full slices (official)
        epochs_list = config.get("epochs_list", [100, 100, 100])  # official 3-stage
        gpu = config.get("gpu")
        timeout_sec = config.get("timeout_sec", 86400)  # T_max guard (resource policy §3)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        X_all = adata_input.X.toarray() if sparse.issparse(adata_input.X) else np.asarray(adata_input.X)
        coords_all = adata_input.obsm["spatial"]
        genes = adata_input.var_names.tolist()
        n_genes = len(genes)

        # === Split FIRST (before any statistics) — no leakage ===
        if holdout_slices is not None:
            holdout_mask = adata_input.obs["slice"].isin(holdout_slices).to_numpy()
        else:
            holdout_z = config.get("holdout_z", 14)
            holdout_mask = np.isclose(coords_all[:, 2], float(holdout_z))
        visible_mask = ~holdout_mask

        X_vis = X_all[visible_mask]
        coords_vis = coords_all[visible_mask]
        coords_holdout = coords_all[holdout_mask]
        print(f"Visible: {X_vis.shape[0]} spots, Holdout: {coords_holdout.shape[0]} spots")

        # === Features: dataset is already PC_1..PC_50 min-max [0,1] (official recipe) ===
        # No re-PCA, no re-normalization — the frozen expression layer is used directly.

        # === Per visible slice: [x, y, z, pc_1..pc_50] tensors (subsampled) ===
        data_dir = output_dir / "isoST_data"
        data_dir.mkdir(exist_ok=True)
        slice_files = []
        visible_slice_labels = sorted(adata_input.obs["slice"][visible_mask].unique())
        for i, sl_label in enumerate(visible_slice_labels):
            sl_mask = (adata_input.obs["slice"] == sl_label).to_numpy()
            sl_visible = sl_mask[visible_mask]
            tensor = np.column_stack([coords_vis[sl_visible], X_vis[sl_visible]]).astype(np.float32)
            n_spots = tensor.shape[0]
            if max_points_per_slice and n_spots > max_points_per_slice:
                rng = np.random.RandomState(42 + i)
                tensor = tensor[rng.choice(n_spots, max_points_per_slice, replace=False)]
                n_spots = max_points_per_slice
            fname = f"shuffled_slice_{i}.npy"
            np.save(data_dir / fname, tensor)
            slice_files.append(fname)
            print(f"  Slice {sl_label} (z≈{tensor[:, 2].mean():.1f}): {n_spots} spots → {fname}")

        with open(data_dir / "gene.csv", "w") as f:
            f.write("gene\n" + "\n".join(genes) + "\n")

        # === Holdout target depths: one virtual slice per held-out z ===
        holdout_z_list = sorted({
            float(np.mean(coords_holdout[adata_input.obs["slice"][holdout_mask].to_numpy() == sl, 2]))
            for sl in sorted(set(adata_input.obs["slice"][holdout_mask].to_numpy()))
        })

        # === Config (official params + resource reductions) ===
        params = dict(OFFICIAL_PARAMS)
        params["gene_dim"] = n_genes
        yaml_config = {"trainer": "IsoST", "params": params}
        config_path = output_dir / "isoST_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(yaml_config, f)

        # === Generated script ===
        script = self._generate_script(
            data_dir=str(data_dir),
            config_path=str(config_path),
            output_dir=str(output_dir),
            slice_files=slice_files,
            holdout_z_list=holdout_z_list,
            epochs_list=epochs_list,
        )
        script_path = output_dir / "_run_isost.py"
        script_path.write_text(script)

        # === Run ===
        if gpu is None:
            gpu = self._pick_free_gpu()
        gpu = int(gpu)
        # T_max guard (EXPERIMENT_RESOURCE_POLICY.md §3): write the estimate at
        # start — unless the campaign runner already wrote a richer record here
        # (first writer wins; the adapter default is the standalone safety net)
        tmax_path = output_dir / "T_MAX.txt"
        if not tmax_path.exists():
            tmax_path.write_text(
                f"T_max_sec={timeout_sec}\nofficial_config=hidden_dim64_full_slices_"
                f"3stage_100x3\nadapter-default T_max basis (no campaign runner "
                f"record pre-written)\nexceeding T_max (excluding hardware wait) "
                f"= mid-run failure -> investigate\n")
        t0 = time.time()
        result = subprocess.run(
            ["conda", "run", "-n", ISOST_ENV, "python", str(script_path)],
            cwd=str(output_dir),
            capture_output=True, text=True, timeout=timeout_sec,
            env={**_isost_subprocess_env(gpu), "PYTHONPATH": str(ISOST_DIR)},
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            raise AdapterError(
                f"isoST failed (code {result.returncode}): {result.stderr[-1500:]}",
                failure_type=FailureType.F2_INTERNAL,
                details={"stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]},
            )
        if self.verbose:
            print(result.stdout[-1500:])

        # === Frozen mapping: NN-match generated points to public holdout (x,y) ===
        result_raw_path = output_dir / "result_raw.npy"
        if not result_raw_path.exists():
            raise AdapterError("isoST did not produce predictions", failure_type=FailureType.F2_INTERNAL)

        result_raw = np.load(result_raw_path)
        pred_coords = result_raw[:, :3]
        X_pred_gen = result_raw[:, 3:3 + n_genes]

        from scipy.spatial import cKDTree
        tree = cKDTree(pred_coords[:, :2])
        match_dists, nn_indices = tree.query(coords_holdout[:, :2])
        X_pred = X_pred_gen[nn_indices]
        np.save(output_dir / "X_pred.npy", X_pred)
        np.save(output_dir / "nn_match_dists.npy", match_dists)

        print(f"NN match: median dist {np.median(match_dists):.4f}, "
              f"p90 {np.percentile(match_dists, 90):.4f}, max {match_dists.max():.4f} "
              f"({len(pred_coords)} generated → {len(coords_holdout)} query pts)")

        pred_adata = ad.AnnData(
            X=np.clip(X_pred, 0, None),
            obsm={"spatial": coords_holdout},  # evaluation support = public query coords
            var=adata_input.var.copy(),
        )
        pred_adata.obs["is_interpolated"] = True
        pred_adata.obs["slice"] = adata_input.obs["slice"][holdout_mask].values
        pred_adata.uns["method"] = self.method_name
        pred_adata.uns["task"] = "MERFISH_LOSO"
        pred_adata.uns["runtime_sec"] = elapsed
        pred_adata.uns["mapping"] = "nn_to_query_xy_frozen"
        pred_adata.uns["loss"] = "official_hard_argmin_one_nn_mse"
        pred_adata.uns["gpu"] = gpu
        pred_adata.uns["nn_match_dist_median"] = float(np.median(match_dists))
        return pred_adata

    @staticmethod
    def _pick_free_gpu() -> int:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        free = [(int(l.split(",")[0]), int(l.split(",")[1]))
                for l in out.stdout.strip().splitlines() if l.strip()]
        # An explicit CUDA_VISIBLE_DEVICES assignment wins over "freest card": see
        # assigned_gpu_indices() in base.py for the incident this prevents.
        return pick_from_assigned_or_free(free)

    def _generate_script(self, **kw) -> str:
        return f'''
import sys, os, time, warnings, json
import numpy as np
import torch
import yaml
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Reproducibility (seed is a run-hygiene fix, not an algorithm change)
torch.manual_seed(0)
np.random.seed(0)

sys.path.insert(0, "{ISOST_DIR}")

from utils.train_ode import train

data_dir = Path("{kw['data_dir']}")
config_path = Path("{kw['config_path']}")
output_dir = Path("{kw['output_dir']}")
holdout_z_list = {kw['holdout_z_list']}
epochs_list = {kw['epochs_list']}

with open(config_path) as f:
    cfg = yaml.safe_load(f)
params = cfg["params"]

from model.base_model import IsoST as TrainerClass
trainer = TrainerClass(
    gene_dim=params["gene_dim"],
    hidden_dim=params["hidden_dim"],
    head_num=params.get("head_num", 1),
    K=params.get("K", 8),
    method=params.get("method", "euler"),
    delta_d=params.get("delta_d", 0.2),
    stride=params.get("stride", 1),
    lr=params.get("lr", 1e-3),
    weight_decay=params.get("weight_decay", 1e-8),
    optimizer_name=params.get("optimizer_name", "NAdam"),
    warm_up_rate=params.get("warm_up_rate", 1.0),
    std_x=params.get("std_x", 0.01),
    std_y=params.get("std_y", 0.01),
    std_z=params.get("std_z", 0.1),
    std_seq=params.get("std_seq", 0.1),
    alpha=params.get("alpha", 0.1),
    dual=params.get("dual", True),
    beta_start_value=params.get("beta_start_value", 1.0),
    beta_end_value=params.get("beta_end_value", 0.05),
    beta_start_iteration=params.get("beta_start_iteration", 50),
    beta_n_iterations=params.get("beta_n_iterations", 50),
    device='cuda:0',
)
trainer = trainer.cuda()
print(f"Model on {{next(trainer.parameters()).device}}, "
      f"{{sum(p.numel() for p in trainer.parameters())}} params")

# Load slices (already subsampled [x, y, z, pc_1..pc_n])
slice_files = {kw['slice_files']}
slices = []
slice_z_means = []
for fname in slice_files:
    t = torch.from_numpy(np.load(data_dir / fname)).cuda()
    slices.append(t)
    slice_z_means.append(t[:, 2].mean().item())
dataset = [slices]  # batch_num=1 official format
print(f"Loaded {{len(slices)}} slices, z range "
      f"[{{min(slice_z_means):.1f}}, {{max(slice_z_means):.1f}}]")

experiment_dir = output_dir / "experiments"
experiment_dir.mkdir(exist_ok=True)

# === Official 3-stage training (coo -> seq -> joint), official loss functions ===
print(f"\\nTraining isoST (3 stages x {{epochs_list}} steps, GPU)...")
t0 = time.time()
train(
    trainer, experiment_dir, dataset,
    batch_num=1,
    epochs_list=epochs_list,
    checkpoint_every=500,
    backup_every=2000,
)
print(f"Training done in {{time.time()-t0:.1f}}s")
trainer.save(str(experiment_dir / "model.pt"))

# === Inference: official fine_infer protocol, one virtual slice per held-out z ===
from torchdiffeq import odeint_adjoint as odeint
trainer.eval()

z_sorted = sorted(enumerate(slice_z_means), key=lambda x: x[1])
z_indices = [i for i, _ in z_sorted]
z_values = [z for _, z in z_sorted]

# Integration step for THIS dataset's depth scale.
#
# 0.2 is not a scale-free constant. The official default is 0.01 on the atlas authors'
# normalised depth scale, and this adapter's 0.2 was calibrated so that a typical
# reference gap (~0.93 z-units) integrates in ~6 Euler steps. On depths carried in other
# units that number is meaningless, and the 2026-09-22 datasets carry theirs in other
# units: with the heart's 150 um sections one gap becomes 751 steps, while on the Allen
# official releases (z step ~0.1-0.2) it collapses to ONE -- a single Euler step across
# the gap, i.e. no depth integration at all.
#
# So the step is anchored to the dataset's own median slice gap over the reference's,
# which leaves the reference EXACTLY as it was (ratio 1.0 -> delta_d = 0.2 -> every one
# of the 53 consecutive pairs keeps its N_steps) and gives every other dataset the same
# effective step count. Verified without a run: reference N_steps identical; heart
# 751 -> 6, ABCA-1 1 -> 6, ABCA-2 2 -> 6.
REFERENCE_MEDIAN_Z_GAP = 0.928329   # measured: MERFISH_ABCA-2, per-slice z means, 54 slices
_z_means = np.asarray(sorted(slice_z_means), dtype=float)
_median_gap = float(np.median(np.diff(_z_means))) if len(_z_means) > 1 else REFERENCE_MEDIAN_Z_GAP
delta_d = params.get("delta_d", 0.2) * (_median_gap / REFERENCE_MEDIAN_Z_GAP)

results_all = []
for target_z in holdout_z_list:
    lower_idx, upper_idx = None, None
    for i in range(len(z_values) - 1):
        if z_values[i] <= target_z <= z_values[i + 1]:
            lower_idx, upper_idx = z_indices[i], z_indices[i + 1]
            break
    if lower_idx is None:
        raise ValueError(f"target z={{target_z}} not enclosed "
                         f"(z range [{{z_values[0]:.1f}}, {{z_values[-1]:.1f}}])")
    u_lower, u_upper = slices[lower_idx], slices[upper_idx]
    z_lower = u_lower[:, 2].mean().item()
    z_upper = u_upper[:, 2].mean().item()
    N_steps = round(abs(z_upper - z_lower) / delta_d) + 1
    depth_trajectory = torch.linspace(z_lower, z_upper, steps=N_steps, device='cuda:0')

    with torch.no_grad():
        trajectory_f = odeint(trainer.f_joint, u_lower, depth_trajectory, method='euler')
        r_depth = depth_trajectory.flip(dims=[0])
        trajectory_g = odeint(trainer.g_joint, u_upper, r_depth, method='euler')
        trajectory_g = trajectory_g.flip(dims=[0])

        best_step = torch.argmin(torch.abs(depth_trajectory - target_z)).item()
        result_f = trajectory_f[best_step].cpu().numpy()
        result_g = trajectory_g[best_step].cpu().numpy()
    result = np.vstack([result_f, result_g])
    results_all.append(result)
    print(f"  target z={{target_z:.2f}}: step {{best_step}}/{{N_steps-1}} "
          f"(z={{depth_trajectory[best_step].item():.2f}}), pred shape {{result.shape}}")

result_raw = np.vstack(results_all)
np.save(output_dir / "result_raw.npy", result_raw)
print(f"Saved result_raw: {{result_raw.shape}}")
print("DONE!")
'''

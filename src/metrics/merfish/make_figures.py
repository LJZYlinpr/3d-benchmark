#!/usr/bin/env python
"""Phase 6: figures for the MERFISH 5-slice metric protocol v1."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlib as M

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
METHODS = ["NTF", "Spateo"]
COLORS = {"NTF": "#C0504D", "Spateo": "#2F5F98", "GT": "#333333"}
COMBO_ORDER = ["M1a", "M1b", "M1c", "M2a", "M2b", "M2c", "M3a"]
N_MISSING = {"M1a": 1, "M1b": 1, "M1c": 1, "M2a": 2, "M2b": 2, "M2c": 2, "M3a": 3}
SHORT = {
    "Bergmann glial cell": "Bergmann glia",
    "GABAergic neuron": "GABAergic",
    "choroid plexus epithelial cell": "Choroid plexus",
    "dopaminergic neuron": "Dopaminergic",
    "endothelial cell": "Endothelial",
    "ependymal cell": "Ependymal",
    "glutamatergic neuron": "Glutamatergic",
    "glycinergic neuron": "Glycinergic",
    "macrophage": "Macrophage",
    "microglial cell": "Microglial",
    "neuroblast (sensu Vertebrata)": "Neuroblast",
    "olfactory ensheathing cell": "Olf. ensheathing",
    "oligodendrocyte": "Oligodendrocyte",
    "oligodendrocyte precursor cell": "OPC",
    "pericyte": "Pericyte",
    "smooth muscle cell": "Smooth muscle",
    "vascular leptomeningeal cell": "Vasc. leptomeningeal",
}


def short(n):
    return SHORT.get(n, n)


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_heatmap(core):
    pool = core[core["scope"] == "pooled"]
    pv = pool.pivot_table(index="module", columns="method", values="module_pcc",
                          aggfunc="mean")
    order = pv.mean(1).sort_values(ascending=False).index
    pv = pv.loc[order, METHODS]
    fig, ax = plt.subplots(figsize=(4.6, 7.2))
    im = ax.imshow(pv.values, cmap="RdYlBu_r", vmin=-0.2, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(METHODS)), METHODS)
    ax.set_yticks(range(len(pv)), [short(i) for i in pv.index])
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.25 or v > 0.75 else "black")
    ax.set_title("Module PCC (mean over 7 combos, pooled target bins)")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    save(fig, "fig_module_pcc_heatmap.png")


def fig_combo_response(core):
    pool = core[core["scope"] == "pooled"]
    metrics = [("module_pcc", "Module PCC"), ("ms_ssim", "MS-SSIM"),
               ("var_ratio_log2", "log2 variance ratio"),
               ("abs_delta_moran_i", "|Δ Moran's I|"),
               ("hotspot_recall", "Hotspot recall"),
               ("hotspot_precision", "Hotspot precision"),
               ("boundary_recall_d0", "Boundary recall (δ=0)"),
               ("boundary_precision_d0", "Boundary precision (δ=0)")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), sharex=True)
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        for method in METHODS:
            sub = pool[pool["method"] == method].groupby("combo")[metric].mean()
            xs = [COMBO_ORDER.index(c) for c in sub.index]
            ax.plot(xs, sub.values, "o-", color=COLORS[method], label=method,
                    lw=1.8, ms=4)
        ax.set_xticks(range(len(COMBO_ORDER)), COMBO_ORDER, rotation=45)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.25)
        if metric == "var_ratio_log2":
            ax.axhline(0, color="grey", lw=0.8, ls="--")
        if metric in ("module_pcc", "ms_ssim", "hotspot_recall", "hotspot_precision"):
            ax.set_ylim(-0.15, 1.0)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Core metrics vs missing-slice scenario (mean over 18 modules; 5-slice testbed)",
                 fontsize=12)
    fig.tight_layout()
    save(fig, "fig_combo_response.png")


def fig_perturbation(curves):
    c = curves[curves["combo"] == "ALL"]

    def series(cond, metric, level=None):
        d = c[c["condition"] == cond]
        if level is not None:
            d = d[d["level"] == level]
        d = d[d["metric"] == metric].sort_values("level")
        return d["level"].values, d["mean"].values

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax = axes[0, 0]
    for metric, lab in [("module_pcc", "module PCC"), ("ms_ssim", "MS-SSIM"),
                        ("hotspot_f1", "hotspot F1"),
                        ("boundary_f1_d1", "boundary F1 (δ=1)")]:
        x, y = series("blur", metric)
        ax.plot(x, y, "o-", label=lab, lw=1.6, ms=4)
    ax.set_title("Gaussian blur (σ bins)")
    ax.set_xlabel("σ (bins)")
    ax.set_ylim(-0.05, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    for metric, lab in [("module_pcc", "module PCC"), ("ms_ssim", "MS-SSIM"),
                        ("hotspot_f1", "hotspot F1"),
                        ("boundary_f1_d1", "boundary F1 (δ=1)")]:
        x, y = series("shift", metric)
        ax.plot(x, y, "o-", label=lab, lw=1.6, ms=4)
    ax.set_title("Spatial shift (bins)")
    ax.set_xlabel("shift (bins)")
    ax.set_ylim(-0.05, 1.02)
    ax.grid(alpha=0.25)

    ax = axes[0, 2]
    for metric, lab in [("module_pcc", "module PCC"), ("ms_ssim", "MS-SSIM"),
                        ("hotspot_f1", "hotspot F1"),
                        ("boundary_f1_d1", "boundary F1 (δ=1)")]:
        x, y = series("correlated_noise", metric)
        ax.plot(x, y, "o-", label=lab, lw=1.6, ms=4)
    ax.set_title("Correlated noise (σ_bins=1.5, matched variance)")
    ax.set_xlabel("noise std / GT std")
    ax.set_ylim(-0.05, 1.02)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    conds = ["identity", "add_hotspot", "merge_components", "split_components"]
    labs = ["identity", "add hotspot", "merge", "split"]
    mets = [("module_pcc", "PCC"), ("hotspot_precision", "precision"),
            ("hotspot_recall", "recall"), ("hotspot_AP", "AP")]
    w = 0.2
    for k, (metric, lab) in enumerate(mets):
        vals = [c[(c["condition"] == cd) & (c["metric"] == metric)]["mean"].mean()
                for cd in conds]
        ax.bar(np.arange(len(conds)) + (k - 1.5) * w, vals, w, label=lab)
    ax.set_xticks(range(len(conds)), labs)
    ax.set_title("Hotspot perturbations")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    conds = ["identity", "add_boundary_band"]
    labs = ["identity", "added boundary band"]
    mets = [("boundary_recall_d1", "recall"), ("boundary_precision_d1", "precision"),
            ("boundary_f1_d1", "F1"), ("abs_delta_moran_i", "|Δ Moran|")]
    for k, (metric, lab) in enumerate(mets):
        vals = [c[(c["condition"] == cd) & (c["metric"] == metric)]["mean"].mean()
                for cd in conds]
        ax.bar(np.arange(len(conds)) + (k - 1.5) * w, vals, w, label=lab)
    ax.set_xticks(range(len(conds)), labs)
    ax.set_title("Boundary perturbation (δ=1)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 2]
    ax.axis("off")
    ax.text(0.02, 0.95, "Metric response summary (mean over GT module maps)\n",
            fontsize=10, va="top", fontweight="bold")
    lines = []
    for metric, lab in [("module_pcc", "module PCC"), ("ms_ssim", "MS-SSIM"),
                        ("var_ratio_log2", "log2 var ratio"),
                        ("abs_delta_moran_i", "|Δ Moran's I|"),
                        ("hotspot_AP", "hotspot AP"),
                        ("boundary_f1_d1", "boundary F1")]:
        v = c[(c["condition"] == "identity") & (c["metric"] == metric)]["mean"].values
        lines.append(f"{lab:16s} identity = {v[0]:.3f}" if len(v) else lab)
    lines += ["", "Response direction tests:", 
              "blur: PCC/MS-SSIM/F1 should fall with σ",
              "shift: PCC/MS-SSIM fall; boundary F1 falls",
              "noise: all agreement metrics fall",
              "add hotspot: precision (and AP) falls",
              "add boundary: boundary precision/F1 falls",
              "merge/split: hotspot component recall/precision change"]
    ax.text(0.02, 0.86, "\n".join(lines), fontsize=8.5, va="top", family="monospace")
    fig.suptitle("Perturbation validation on held-out GT module maps (7 combos x 18 modules)",
                 fontsize=12)
    fig.tight_layout()
    save(fig, "fig_perturbation_curves.png")


def fig_hotspot_boundary(pr):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    d = pr.groupby(["combo", "method"])[["hotspot_AP"]].mean().reset_index()
    for k, method in enumerate(METHODS):
        sub = d[d["method"] == method].set_index("combo").loc[COMBO_ORDER]
        ax.bar(np.arange(len(COMBO_ORDER)) + (k - 0.5) * 0.38, sub["hotspot_AP"],
               0.38, color=COLORS[method], label=method)
    ax.set_xticks(range(len(COMBO_ORDER)), COMBO_ORDER, rotation=45)
    ax.set_title("Hotspot AP (threshold-free)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    d = pr.groupby(["combo", "method"])[["boundary_f1_d1"]].mean().reset_index()
    for k, method in enumerate(METHODS):
        sub = d[d["method"] == method].set_index("combo").loc[COMBO_ORDER]
        ax.bar(np.arange(len(COMBO_ORDER)) + (k - 0.5) * 0.38, sub["boundary_f1_d1"],
               0.38, color=COLORS[method], label=method)
    ax.set_xticks(range(len(COMBO_ORDER)), COMBO_ORDER, rotation=45)
    ax.set_title("Boundary F1 (δ=1 bin)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    for k, method in enumerate(METHODS):
        sub = pr[pr["method"] == method]
        ax.scatter(sub["sens_p80_precision"], sub["sens_p80_recall"], s=8,
                   alpha=0.45, color=COLORS[method], label=f"{method} p80")
        ax.scatter(sub["sens_p95_precision"], sub["sens_p95_recall"], s=8,
                   alpha=0.8, color=COLORS[method], marker="^", label=f"{method} p95")
    ax.set_xlabel("hotspot precision")
    ax.set_ylabel("hotspot recall")
    ax.set_title("Threshold sensitivity (p80 vs p95)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6)
    fig.tight_layout()
    save(fig, "fig_hotspot_boundary.png")


def fig_example(combo="M2b", module_name="GABAergic neuron"):
    import anndata as ad
    with open(os.path.join(HERE, "metric_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    modules = {m["name"]: m for m in M.load_module_defs()["modules"]}
    obs = ad.read_h5ad(M.OBS, backed="r")
    ref = M.reference_stats(obs)
    tg = M.load_target(combo)
    Xg = np.asarray(tg["X"], dtype=np.float64)
    xy = np.asarray(tg["coords"], dtype=np.float64)[:, :2]
    sl = tg["slice"].astype(str)
    s = sorted(set(sl))[0]
    sel = sl == s
    m = modules[module_name]
    idx = np.asarray(m["gene_idx"], dtype=np.int64)
    sg, _ = M.module_scores(Xg[sel], idx, ref["mean"], ref["std"], ref["valid"])
    A_gt, mask, _ = M.rasterize(xy[sel], sg)
    thr = cfg["modules"]["hotspot_thresholds_reference"][module_name][
        "hotspot_threshold_p90"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for r, method in enumerate(METHODS):
        p = M.load_pred(combo, method)
        Xp = np.asarray(p["X"], dtype=np.float64)[sel]
        sp, _ = M.module_scores(Xp, idx, ref["mean"], ref["std"], ref["valid"])
        A_p, _, _ = M.rasterize(xy[sel], sp)
        vmin, vmax = np.nanpercentile(A_gt[mask], [1, 99])
        for c, (A, title) in enumerate([(A_gt, "GT"), (A_p, f"{method}")]):
            im = axes[r, c].imshow(A, cmap="magma", vmin=vmin, vmax=vmax)
            axes[r, c].set_title(f"{title} | {short(module_name)} | {combo} {s[-3:]}")
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            fig.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.02)
        err = A_p - A_gt
        v = np.nanpercentile(np.abs(err[mask]), 98)
        im = axes[r, 2].imshow(err, cmap="coolwarm", vmin=-v, vmax=v)
        axes[r, 2].set_title(f"error (pred - GT) | RMSE={np.sqrt(np.nanmean(err[mask]**2)):.3f}")
        axes[r, 2].set_xticks([])
        axes[r, 2].set_yticks([])
        fig.colorbar(im, ax=axes[r, 2], fraction=0.046, pad=0.02)
    fig.suptitle(f"Example module map: {module_name}, {combo} target section {s} "
                 f"(hotspot t90={thr:.3f})", fontsize=11)
    fig.tight_layout()
    save(fig, "fig_example_maps.png")


def fig_reliability(rel, core):
    pool = core[core["scope"] == "pooled"]
    ceiling = rel.groupby("module")["split_half_pcc"].mean()
    pcc = pool.groupby(["module", "method"])["module_pcc"].mean().unstack()
    order = ceiling.sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(order))
    ax.barh(y + 0.18, ceiling.loc[order].values, 0.36, color=COLORS["GT"],
            label="GT split-half reliability (ceiling)")
    for k, method in enumerate(METHODS):
        ax.barh(y + (k - 0.5) * 0.36, pcc[method].loc[order].values, 0.36,
                color=COLORS[method], alpha=0.85, label=method)
    ax.set_yticks(y, [short(i) for i in order])
    ax.set_xlabel("module PCC")
    ax.set_xlim(-0.3, 1.02)
    ax.axvline(0, color="grey", lw=0.8)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Reliability ceiling vs method module PCC (mean over 7 combos)")
    fig.tight_layout()
    save(fig, "fig_reliability_ceiling.png")


def main():
    core = pd.read_csv(os.path.join(HERE, "metrics_core.csv"))
    pr = pd.read_csv(os.path.join(HERE, "metrics_spatial_pr.csv"))
    curves = pd.read_csv(os.path.join(HERE, "perturbation_curves.csv"))
    rel = pd.read_csv(os.path.join(HERE, "gt_split_half_reliability.csv"))
    fig_heatmap(core)
    fig_combo_response(core)
    fig_perturbation(curves)
    fig_hotspot_boundary(pr)
    fig_example()
    fig_reliability(rel, core)
    print("figures done")


if __name__ == "__main__":
    main()

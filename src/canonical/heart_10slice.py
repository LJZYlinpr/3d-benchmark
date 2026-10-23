#!/usr/bin/env python3
"""heart-infra step 1: canonical bins, sections, scenarios, target GT, modules.

Single sequential pass over the source h5ad (no GPU, no PCA, no dense volume).

Outputs
  canonical/bins_all.h5ad      10 sections (21-30), 50 um bins, mean X, n_cells
  canonical/sections.json
  canonical/scenarios.json     26 contiguous hidden blocks k=1..4, 60 target slices
  canonical/target_gt.npz      scenario, section, bin_id, x, y, gt (eval only)
  modules/heart_modules.json   12 class-level disjoint modules + Other
  modules/reference_stats.npz  gene mean/std from sections 1-20 + 31-53 only
"""
import json
import os
import sys
import time
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SRC = "/storage/yuhongtaoLab/linpeiran/xpy/data/data/dryad/Fig4to6_3DHeart.h5ad"
ROOT = Path("/storage/yuhongtaoLab/linpeiran/projects/3d_benchmark/Workspace/merfish_test/heart_10slice")
CANON = ROOT / "canonical"
MODDIR = ROOT / "modules"
LOGDIR = ROOT / "logs"
for d in (CANON, MODDIR, LOGDIR):
    d.mkdir(parents=True, exist_ok=True)

SECTIONS = list(range(21, 31))
REF_SECTIONS = list(range(1, 21)) + list(range(31, 54))
BIN_UM = 50.0
MIN_CELLS = 5
CHUNK = 100_000
CLASS_NAMES = ["BEC", "Endocardial", "Epicardial", "Fibro", "LEC", "Neuronal",
               "Pericyte", "VSMC", "WBC", "aCM", "ncCM", "vCM"]

T0 = time.time()
LOG = (LOGDIR / "01_build_canonical.log").open("a")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path, obj):
    tmp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=jdefault))
    tmp.replace(path)


def sha256(path, blocks=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(blocks)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- metadata
log(f"start; source={SRC} ({os.path.getsize(SRC)/1e9:.2f} GB)")
with h5py.File(SRC, "r") as f:
    genes = np.array([g.decode() if isinstance(g, bytes) else str(g)
                      for g in f["var/_index"][:]])
    sec = f["obs/section"][:].astype(np.int32)
    cls_codes = f["obs/class/codes"][:].astype(np.int16)
    cls_cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in f["obs/class/categories"][:]]
    xyz = f["obsm/X_spateo"][:].astype(np.float64)
    n_cells_total, n_genes = f["X"].shape

assert n_genes == 238 and genes.size == 238, (n_genes, genes.size)
assert cls_cats == CLASS_NAMES, cls_cats
log(f"cells={n_cells_total} genes={n_genes} classes={cls_cats}")
log(f"X_spateo extents x[{xyz[:,0].min():.1f},{xyz[:,0].max():.1f}] "
    f"y[{xyz[:,1].min():.1f},{xyz[:,1].max():.1f}] z[{xyz[:,2].min():.1f},{xyz[:,2].max():.1f}]")

is_tgt = np.isin(sec, SECTIONS)
is_ref = np.isin(sec, REF_SECTIONS)
assert not np.any(is_tgt & is_ref)
assert is_tgt.sum() + is_ref.sum() == n_cells_total
log(f"target cells (21-30)={int(is_tgt.sum())} reference cells (1-20,31-53)={int(is_ref.sum())}")

# global 50 um lattice anchored at a multiple of 50 um
ox = np.floor(xyz[:, 0].min() / BIN_UM) * BIN_UM
oy = np.floor(xyz[:, 1].min() / BIN_UM) * BIN_UM
ix_all = np.floor((xyz[:, 0] - ox) / BIN_UM).astype(np.int64)
iy_all = np.floor((xyz[:, 1] - oy) / BIN_UM).astype(np.int64)
nx = int(ix_all.max()) + 1
ny = int(iy_all.max()) + 1
assert nx < 1000 and ny < 1000
log(f"lattice origin=({ox},{oy}) nx={nx} ny={ny}")

# target bin grouping (section, ix, iy) -> global group id, sorted by key
key_all = sec.astype(np.int64) * 1_000_000 + ix_all * 1000 + iy_all
tkey = key_all[is_tgt]
uniq_key, inv = np.unique(tkey, return_inverse=True)
n_bins = uniq_key.size
b_sec = (uniq_key // 1_000_000).astype(np.int32)
b_ix = ((uniq_key % 1_000_000) // 1000).astype(np.int32)
b_iy = (uniq_key % 1000).astype(np.int32)
log(f"target bins before MIN_CELLS filter: {n_bins}")

# row -> position within target rows (for streaming accumulation)
tgt_pos = np.full(n_cells_total, -1, dtype=np.int64)
tgt_pos[is_tgt] = inv

# ---------------------------------------------------------------- stream X
sums = np.zeros((n_bins, n_genes), dtype=np.float64)
sum_x = np.zeros(n_bins, dtype=np.float64)
sum_y = np.zeros(n_bins, dtype=np.float64)
sum_z = np.zeros(n_bins, dtype=np.float64)

n_cls = len(CLASS_NAMES)
ref_n = np.zeros(n_cls, dtype=np.int64)
ref_sum = np.zeros((n_cls, n_genes), dtype=np.float64)
ref_sumsq = np.zeros((n_cls, n_genes), dtype=np.float64)
ref_gsum = np.zeros(n_genes, dtype=np.float64)
ref_gsq = np.zeros(n_genes, dtype=np.float64)

log("streaming X ...")
with h5py.File(SRC, "r") as f:
    X = f["X"]
    for r0 in range(0, n_cells_total, CHUNK):
        r1 = min(r0 + CHUNK, n_cells_total)
        Xc = X[r0:r1]
        Xc = np.asarray(Xc, dtype=np.float64)
        sel = tgt_pos[r0:r1] >= 0
        if sel.any():
            g = tgt_pos[r0:r1][sel]
            for j in range(n_genes):
                sums[:, j] += np.bincount(g, weights=Xc[sel, j], minlength=n_bins)
            sum_x += np.bincount(g, weights=xyz[r0:r1, 0][sel], minlength=n_bins)
            sum_y += np.bincount(g, weights=xyz[r0:r1, 1][sel], minlength=n_bins)
            sum_z += np.bincount(g, weights=xyz[r0:r1, 2][sel], minlength=n_bins)
        rsel = is_ref[r0:r1]
        if rsel.any():
            codes = cls_codes[r0:r1][rsel]
            Xr = Xc[rsel]
            oh = np.zeros((Xr.shape[0], n_cls), dtype=np.float64)
            oh[np.arange(Xr.shape[0]), codes] = 1.0
            ref_sum += (Xr.T @ oh).T
            ref_sumsq += ((Xr * Xr).T @ oh).T
            ref_n += np.bincount(codes, minlength=n_cls)
            ref_gsum += Xr.sum(axis=0)
            ref_gsq += (Xr * Xr).sum(axis=0)
        if (r0 // CHUNK) % 5 == 0:
            log(f"  rows {r1}/{n_cells_total} ({time.time()-T0:.0f}s)")
log(f"stream done ({time.time()-T0:.0f}s)")

# ---------------------------------------------------------------- bins
n_cells_bin = np.bincount(inv, minlength=n_bins).astype(np.int64)
keep = n_cells_bin >= MIN_CELLS
log(f"bins kept (>= {MIN_CELLS} cells): {int(keep.sum())} / {n_bins}")

b_sec, b_ix, b_iy = b_sec[keep], b_ix[keep], b_iy[keep]
cnt = n_cells_bin[keep]
Xmean = (sums[keep] / cnt[:, None]).astype(np.float32)
cx = sum_x[keep] / cnt
cy = sum_y[keep] / cnt
cz = sum_z[keep] / cnt
bx = ox + (b_ix + 0.5) * BIN_UM
by = oy + (b_iy + 0.5) * BIN_UM
bin_id = np.array([f"s{s}_{i}_{j}" for s, i, j in zip(b_sec, b_ix, b_iy)])
log(f"Xmean {Xmean.shape} finite={bool(np.isfinite(Xmean).all())} "
    f"min={float(Xmean.min()):.4f} max={float(Xmean.max()):.4f}")

# per section z / counts
sec_z, sec_counts, sec_bins = {}, {}, {}
for s in SECTIONS:
    m = b_sec == s
    sec_z[str(s)] = float(np.mean(cz[m]))
    sec_bins[str(s)] = int(m.sum())
for s in SECTIONS:
    sec_counts[str(s)] = int(np.sum(sec == s))
    log(f"  section {s}: cells={sec_counts[str(s)]} bins={sec_bins[str(s)]} z={sec_z[str(s)]:.1f}")
exp_z = {str(s): 3150.0 + 150.0 * (s - 21) for s in SECTIONS}
for s in SECTIONS:
    assert abs(sec_z[str(s)] - exp_z[str(s)]) < 1e-6, (s, sec_z[str(s)], exp_z[str(s)])

# ---------------------------------------------------------------- sections.json
sections_json = {
    "source": SRC,
    "grains": "50 um square bins in x/y from obsm['X_spateo'][:,:2]; section depth = X_spateo[:,2]",
    "bin_size_um": BIN_UM,
    "grid": {"origin_x": float(ox), "origin_y": float(oy), "nx": nx, "ny": ny,
             "index_convention": "ix=floor((x-origin_x)/50), iy=floor((y-origin_y)/50); "
                                 "x,y stored as bin centre = origin + (i+0.5)*50"},
    "n_genes": int(n_genes),
    "genes": genes.tolist(),
    "x_normalized": True,
    "renormalized": False,
    "min_cells_per_bin": MIN_CELLS,
    "sections": [{"section": str(s), "section_id": s, "z": sec_z[str(s)],
                  "n_cells": sec_counts[str(s)], "n_bins": sec_bins[str(s)],
                  "z_expected": exp_z[str(s)]} for s in SECTIONS],
    "total_bins": int(keep.sum()),
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
write_json(CANON / "sections.json", sections_json)
log("wrote canonical/sections.json")

# ---------------------------------------------------------------- bins_all.h5ad
import anndata as ad
import scipy.sparse as sp

obs = pd.DataFrame({
    "section": pd.Categorical([str(s) for s in b_sec], categories=[str(s) for s in SECTIONS]),
    "x": bx, "y": by, "z": cz,
    "x_centroid": cx, "y_centroid": cy,
    "ix": b_ix.astype(np.int32), "iy": b_iy.astype(np.int32),
    "n_cells": cnt,
}, index=pd.Index(bin_id.astype(str), name="bin_id"))
var = pd.DataFrame(index=pd.Index(genes.astype(str), name="gene"))
adata = ad.AnnData(X=Xmean, obs=obs, var=var)
adata.obsm["spatial"] = np.column_stack([bx, by, cz]).astype(np.float64)
adata.uns["bin_size_um"] = BIN_UM
adata.uns["grid_origin"] = np.array([ox, oy], dtype=np.float64)
adata.uns["source"] = SRC
adata.uns["x_normalized"] = True
adata.uns["min_cells_per_bin"] = MIN_CELLS
adata.write_h5ad(CANON / "bins_all.h5ad", compression="lzf")
log(f"wrote canonical/bins_all.h5ad ({os.path.getsize(CANON/'bins_all.h5ad')/1e6:.1f} MB)")

# ---------------------------------------------------------------- scenarios.json
scenarios = []
for k in (1, 2, 3, 4):
    for s0 in range(21 + 1, 31 - k + 1):
        block = list(range(s0, s0 + k))
        if s0 - 1 < 21 or s0 + k > 30:
            continue
        # both sides visible: s0-1 below and s0+k above, plus all further sections
        visible = [s for s in SECTIONS if s < block[0] or s > block[-1]]
        assert any(s < block[0] for s in visible) and any(s > block[-1] for s in visible)
        name = f"k{k}_s{block[0]}" + ("" if k == 1 else f"-{block[-1]}")
        scenarios.append({
            "name": name, "k": k,
            "visible": [str(s) for s in visible],
            "target": [str(s) for s in block],
            "left_visible": str(block[0] - 1), "right_visible": str(block[-1] + 1),
            "n_target_sections": k,
        })
n_target_slices = sum(len(sc["target"]) for sc in scenarios)
assert len(scenarios) == 26 and n_target_slices == 60, (len(scenarios), n_target_slices)
write_json(CANON / "scenarios.json", {
    "source_bins": "canonical/bins_all.h5ad",
    "sections_available": [str(s) for s in SECTIONS],
    "design": "contiguous hidden blocks of k=1..4 sections; visible slices on both sides; "
              "hidden slices are never in the visible list; 26 scenarios / 60 target slices",
    "n_scenarios": len(scenarios),
    "n_target_slices": n_target_slices,
    "scenarios": scenarios,
})
log(f"wrote canonical/scenarios.json ({len(scenarios)} scenarios, {n_target_slices} target slices)")

# ---------------------------------------------------------------- target_gt.npz
row_sc, row_sec, row_id, row_x, row_y, row_z = [], [], [], [], [], []
for sc in scenarios:
    for s in sc["target"]:
        m = np.flatnonzero(b_sec == int(s))
        row_sc.append(np.full(m.size, sc["name"], dtype=object))
        row_sec.append(np.full(m.size, s, dtype=object))
        row_id.append(bin_id[m].astype(object))
        row_x.append(bx[m]); row_y.append(by[m]); row_z.append(cz[m])
row_sc = np.concatenate(row_sc).astype(str)
row_sec = np.concatenate(row_sec).astype(str)
row_id = np.concatenate(row_id).astype(str)
row_x = np.concatenate(row_x).astype(np.float64)
row_y = np.concatenate(row_y).astype(np.float64)
row_z = np.concatenate(row_z).astype(np.float64)
gt = np.empty((row_id.size, n_genes), dtype=np.float32)
bin_pos = {b: i for i, b in enumerate(bin_id)}
for i, b in enumerate(row_id):
    gt[i] = Xmean[bin_pos[b]]
np.savez_compressed(CANON / "target_gt.npz", scenario=row_sc, section=row_sec,
                    bin_id=row_id, x=row_x, y=row_y, z=row_z,
                    ix=np.array([int(v.split("_")[1]) for v in row_id], dtype=np.int32),
                    iy=np.array([int(v.split("_")[2]) for v in row_id], dtype=np.int32),
                    gt=gt, genes=genes.astype(str))
log(f"wrote canonical/target_gt.npz rows={row_id.size} gt={gt.shape} "
    f"({os.path.getsize(CANON/'target_gt.npz')/1e6:.1f} MB) GT=sections 21-30 only, eval only")

# ---------------------------------------------------------------- modules
dof_n = ref_n.sum()
g_mean = ref_gsum / dof_n
g_var = (ref_gsq - dof_n * g_mean ** 2) / (dof_n - 1)
g_std = np.sqrt(np.clip(g_var, 0, None))
cls_mean = ref_sum / ref_n[:, None]
cls_var = (ref_sumsq - ref_n[:, None] * cls_mean ** 2) / (ref_n[:, None] - 1)
cls_std = np.sqrt(np.clip(cls_var, 0, None))

# one-vs-rest Cohen's d for each class x gene
n_rest = dof_n - ref_n
rest_mean = (ref_gsum[None, :] - ref_sum) / n_rest[:, None]
rest_gvar = ((ref_gsq[None, :] - ref_sumsq) - n_rest[:, None] * rest_mean ** 2) / (n_rest[:, None] - 1)
pooled = np.sqrt(np.clip((cls_var * (ref_n[:, None] - 1) + rest_gvar * (n_rest[:, None] - 1))
                         / (dof_n - 2), 0, None))
d = (cls_mean - rest_mean) / np.where(pooled > 0, pooled, np.nan)

assign = np.nanargmax(d, axis=0)
best = np.nanmax(d, axis=0)
bad = ~np.isfinite(best)
assign[bad] = -1

modules = {c: [] for c in CLASS_NAMES}
for gi, a in enumerate(assign):
    if a >= 0:
        modules[CLASS_NAMES[a]].append(genes[gi])
small = [c for c in CLASS_NAMES if len(modules[c]) < 5]
for c in small:
    log(f"module {c} had {len(modules[c])} genes (<5) -> moved to Other")
    modules[c] = []
other = [g for g in genes if not any(g in set(m) for m in modules.values())]
covered = sum(len(m) for m in modules.values())

# margin diagnostics: how clearly did the winning class beat the runner-up?
dsorted = np.sort(np.where(np.isfinite(d), d, -np.inf), axis=0)
best_d = dsorted[-1]
second_d = dsorted[-2]
margin = best_d - second_d
rel_margin = margin / np.where(best_d > 0, best_d, np.nan)
weak = np.flatnonzero(rel_margin < 0.2)
log(f"assignment margin: median {np.median(margin):.2f}; "
    f"genes with rel_margin<0.2: {weak.size}/{n_genes}")
log("weak-margin genes: " + ", ".join(f"{genes[i]}({CLASS_NAMES[assign[i]] if assign[i]>=0 else 'None'})"
                                      for i in weak[:40]))

write_json(MODDIR / "heart_modules.json", {
    "source": SRC,
    "derivation": "each gene assigned to the obs['class'] with the largest one-vs-rest Cohen's d "
                  "(reference sections 1-20 and 31-53 only; sections 21-30 never used)",
    "reference_sections": [str(s) for s in REF_SECTIONS],
    "target_sections_excluded": [str(s) for s in SECTIONS],
    "effect_size": "cohen_d_one_vs_rest",
    "min_module_size": 5,
    "n_genes": int(n_genes),
    "class_names": CLASS_NAMES,
    "modules": {c: modules[c] for c in CLASS_NAMES},
    "module_sizes": {c: len(modules[c]) for c in CLASS_NAMES},
    "dropped_small_modules": small,
    "other": other,
    "n_other": len(other),
    "n_assigned": int(covered),
    "coverage_frac": float(covered / n_genes),
    "module_score": "mean over module genes of (x - gene_mean)/gene_std; stats from reference only",
    "assignment_diagnostics": {
        "rule": "argmax over classes of one-vs-rest Cohen's d (reference cells only)",
        "best_effect_size": {g: float(best_d[i]) for i, g in enumerate(genes)},
        "second_best_effect_size": {g: float(second_d[i]) for i, g in enumerate(genes)},
        "best_minus_second": {g: float(margin[i]) for i, g in enumerate(genes)},
        "n_genes_rel_margin_lt_0.2": int(weak.size),
        "median_best_minus_second": float(np.median(margin)),
        "note": "tiny reference classes (Neuronal n=677, ncCM n=622) can win a gene with a small "
                "but statistically real effect; the rule is applied literally and the margin is "
                "recorded so downstream users can filter if desired",
    },
})

std0 = np.flatnonzero(g_std <= 0)
np.savez_compressed(MODDIR / "reference_stats.npz",
                    genes=genes.astype(str),
                    gene_mean=g_mean.astype(np.float64),
                    gene_std=g_std.astype(np.float64),
                    gene_std_safe=np.where(g_std > 0, g_std, 1.0).astype(np.float64),
                    constant_genes=genes[std0].astype(str),
                    class_names=np.array(CLASS_NAMES, dtype=str),
                    class_mean=cls_mean.astype(np.float64),
                    class_std=cls_std.astype(np.float64),
                    class_n=ref_n.astype(np.int64),
                    effect_size=d.astype(np.float64),
                    gene_assign=np.array([CLASS_NAMES[a] if a >= 0 else "None" for a in assign], dtype=str),
                    reference_sections=np.array([str(s) for s in REF_SECTIONS], dtype=str),
                    n_reference_cells=np.array(dof_n, dtype=np.int64),
                    target_cells_used=np.array(0, dtype=np.int64))
log(f"wrote modules/heart_modules.json + reference_stats.npz (ref cells={dof_n})")
log("module sizes: " + ", ".join(f"{c}={len(modules[c])}" for c in CLASS_NAMES)
    + f", Other={len(other)}, coverage={covered/n_genes:.3f}")

summary = {
    "n_bins": int(keep.sum()), "n_bins_before_filter": int(n_bins),
    "bins_per_section": sec_bins, "cells_per_section": sec_counts, "z_per_section": sec_z,
    "n_scenarios": len(scenarios), "n_target_slices": n_target_slices,
    "target_gt_rows": int(row_id.size),
    "module_sizes": {c: len(modules[c]) for c in CLASS_NAMES},
    "n_other": len(other), "coverage_frac": covered / n_genes,
    "reference_cells": int(dof_n),
    "elapsed_sec": round(time.time() - T0, 1),
    "sha256": {p.name: sha256(p) for p in [CANON/"sections.json", CANON/"bins_all.h5ad",
                                           CANON/"scenarios.json", CANON/"target_gt.npz",
                                           MODDIR/"heart_modules.json", MODDIR/"reference_stats.npz"]},
}
write_json(CANON / "build_summary.json", summary)
log(f"done in {time.time()-T0:.0f}s")

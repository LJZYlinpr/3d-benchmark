#!/usr/bin/env python3
"""Build canonical 5-slice target-only MERFISH dataset for all 10 methods.

Source: $MERFISH5_ROOT/observed_50um.h5ad (122,669 observed bins, 1,122 genes,
5 slices). Set MERFISH5_ROOT to the experiment root; targets live in
$MERFISH5_ROOT/targets and canonical output goes to $MERFISH5_ROOT/canonical.
Output (canonical/):
  bins_all.h5ad   all 5 sections, 50 um bins, mean expression (float32)
  sections.json   numeric label -> slice id, z, n_bins
  scenarios.json  7 held-out scenarios M1a..M3a
  target_gt.npz   shared superset rows (scenario, section, bin_id, x, y, z, gt)
  READY

Section labels are numeric strings "1".."5" (= S1..S5 = C57BL6J-3.006..010) so
the target-only runners can sort blocks numerically.  `gt` is written for the
evaluation stage only; runners never read it.
"""
import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MERFISH5_ROOT", Path.cwd())).resolve()
DATA = ROOT
MERFISH = ROOT
CANON = ROOT / "canonical"
OBS = Path(os.environ.get("MERFISH5_OBSERVED", ROOT / "observed_50um.h5ad"))

SLICE_ORDER = ["S1", "S2", "S3", "S4", "S5"]
SLICE_IDS = {
    "S1": "C57BL6J-3.006",
    "S2": "C57BL6J-3.007",
    "S3": "C57BL6J-3.008",
    "S4": "C57BL6J-3.009",
    "S5": "C57BL6J-3.010",
}
LABEL = {s: str(i + 1) for i, s in enumerate(SLICE_ORDER)}
NAME = {v: k for k, v in LABEL.items()}

COMBOS = {
    "M1a": {"missing": ["S2"], "visible": ["S1", "S3", "S4", "S5"]},
    "M1b": {"missing": ["S3"], "visible": ["S1", "S2", "S4", "S5"]},
    "M1c": {"missing": ["S4"], "visible": ["S1", "S2", "S3", "S5"]},
    "M2a": {"missing": ["S2", "S3"], "visible": ["S1", "S4", "S5"]},
    "M2b": {"missing": ["S2", "S4"], "visible": ["S1", "S3", "S5"]},
    "M2c": {"missing": ["S3", "S4"], "visible": ["S1", "S2", "S5"]},
    "M3a": {"missing": ["S2", "S3", "S4"], "visible": ["S1", "S5"]},
}


def main():
    CANON.mkdir(parents=True, exist_ok=True)
    print(f"[read] {OBS}", flush=True)
    a = ad.read_h5ad(OBS)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    X = np.ascontiguousarray(X, dtype=np.float32)
    genes = np.asarray(a.var_names).astype(str)
    assert X.shape == (a.n_obs, genes.size), (X.shape, genes.size)
    assert genes.size == 1122, genes.size

    sl = np.asarray(a.obs["slice"]).astype(str)
    assert set(sl) == set(SLICE_IDS.values()), sorted(set(sl))
    id2name = {v: k for k, v in SLICE_IDS.items()}
    section = np.array([LABEL[id2name[s]] for s in sl], dtype=object)

    xyz = np.asarray(a.obsm["X_aligned"], dtype=np.float64)
    if xyz.shape[1] != 3:
        raise ValueError(xyz.shape)
    # X_aligned is in mm; canonical uses um. obs columns are also mm.
    x_um = np.asarray(a.obs["x_mm"], dtype=np.float64) * 1000.0
    y_um = np.asarray(a.obs["y_mm"], dtype=np.float64) * 1000.0
    z_um = np.asarray(a.obs["z_mm"], dtype=np.float64) * 1000.0
    assert np.allclose(xyz[:, 0] * 1000.0, x_um, atol=1e-6)
    assert np.allclose(xyz[:, 1] * 1000.0, y_um, atol=1e-6)
    assert np.allclose(xyz[:, 2] * 1000.0, z_um, atol=1e-6)

    bin_id = np.asarray(a.obs["bin_id"]).astype(str)
    assert np.unique(bin_id).size == bin_id.size
    obs = pd.DataFrame({
        "section": pd.Categorical(section),
        "x": x_um,
        "y": y_um,
        "z": z_um,
        "ix": np.asarray(a.obs["ix"], dtype=np.int64),
        "iy": np.asarray(a.obs["iy"], dtype=np.int64),
        "iz": np.asarray(a.obs["iz"], dtype=np.int64),
        "n_cells": np.asarray(a.obs["n_cells"], dtype=np.int64),
        "slice_id": pd.Categorical(sl),
    }, index=pd.Index(bin_id, name="bin_id"))
    var = pd.DataFrame({"gene_name": genes}, index=pd.Index(genes, name="gene"))
    out = ad.AnnData(X=X, obs=obs, var=var)
    out.obsm["spatial"] = np.stack([x_um, y_um, z_um], axis=1)
    out.uns["voxel_um"] = 50
    out.uns["source"] = str(OBS)
    out.uns["section_labels"] = {LABEL[k]: SLICE_IDS[k] for k in SLICE_ORDER}
    bins_path = CANON / "bins_all.h5ad"
    out.write_h5ad(bins_path, compression="lzf")
    print(f"[write] {bins_path} {out.shape}", flush=True)

    sections = {}
    for k in SLICE_ORDER:
        lab = LABEL[k]
        m = section == lab
        sections[lab] = dict(
            slice_id=SLICE_IDS[k], label=lab, z_um=float(z_um[m].mean()),
            n_bins=int(m.sum()), x_um=[float(x_um[m].min()), float(x_um[m].max())],
            y_um=[float(y_um[m].min()), float(y_um[m].max())],
        )
    (CANON / "sections.json").write_text(json.dumps(sections, indent=2))

    # target_gt superset: concatenate existing per-combo targets in combo order.
    bid2row = {b: i for i, b in enumerate(bin_id)}
    rows_all = []
    scen_all = []
    gt_all = []
    per_scen = {}
    for combo, spec in COMBOS.items():
        t = np.load(DATA / "targets" / combo / "target.npz", allow_pickle=True)
        tbid = np.asarray(t["obs_names"]).astype(str)
        tsl = np.asarray(t["slice"]).astype(str)
        tX = np.asarray(t["X"], dtype=np.float32)
        tc = np.asarray(t["coords"], dtype=np.float64) * 1000.0  # mm -> um
        assert tX.shape == (tbid.size, genes.size), (combo, tX.shape)
        assert tbid.size > 0
        sec_lab = np.array([LABEL[id2name[s]] for s in tsl], dtype=object)
        # scenario rows must be section-ascending for the runner contract
        sec_int = np.array([int(s) for s in sec_lab])
        order = np.argsort(sec_int, kind="stable")
        assert np.array_equal(order, np.arange(order.size)), f"{combo}: target not section-sorted"
        idx = np.array([bid2row[b] for b in tbid], dtype=np.int64)
        # verify canonical metadata matches the GT file exactly
        assert np.allclose(x_um[idx], tc[:, 0], atol=1e-6)
        assert np.allclose(y_um[idx], tc[:, 1], atol=1e-6)
        assert np.allclose(z_um[idx], tc[:, 2], atol=1e-6)
        # verify GT is the observed expression of those exact bins (eval-only copy)
        maxdiff = float(np.abs(X[idx] - tX).max())
        assert maxdiff == 0.0, f"{combo}: GT differs from observed rows by {maxdiff}"
        rows_all.append(idx)
        scen_all.append(np.array([combo] * tbid.size, dtype=object))
        gt_all.append(tX)
        per_scen[combo] = dict(k=len(spec["missing"]),
                               visible=[LABEL[s] for s in spec["visible"]],
                               target=[LABEL[s] for s in spec["missing"]],
                               n_rows=int(tbid.size),
                               n_per_section={str(k): int(v) for k, v in
                                              zip(*np.unique(sec_lab, return_counts=True))})
        print(f"[target] {combo}: rows={tbid.size} sections={sorted(set(sec_lab))}", flush=True)

    row_index = np.concatenate(rows_all).astype(np.int64)
    scenario = np.concatenate(scen_all).astype(str)
    gt = np.concatenate(gt_all).astype(np.float32)
    sec_rows = section[row_index].astype(str)
    tg_path = CANON / "target_gt.npz"
    np.savez_compressed(
        tg_path,
        gt=gt,
        row_index=row_index,
        scenario=scenario,
        section=sec_rows,
        bin_id=bin_id[row_index],
        x=x_um[row_index],
        y=y_um[row_index],
        z=z_um[row_index],
        genes=genes,
    )
    print(f"[write] {tg_path} rows={row_index.size} gt={gt.shape}", flush=True)

    scen_json = {
        "dataset": "MERFISH ABCA-3 sections .006-.010 (5 slices), 50 um bins",
        "n_genes": int(genes.size),
        "n_bins_all": int(bin_id.size),
        "n_scenarios": len(COMBOS),
        "voxel_um": 50,
        "section_order": [LABEL[s] for s in SLICE_ORDER],
        "slice_ids": {LABEL[s]: SLICE_IDS[s] for s in SLICE_ORDER},
        "scenarios": [
            dict(name=c, k=len(COMBOS[c]["missing"]),
                 visible=[LABEL[s] for s in COMBOS[c]["visible"]],
                 target=[LABEL[s] for s in COMBOS[c]["missing"]],
                 visible_slice_ids=[SLICE_IDS[s] for s in COMBOS[c]["visible"]],
                 target_slice_ids=[SLICE_IDS[s] for s in COMBOS[c]["missing"]],
                 **{k: v for k, v in per_scen[c].items() if k not in ("k", "visible", "target")})
            for c in COMBOS
        ],
    }
    (CANON / "scenarios.json").write_text(json.dumps(scen_json, indent=2))

    summary = {
        "status": "OK", "n_obs": int(out.n_obs), "n_genes": int(genes.size),
        "sections": {k: v["n_bins"] for k, v in sections.items()},
        "target_rows": int(row_index.size), "per_scenario": per_scen,
        "files": {p.name: p.stat().st_size for p in CANON.iterdir() if p.is_file()},
    }
    (CANON / "build_summary.json").write_text(json.dumps(summary, indent=2))
    (CANON / "READY").write_text(json.dumps(summary, indent=2))
    print("[READY]", json.dumps(summary["sections"]), "target_rows", row_index.size)


if __name__ == "__main__":
    main()

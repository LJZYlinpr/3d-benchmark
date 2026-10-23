#!/usr/bin/env python
"""Core metrics for the held-out missing-slice predictions.

Reads predictions/<combo>/<method>.npz and targets/<combo>/target.npz, never
touches the model or the training data.  Writes:
  metrics.csv
  <method>/<combo>/metrics.json
  <combo>/metrics.json   (both methods side by side)
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FIELDS = ["combo", "method", "missing", "visible", "n_target", "n_visible",
          "n_genes", "pcc_pooled", "pcc_gene_mean", "pcc_gene_median",
          "pcc_gene_p10", "pcc_spot_mean", "pcc_spot_median", "rmse", "mae",
          "pred_mean", "gt_mean", "finite_frac", "n_constant_rows",
          "n_genes_scored", "n_spots_scored", "train_sec", "infer_sec",
          "gpu_alloc_peak_mb", "status"]


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", nargs="+", default=sorted(C.COMBOS))
    ap.add_argument("--methods", nargs="+", default=C.METHODS)
    args = ap.parse_args()

    combos_json = load_json(os.path.join(C.ROOT, "combos.json"))
    rows, per_combo = [], {}
    for combo in args.combos:
        tpath = os.path.join(C.ROOT, "targets", combo, "target.npz")
        if not os.path.exists(tpath):
            print(f"[skip] {combo}: no target.npz")
            continue
        tgt = np.load(tpath, allow_pickle=True)
        y = np.asarray(tgt["X"], dtype=np.float64)
        tnames = tgt["obs_names"].astype(str)
        per_combo[combo] = {}
        for method in args.methods:
            ppath = os.path.join(C.ROOT, "predictions", combo, f"{method}.npz")
            if not os.path.exists(ppath):
                print(f"[missing] {combo}/{method}: no prediction")
                continue
            pred = np.load(ppath, allow_pickle=True)
            x = np.asarray(pred["X"], dtype=np.float64)
            pnames = pred["obs_names"].astype(str)
            assert x.shape == y.shape, f"{combo}/{method} shape {x.shape} != {y.shape}"
            if not np.array_equal(pnames, tnames):
                order = {n: i for i, n in enumerate(tnames)}
                idx = np.array([order[n] for n in pnames], dtype=np.int64)
                y = y[idx]
                tnames = tnames[idx]
            assert np.isfinite(x).all(), f"{combo}/{method} non-finite prediction"
            m = C.core_metrics(x, y)
            st = load_json(os.path.join(C.ROOT, method, combo, "status.json"))
            ex = load_json(os.path.join(C.ROOT, "predictions", combo, f"{method}.json"))
            spec = C.COMBOS[combo]
            m.update({
                "combo": combo, "method": method,
                "missing": "+".join(spec["missing"]),
                "visible": "+".join(spec["visible"]),
                "n_target": int(x.shape[0]), "n_genes": int(x.shape[1]),
                "train_sec": ex.get("train_sec"),
                "infer_sec": ex.get("infer_sec"),
                "gpu_alloc_peak_mb": ex.get("gpu_alloc_peak_mb"),
                "status": st.get("status", "UNKNOWN"),
                "in_sample_pcc_pooled": (ex.get("in_sample") or {}).get("pcc_pooled"),
            })
            if combo in (combos_json.get("combos") or {}):
                m["n_visible"] = combos_json["combos"][combo].get("n_visible")
            os.makedirs(os.path.join(C.ROOT, method, combo), exist_ok=True)
            with open(os.path.join(C.ROOT, method, combo, "metrics.json"), "w") as f:
                json.dump(m, f, indent=2, ensure_ascii=False)
            per_combo[combo][method] = m
            rows.append(m)
            print(f"[{combo}/{method}] pooled={m['pcc_pooled']:.4f} "
                  f"gene_mean={m['pcc_gene_mean']:.4f} spot_mean={m['pcc_spot_mean']:.4f} "
                  f"rmse={m['rmse']:.4f} mae={m['mae']:.4f}")

        os.makedirs(os.path.join(C.ROOT, combo), exist_ok=True)
        with open(os.path.join(C.ROOT, combo, "metrics.json"), "w") as f:
            json.dump(per_combo[combo], f, indent=2, ensure_ascii=False)

    with open(os.path.join(C.ROOT, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> metrics.csv")


if __name__ == "__main__":
    main()

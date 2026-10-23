#!/usr/bin/env python3
"""Independent acceptance check for the So3D UniST predictions.

Reads canonical/<dataset>/{target.npz,meta.json} metadata (never target['X']) and
each predictions/<dataset>/UniST.npz; writes verify.json + a summary.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "canonical"
PRED = ROOT / "predictions"
METHOD = "UniST"

checks, per_ds, fails, warns = 0, [], [], []


def check(cond, msg):
    global checks
    checks += 1
    if not cond:
        fails.append(msg)


for d in sorted(p for p in CANON.iterdir() if p.is_dir()):
    ds = d.name
    mp, tp = d / "meta.json", d / "target.npz"
    pp, sp = PRED / ds / f"{METHOD}.npz", PRED / ds / f"{METHOD}.status.json"
    if not pp.exists():
        per_ds.append(dict(dataset=ds, status="MISSING"))
        fails.append(f"{ds}: prediction file missing")
        continue
    meta = json.loads(mp.read_text())
    t = np.load(tp, allow_pickle=False)
    p = np.load(pp, allow_pickle=False)
    n, g = meta["n_target"], meta["n_genes"]
    X = p["pred"]
    check(X.shape == (n, g), f"{ds}: pred shape {X.shape} != {(n, g)}")
    check(X.dtype == np.float32, f"{ds}: pred dtype {X.dtype}")
    check(np.isfinite(X).all(), f"{ds}: non-finite predictions")
    check(float(X.min()) >= 0.0, f"{ds}: negative predictions min={X.min()}")
    check(np.array_equal(p["obs_names"], t["obs_names"]), f"{ds}: obs_names not aligned")
    check(np.array_equal(p["coords"], t["coords"].astype(np.float32)), f"{ds}: coords not aligned")
    check(np.array_equal(p["slice"], t["slice"]), f"{ds}: slice not aligned")
    check(np.all(p["slice"] == meta["target_slice"]), f"{ds}: slice != target_slice")
    check(list(p["genes"]) == meta["genes"], f"{ds}: gene order != meta.genes")
    check(str(p["method"]) == METHOD, f"{ds}: method tag wrong")
    # not degenerate: rows must differ, and some genes must vary across spots
    std = X.std(axis=0)
    nonconst = int((std > 0).sum())
    n_unique_rows = int(len(np.unique(np.round(X, 6), axis=0)))
    check(n_unique_rows > max(1, int(0.5 * n)),
          f"{ds}: only {n_unique_rows}/{n} distinct prediction rows")
    check(nonconst >= max(1, int(0.02 * g)),
          f"{ds}: too many constant genes ({nonconst}/{g})")
    if nonconst < 0.10 * g:
        warns.append(f"{ds}: only {nonconst}/{g} genes vary across target spots "
                     f"(INR/decoder smoothing; rows still distinct)")
    st = json.loads(sp.read_text()) if sp.exists() else {}
    check(st.get("status") == "DONE", f"{ds}: status file {st.get('status')}")
    per_ds.append(dict(dataset=ds, status="PASS", n_target=n, n_genes=g,
                       pred_mean=float(X.mean()), pred_max=float(X.max()),
                       pred_zero_frac=float((X == 0).mean()),
                       genes_nonconstant=nonconst,
                       genes_nonconstant_frac=round(nonconst / g, 4),
                       distinct_rows_frac=round(n_unique_rows / n, 4)))

summary = dict(method=METHOD, datasets=len(per_ds), passed=sum(r["status"] == "PASS" for r in per_ds),
               n_checks=checks, n_failures=len(fails), failures=fails,
               n_warnings=len(warns), warnings=warns, per_dataset=per_ds)
(Path(__file__).resolve().parent / "verify.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "per_dataset"}, indent=2))

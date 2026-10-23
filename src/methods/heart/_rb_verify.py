#!/usr/bin/env python
"""Independent verification of runner B prediction files."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _rb_common as C  # noqa: E402

METHODS = ["PASTE-OT-Linear", "ReHo", "Ours-EncMTP"]


def main():
    tg = C.load_target_gt(keys=("bin_id", "section", "x", "y", "scenario"))
    scs = C.load_scenarios()
    bins = C.load_bins()
    bin_sec = {b: s for b, s in zip(bins["bin_id"], bins["section"])}
    out = {"methods": {}, "n_scenarios": len(scs), "checks": []}
    fail = []
    for meth in METHODS:
        per = {}
        for sc in scs:
            p = C.PRED_ROOT / sc["name"] / f"{meth}.npz"
            if not p.exists():
                fail.append(f"{meth}/{sc['name']}: missing")
                continue
            z = np.load(p, allow_pickle=False)
            rows = C.scenario_rows(tg, sc)
            pred = z["pred"]
            sec = np.asarray(z["section"]).astype(str)
            ids = np.asarray(z["bin_id"]).astype(str)
            chk = {
                "shape_ok": pred.shape == (rows.size, C.N_GENES),
                "dtype": str(pred.dtype),
                "row_index_ok": bool(np.array_equal(z["row_index"], rows)),
                "bin_id_ok": bool(np.array_equal(ids, tg["bin_id"][rows])),
                "sections_ok": set(sec.tolist()) == set(sc["target"]),
                "bins_in_canonical": all(bin_sec.get(b) in sc["target"] for b in ids),
                "finite": bool(np.isfinite(pred).all()),
                "nonzero_frac": round(float((pred != 0).mean()), 4),
                "per_gene_std_mean": round(float(pred.std(0).mean()), 5),
                "mean": round(float(pred.mean()), 5),
                "min": round(float(pred.min()), 5),
                "max": round(float(pred.max()), 5),
            }
            chk["pass"] = all([chk["shape_ok"], chk["row_index_ok"], chk["bin_id_ok"],
                               chk["sections_ok"], chk["bins_in_canonical"], chk["finite"]])
            if not chk["pass"]:
                fail.append(f"{meth}/{sc['name']}: {chk}")
            per[sc["name"]] = chk
        out["methods"][meth] = dict(
            n_files=len(per),
            all_pass=all(c["pass"] for c in per.values()),
            n_rows=int(sum(C.scenario_rows(tg, s).size for s in scs
                           if s["name"] in per)),
            scenarios=per)
    # cross-method: predictions must not be identical copies of each other
    for sc in scs:
        ps = {}
        for meth in METHODS:
            p = C.PRED_ROOT / sc["name"] / f"{meth}.npz"
            if p.exists():
                ps[meth] = np.load(p, allow_pickle=False)["pred"]
        names = list(ps)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                same = np.array_equal(ps[names[i]], ps[names[j]])
                out["checks"].append(dict(scenario=sc["name"], pair=[names[i], names[j]],
                                          identical=bool(same)))
                if same:
                    fail.append(f"{sc['name']}: {names[i]} == {names[j]}")
    out["failures"] = fail
    out["pass"] = len(fail) == 0
    C.write_json(C.METHODS_ROOT / "RUNNER_B_VERIFICATION.json", out)
    print(json.dumps({m: {k: v[k] for k in ("n_files", "n_rows", "all_pass")}
                      for m, v in out["methods"].items()}, indent=1))
    print("failures:", len(fail))
    for f in fail[:20]:
        print("  ", f)


if __name__ == "__main__":
    main()

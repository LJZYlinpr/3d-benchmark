#!/usr/bin/env python
"""Write status.json and the run.log summary for metrics_v1."""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlib as M


def read_json(name, default=None):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


def main():
    audit = read_json("audit_summary.json", {})
    core = read_json("core_summary.json", {})
    pert = read_json("perturbation_summary.json", {})
    files = ["metric_config.yaml", "input_audit.csv", "metrics_core.csv",
             "metrics_module.csv", "metrics_spatial_pr.csv",
             "metrics_diagnostics.csv", "perturbation_curves.csv",
             "perturbation_samples.csv", "gt_split_half_reliability.csv",
             "audit_summary.json", "core_summary.json",
             "perturbation_summary.json"]
    file_info = {}
    for f in files:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            file_info[f] = {"bytes": os.path.getsize(p),
                            "md5": M.md5_file(p)}
    figs = {}
    fdir = os.path.join(HERE, "figures")
    if os.path.isdir(fdir):
        for f in sorted(os.listdir(fdir)):
            if f.endswith(".png"):
                figs[f] = os.path.getsize(os.path.join(fdir, f))

    status = "DONE" if (core and pert and audit and
                        core.get("n_core_rows", 0) > 0) else "PARTIAL"
    caveats = [
        "The 7 scenarios reuse the same 5 slices (S1-S5); they are NOT independent biological replicates.",
        "M3a keeps only S1/S5 visible (3 consecutive hidden slices) and is a stress case, not a typical operating point.",
        "This 5-slice experiment validates the metric protocol and compares only NTF vs Spateo; it is not a 10-method ranking.",
        "Only 5 slices: slice-level bootstrap CIs are wide and descriptive; no population-level method ranking is claimed.",
        "Metrics that cannot be computed (no GT positives, no prediction positives, constant maps, zero-variance genes) are N/A with an explicit reason, never fabricated.",
        "No dense 3D or full-volume output is produced; all spatial metrics are 2D per target section.",
        "No retraining and no target expression in module definitions, z-score statistics, thresholds or calibration.",
        "MS-SSIM is a masked 3D-free variant: 7x7 window, 3 scales, weighted geometric mean of per-scale masked SSIM (weights 0.0448/0.2856/0.3001 normalised).",
        "Hotspot/boundary thresholds are the S1/S5 reference p90 values and are applied unchanged to GT and predictions.",
        "Spateo predictions are strongly zero-inflated; this affects hotspot precision and variance-ratio diagnostics, not the alignment audit.",
    ]
    status_obj = {
        "protocol": "MERFISH 5-slice metric protocol v1",
        "status": status,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lane": {"jobid": 9383994, "node": "gvna02", "partition": "a100-40g",
                 "cpus_per_task": 4, "tasks": 1},
        "python": sys.executable,
        "caveats": caveats,
        "config_sha256": audit.get("config_sha256"),
        "config_written_utc": audit.get("config_written_utc"),
        "input_audit": {k: audit.get(k) for k in
                        ("n_checks", "n_pass", "n_fail", "n_info")},
        "audit_failures": audit.get("failures", []),
        "core": {k: core.get(k) for k in
                 ("n_core_rows", "n_pr_rows", "n_diag_rows", "n_reliability_rows",
                  "coverage_ok_mean", "bootstrap", "bootstrap_diff",
                  "seconds", "gpu")},
        "perturbation": {k: pert.get(k) for k in
                         ("n_samples", "n_maps", "conditions", "levels",
                          "identity_sanity", "component_changes", "noise",
                          "seconds", "gpu")},
        "files": file_info,
        "figures": figs,
    }
    with open(os.path.join(HERE, "status.json"), "w") as f:
        json.dump(status_obj, f, indent=2, default=M.jsonable)
    with open(os.path.join(HERE, "run.log"), "a") as f:
        f.write("\n===== metrics_v1 finalize %s =====\n" %
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        f.write(json.dumps({k: status_obj[k] for k in
                            ("status", "input_audit", "core", "perturbation")},
                           indent=2, default=M.jsonable))
        f.write("\n")
    print(json.dumps(status_obj, indent=2, default=M.jsonable))


if __name__ == "__main__":
    main()

# Project status

## Completed

| Track | Status |
|---|---|
| MERFISH 5-slice 10 methods × 7 scenarios | predictions 70/70; new metrics 3,420 rows |
| Heart 10-slice 10 methods × 26 scenarios | predictions 260/260; metric aggregation pending |
| So3D 46 datasets × 8 methods | predictions 368/368; metrics 164,800 rows |
| New metric package | MERFISH and So3D implementations present |
| Baseline audit | Ours, PASTE, ReHo audited |
| Grid builder | 10/20/40/100 µm implemented and smoke-tested |

## Not completed

- 54-slice paper main matrix (legacy "50-PC" formulation is not used by this package);
- formal 4-group × 26-scenario MERFISH matrix;
- heart metric summary;
- baseline P0 fixes and reruns;
- one unified environment and one unified runner;
- exact per-scenario seeds/checkpoints/hashes.

## Current headline signals

### MERFISH 5-slice

- Ours-EncMTP leads the module-level table; ReHo is a strong simple lower bound.
- NTF shows the classic in-sample/held-out gap.
- SINTER3D/isoST are weak in the current reduced configurations.

### So3D

- SpatialZ/NTF have the strongest module PCC (~0.256).
- DeepSpatial/isoST/ReHo have lower |ΔMoran| and higher boundary recall, but
  weaker module PCC.
- Gene PCC is near zero for every method and is not a main ranking signal.
- SINTER3D is effectively unsupported on So3D module structure.
- UniST has severe gene coverage/variance collapse.

## Server 1 reproduction (2026-10-23)

- Rebuilt the MERFISH 5-slice input from the committed shards on `aicb1`;
  the end-to-end package pipeline (shards -> observed -> targets -> canonical)
  produces `bins_all` X and `target_gt` hashes identical to the HPC originals.
- Ran six methods on held-out M1a: ReHo, Ours-EncMTP, PASTE-OT-Linear,
  SpatialZ, SINTER3D, DeepSpatial. Deterministic methods reproduce exactly;
  learned methods differ by <=0.005 module-mean gene PCC (SINTER3D 0.013 on a
  near-zero signal).
- NTF, Spateo, UniST and isoST are blocked by missing upstream code or CUDA
  PyG extensions. Full issue list and commands: `docs/REPRO_SERVER1.md`.

## Release blockers

See `docs/BASELINE_AUDIT.md`.

1. PASTE M2b bracket bug.
2. PASTE not faithful PASTE.
3. Ours provenance and relative-offset unit defect.
4. ReHo nearest-vs-preceding semantics.
5. Missing 54-slice decision/manifest.
6. Missing heart metrics and unified environment.

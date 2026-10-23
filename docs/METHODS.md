# Methods

## Counts

- **8 paper common-task methods**: SINTER3D, UniST, isoST, SpatialZ, NTF,
  DeepSpatial, PASTE-OT-Linear, ReHo.
- **3 self-built baselines**: Ours-EncMTP, PASTE-OT-Linear, ReHo.
- **1 supplemental**: Spateo.
- **10 current runnable implementations**: the 8 common + Ours-EncMTP + Spateo.
- **2 blocked**: Spa3D (domain-only, no expression decoder) and ASIGN
  (H&E + 250-gene dependency).

## Method map

| Method | Code | Family | Current caveat |
|---|---|---|---|
| NTF | `src/methods/merfish/NTF/run_ntf_50um.py` | neural tensor factorization | GPU nondeterminism; underfit at small budgets |
| Spateo | `src/methods/merfish/Spateo/run_spateo_50um.py` | 3D interpolation | supplemental, not in 8-method table; 10 µm calibration risk |
| SINTER3D | `src/methods/merfish/SINTER3D/` | deep 3D | variable mini-batch/step budget; So3D weak |
| isoST | `src/methods/merfish/isoST/` | implicit neural 3D | reduced epochs; over-dispersion |
| SpatialZ | `src/methods/merfish/SpatialZ/` | Gaussian splatting | fast; low target coverage on sparse data |
| DeepSpatial | `src/methods/merfish/DeepSpatial/` | deep spatial | reduced epochs/patch adaptation |
| UniST | `src/methods/merfish/UniST/` | SUICA + INR | reduced GAE/INR budgets; variance collapse |
| PASTE-OT-Linear | `src/methods/merfish/PASTE-OT-Linear/` | OT + z-linear | PPA, 6k anchors, not faithful PASTE; M2b bug |
| ReHo | `src/methods/merfish/ReHo/` | nearest-visible copy | not preceding-slice ReHo |
| Ours-EncMTP | `src/methods/merfish/Ours-EncMTP/` | coordinate encoder + MTP | relative-offset unit bug; provenance |

Heart and So3D have their own adapters under `src/methods/heart/` and
`src/methods/so3d/`.

## Method-specific notes

### Ours-EncMTP

- 2.34M parameters, 3-NN visible context per visible slice, 2 cross-attention
  blocks, K=4 MTP auxiliary heads.
- 3,000 steps, batch 1,024, 5% visible-only validation for early stopping.
- No target-expression leakage found (target-X zero/NaN controls = bitwise 0).
- P1: relative offsets are computed in nm against a 50 µm bin width; fix before
  presenting the intended relative-geometry model.

### PASTE-OT-Linear

- Custom entropic FGW + OT + z-linear blend.
- **Not** official PASTE: PPA solver, 6,000 anchors, no nearest-anchor
  expansion, max_iter 100; Sinkhorn warnings observed.
- **P0**: M2b uses a single 1→5 bracket and ignores visible section 3; M2b
  predictions equal corresponding M3a rows. Must rerun.

### ReHo

- Current code selects the **nearest visible z**; ties go to lower z.
- Upstream ReHo is **preceding/lower-z visible**; 5/12 target-section source
  choices differ. Rename or implement preceding.

## Blocked methods

- Spa3D: only outputs domains, no expression decoder; no 1,122-gene path.
- ASIGN: requires H&E and a fixed 250-gene panel; not compatible with this
  benchmark's expression task.

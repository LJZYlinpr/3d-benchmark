# Third-party methods and attribution

The benchmark adapters wrap or reimplement methods from external projects.
Minimal upstream code is vendored under `vendor/methods/` so a fresh clone runs
without HPC paths or a separate fetch.  Full upstream trees can be obtained with
`scripts/fetch_upstream.sh` at the pinned commits in
`docs/upstream/method_commits.tsv`.

| Method | Upstream family | Commit / anchor | Vendored path | Modifications | License |
|---|---|---|---|---|---|
| NTF | NeuroTensorField / SpatialOmics | `f0b4941a…` | `vendor/methods/NTF/` | pure-PyTorch hash-grid fallback | `NTF/LICENSE` |
| Spateo | Spateo | `cbea5b32…` | `vendor/methods/Spateo/…/interpolation_deeplearn/` | minimal package shim; DeepInterpolation only | `spateo-release/LICENSE` |
| SINTER3D | SINTER3D | no VCS snapshot | `vendor/methods/SINTER3D/` | vendored source | no LICENSE file in snapshot |
| isoST | isoST | `805981c3…` | `vendor/methods/isoST/` | CPU kNN, minimal model/utils/config | `isoST/LICENSE` |
| SpatialZ | SpatialZ | `e1b5ed01…` | `vendor/methods/SpatialZ/` | `SpatialZ.py`/`Synthesize.py` only | `SpatialZ/LICENSE` |
| DeepSpatial | DeepSpatial | `fdb98657…` | `vendor/methods/DeepSpatial/` | `deepspatial/` package only | `DeepSpatial/LICENSE` |
| UniST | UniST / SUICA | `a2fd35a7…` | `vendor/methods/UniST/external/SUICA_pro/` | SUICA code/configs only | `UniST/LICENSE` |
| PASTE-OT-Linear | PASTE-inspired OT + custom linear blend | local implementation | `src/methods/merfish/PASTE-OT-Linear/` | benchmark adapter | repository license |
| ReHo | local repeat-holding baseline | local implementation | `src/methods/merfish/ReHo/` | benchmark adapter | repository license |
| Ours-EncMTP | self-built coordinate encoder + MTP | local implementation | `src/methods/merfish/Ours-EncMTP/` | benchmark adapter | repository license |

## Modifications

- NTF: `docs/upstream/patches/NTF_torch_fallback.patch` (integer casts, missing
  `n_output_dims`, pure-PyTorch hash grid).
- isoST: `docs/upstream/patches/isoST.patch` and `docs/upstream/new_files/isoST`
  (CPU kNN call path and benchmark input handling).
- UniST: `docs/upstream/patches/UniST.patch` and
  `docs/upstream/new_files/UniST` (per-scenario/query-coordinate support).
- The benchmark adapters add path/data-format adaptations and frozen budgets;
  they are not verbatim upstream copies.

## Redistribution

Keep upstream license files when redistributing.  SINTER3D has no license file
in the captured snapshot; treat that subtree as internal-review only until the
upstream license is confirmed.  The ABC Atlas MERFISH data are CC BY 4.0 (see
`data/merfish_5slice/README.md`).

# Vendored upstream method code

This directory contains the **minimal upstream code** needed by the ten
MERFISH 5-slice method adapters.  It is vendored so a fresh clone does not
depend on HPC workspaces, release downloads, or a separate upstream fetch.
Only imported modules are included; examples, tutorials, checkpoints and data
are intentionally excluded.

| Method | Upstream | Commit | Vendored path | Modifications | License |
|---|---|---|---|---|---|
| NTF | `GYQ-form/NTF` | `f0b4941a07330d9749567ef52df8bc9f5cef976a` | `NTF/` | pure-PyTorch hash-grid fallback patch applied (`docs/upstream/patches/NTF_torch_fallback.patch`) | `NTF/LICENSE` |
| isoST | `deng-ai-lab/isoST` | `805981c3fad66e3c9f5409816f95c17332906600` | `isoST/` | minimal `model/`, `utils/`, `config/`; CPU kNN call; benchmark adaptations | `isoST/LICENSE` |
| SpatialZ | `senlin-lin/SpatialZ` | `e1b5ed01e933729149e08399febca08db16ced14` | `SpatialZ/` | `SpatialZ.py`, `Synthesize.py` only | `SpatialZ/LICENSE` |
| DeepSpatial | `yyh030806/DeepSpatial` | `fdb98657fca95679cc45adf3510a161de05bca08` | `DeepSpatial/` | `deepspatial/` package only | `DeepSpatial/LICENSE` |
| Spateo | `aristoteleo/spateo-release` | `cbea5b322316dd1d3cbcd06bd0bc61368fe0bff3` | `Spateo/spateo-release/spateo/tdr/interpolations/interpolation_deeplearn/` | minimal package shims; only DeepInterpolation | `Spateo/spateo-release/LICENSE` |
| UniST | `lanshui98/UniST` | `a2fd35a70a6080caab5fdd60652242f85059b09a` | `UniST/external/SUICA_pro/` | SUICA code + configs only; no logs/checkpoints/data | `UniST/LICENSE` |
| SINTER3D | no VCS snapshot | — | `SINTER3D/SINTER3D-master/` | vendored source from the project snapshot | **no LICENSE file found in the upstream snapshot** |

The adapters for ReHo, PASTE-OT-Linear and Ours-EncMTP are original benchmark
code under `src/methods/merfish/`; they do not need a vendor tree.

`vendor/shims/torch_cluster/` is a small CPU `knn_graph` implementation used by
isoST when the platform-specific CUDA `torch_cluster` wheel is unavailable.
It is not upstream code.

## Refreshing from upstream

`scripts/fetch_upstream.sh` can rebuild full upstream trees at the pinned
commits under `$MERFISH5_UPSTREAM`.  The default package does **not** need it:
`MERFISH5_UPSTREAM` defaults to this `vendor/methods` directory.

## Provenance and attribution

The upstream repositories and their notices remain the property of their
authors.  Keep the license files when redistributing.  SINTER3D has no license
file in the captured snapshot; treat that subtree as internal-review only until
the upstream license is confirmed.  See `docs/THIRD_PARTY.md` for the method
families and modifications.

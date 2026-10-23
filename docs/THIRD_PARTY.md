# Third-party methods and attribution

The benchmark adapters wrap or reimplement methods from external projects.
This draft package is intended for internal review. Before public release:

- add the upstream repository URL and commit/hash for each method;
- preserve upstream licenses and notices;
- distinguish unmodified code, adapted code and reimplementation;
- document all deviations (epochs, anchors, solver, patch size, batch size).

Method families in this package:

| Method | Upstream family |
|---|---|
| NTF | Neural Tensor Factorization / SpatialOmics |
| Spateo | Spateo |
| SINTER3D | SINTER3D |
| isoST | isoST |
| SpatialZ | SpatialZ |
| DeepSpatial | DeepSpatial |
| UniST | UniST / SUICA |
| PASTE-OT-Linear | PASTE-inspired OT + custom linear blend |
| ReHo | local repeat-holding baseline |
| Ours-EncMTP | self-built coordinate encoder + MTP |

The local `src/methods/` code contains benchmark-specific path and data-format
adaptations; it is not necessarily a verbatim copy of the upstream package.

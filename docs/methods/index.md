# Method reproduction index

Each method has a dedicated document with:

- implementation path;
- environment;
- required inputs;
- command template;
- output contract;
- dataset tracks (MERFISH 5-slice / Heart / So3D);
- known deviations and P0/P1 issues.

## Common-task methods

| Method | Doc | Current status |
|---|---|---|
| NTF | [ntf.md](ntf.md) | 10 methods; NTF common task |
| Spateo | [spateo.md](spateo.md) | supplemental reference |
| SINTER3D | [sinter3d.md](sinter3d.md) | common task |
| isoST | [isost.md](isost.md) | common task |
| SpatialZ | [spatialz.md](spatialz.md) | common task |
| DeepSpatial | [deepspatial.md](deepspatial.md) | common task |
| UniST | [unist.md](unist.md) | common task |
| PASTE-OT-Linear | [paste-ot-linear.md](paste-ot-linear.md) | self-built baseline; P0 M2b issue |
| ReHo | [reho.md](reho.md) | self-built baseline; nearest-visible semantics |
| Ours-EncMTP | [ours-encmtp.md](ours-encmtp.md) | self-built baseline; provenance/unit issues |

## Blocked

- [spa3d.md](spa3d.md)
- [asign.md](asign.md)

## Shared contract

```text
predictions/<dataset>/<method>.npz
  pred or X  float32 [n_target, n_genes]
  obs_names
  coords
  slice
  row_index
```

Rows align to `canonical/<dataset>/target.npz`. Target expression is never used
for fitting/calibration/model selection.

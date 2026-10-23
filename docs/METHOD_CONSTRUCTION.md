# How the methods were constructed

This document records the actual construction path from the xpy reference
implementation to the runnable adapters in this package. It is intentionally
more concrete than `docs/METHODS.md`.

## 1. Reference implementation: xpy

The original benchmark build is in:

```text
/storage/yuhongtaoLab/linpeiran/xpy/lab/baseline/
  methods/                 upstream method snapshots / working trees
  benchmark/src/adapters/  project-owned adapters
  benchmark/method_specs/  machine-readable method capability specs
  benchmark/METHOD_CHANGES.md
  state/method_provenance/ commits, patches, new-file lists
```

This package includes:

- `reference/xpy_adapters/` — reference adapter code;
- `configs/method_specs/` — xpy capability specs;
- `docs/upstream/METHOD_CHANGES.md` — full change log;
- `docs/upstream/method_commits.tsv` — upstream commit anchors;
- `docs/upstream/method_provenance.md` — provenance caveats.

The current package's `src/methods/*` scripts are **workspace-specific
runnable adapters** derived from this reference; they are not a verbatim copy
of the upstream repositories.

## 2. Method construction table

| Method | Upstream anchor | Construction | Key adaptations | Current status |
|---|---|---|---|---|
| SpatialZ | `e1b5ed01e933729149e08399febca08db16ced14`, master | wrapper around official SpatialZ | `lr=1e5→1.0`, `nb_iter_max=1000`, frozen NN mapping generated→query, GPU device auto | 8-method common task |
| UniST | `a2fd35a70a6080caab5fdd60652242f85059b09a`, main | SUICA GAE + INR + decoder, adapter | z-axis `[:,1]→[:,2]`; `pred_config.pipeline.custom_coords_file`; raw query coords; output/query assertion; checkpoint glob; decoder=None; GPU selection | 8-method common task |
| isoST | `805981c3fad66e3c9f5409816f95c17332906600`, main | upstream isoST model + adapter | visible-only PCA/normalization; official 3-stage schedule; holdout_slices; z-axis fix; continuous-z fix; 50-PC feature path is historical only | 8-method common task |
| NTF | `f0b4941a07330d9749567ef52df8bc9f5cef976a`, main | official NTF + adapter | visible-only bbox/PSF, hash-grid INR, slice embedding, dropout network; no target-slice embedding | 8-method common task |
| DeepSpatial | `fdb98657fca95679cc45adf3510a161de05bca08`, main | official DeepSpatial + adapter | patch/thickness/epoch adaptation; visible-only fit | 8-method common task |
| SINTER3D | `no_vcs` | project adapter | subprocess call to `inference_latent(decode=True)` at query coords; z-axis fix; holdout_slices; training-step substitution; visible-only KMeans | 8-method common task |
| PASTE-OT-Linear | `no_vcs`, defined in `naive_baselines.py` | custom OT + z-linear baseline | entropic FGW; frozen OT; z-linear blend; anchor subset; current package uses PPA and 6k anchors | self-built baseline |
| ReHo | `no_vcs`, defined in `naive_baselines.py` | custom repeat-holding baseline | frozen NN mapping source→query; current code uses nearest-visible slice | self-built baseline |
| Spateo | `no_vcs` | upstream Spateo + local runner | 3D interpolation / DeepInterpolation; 10 µm calibration is a risk | supplemental |
| Ours-EncMTP | not in xpy | self-built coordinate encoder + MTP | Fourier(x,y,z), slice embedding, visible context, 2 cross-attn blocks, K=4 MTP | self-built baseline |
| Spa3D | `no_vcs` | upstream domain method | no expression decoder | blocked |
| ASIGN | `no_vcs` | upstream H&E method | requires H&E + 250-gene panel | blocked |

## 3. Adapter design

### Common interface

All adapters consume a canonical visible AnnData and target coordinates, then
write target-only predictions:

```text
visible.h5ad / visible matrix
target coords (and target row order)
        ↓ adapter
predictions/<dataset>/<method>.npz
```

Target expression is not an input. The xpy `method_specs` declare
`fixed_coordinate` and `min_visible_slices: 2`.

### SINTER3D

The adapter does not modify the method algorithm. It:

1. writes a temporary input in SINTER3D's expected format;
2. runs the method in a subprocess;
3. calls `inference_latent(decode=True)` at the target coordinates;
4. fixes the historical z-axis column (`[:,1]→[:,2]`);
5. supports `holdout_slices` and training-step substitution.

### isoST

The historical adapter used visible/holdout splitting before PCA and
normalization, and aligned the training recipe to the official
`config/mouse_brain.yml`. It also fixed:
- z-axis column;
- continuous-z handling by slice labels;
- `holdout_slices` for gap > 1;
- one virtual slice per held-out z for inference.

**Important:** the xpy isoST path was built around a 50-PC feature dataset. The
new v1 metric package does **not** use 50-PC features. The historical 50-PC path
must not be imported into the new metric table.

### UniST

The adapter:
- fixes the z-axis column;
- sets `pred_config.pipeline.custom_coords_file` (not the config root);
- keeps query coordinates raw (INR uses `require_coordnorm: False`);
- asserts output `obsm['spatial'] == query coords`;
- selects the latest checkpoint because `train_inr()` returns None;
- sets `decoder=None` for raw-expression prediction.

Without the custom-coords fix, SUICA silently uses its 5,488-point demo
coordinates; that was a real provenance bug.

### SpatialZ

The adapter:
- fixes `lr` from the unstable default `1e5` to `1.0`;
- uses `nb_iter_max=1000`;
- passes `device=auto` through to the generated script;
- maps generated cells to the public query by a frozen nearest-neighbor map;
- uses a pseudo-mm xy frame.

### DeepSpatial / NTF

Both are wrapped rather than algorithmically modified:
- NTF uses visible-only bounding box, PSF and hash-grid sampling; no target-slice
  embedding;
- DeepSpatial uses visible-only setup and a reduced patch/thickness schedule
  for the current data sizes.

### PASTE-OT-Linear

Defined in `naive_baselines.py`:
- entropic fused Gromov-Wasserstein alignment of the two visible bracket
  slices;
- z-linear blend + inverse-distance weighting;
- frozen OT/anchor support.

The xpy log records 3,000 anchors; the current workspace uses 6,000 anchors
(PPA). This is a resource-driven deviation and must be labelled.

### ReHo

Defined in `naive_baselines.py`:
- frozen NN mapping from source-slice expression to query coordinates;
- current workspace implementation selects the nearest visible z.

The upstream xpy definition uses preceding/lower-z; the current code is a
**nearest-visible ReHo variant**.

## 4. Historical changes that matter for interpretation

From `docs/upstream/METHOD_CHANGES.md`:

- z-axis column fixed from `[:,1]` to `[:,2]` for SINTER3D/UniST/isoST;
- isoST PCA leakage fixed by splitting visible/holdout before fitting;
- continuous-z handling fixed by using slice labels;
- `holdout_slices` added for gap 2/3;
- DataLoader `num_workers` deadlock handled;
- isoST third-party math changes were reverted to upstream (softmin, loss
  wrapping);
- UniST custom-coords bug fixed;
- SpatialZ learning-rate instability fixed;
- PASTE OT fallback/row-order pairing bugs fixed;
- ReHo frozen NN mapping fixed.

## 5. Provenance gaps

The xpy provenance directory records:

- SpatialZ, UniST, isoST, NTF, DeepSpatial have upstream commit anchors;
- UniST and isoST have working-tree patches and untracked new files;
- SINTER3D, PASTE-OT-Linear, ReHo, Spa3D, ASIGN, Spateo have `no_vcs`;
- the main 54-slice table's run manifest is missing;
- `run_manifest` schemas are inconsistent across campaigns.

Therefore, for a public release, every method doc must preserve:
- upstream repo URL and commit;
- adapter path and patch;
- exact config and seed;
- environment lock;
- run manifest and prediction hash.

The current package's `docs/methods/*.md` are the per-method starting point;
`docs/upstream/*` is the evidence trail.

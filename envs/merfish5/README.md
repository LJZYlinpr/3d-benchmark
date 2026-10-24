# MERFISH5 environment locks

These files are **captured `pip freeze` snapshots**, not blind install lists.
They document the tested versions on server 1 (`common`, `deepspatial`, `unist`).
The authoritative install path is:

```bash
bash scripts/setup_merfish5_env.sh
# or, on RTX 4090 / Ada:
MERFISH5_PROFILE=ada bash scripts/setup_merfish5_env_ada.sh
```

The setup scripts pin torch, PyG, KeOps and the method extras explicitly and
create `$MERFISH5_ROOT/env.sh`.  Do not `pip install -r` a lock file over a
working env: several methods need incompatible torch/CUDA stacks.

Local wheel paths have been replaced with comments; use the setup script or
download the matching CUDA PyG wheel for your torch build.

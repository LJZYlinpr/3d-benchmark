# Vendored shims

`torch_cluster/` is a small CPU implementation of `knn_graph` with the
torch_cluster API.  It is used by isoST when the platform-specific CUDA
`torch_cluster` wheel is unavailable.  Add `vendor/shims` to `PYTHONPATH`
before `vendor/methods/isoST`; the setup scripts and the M1a wrapper do this.

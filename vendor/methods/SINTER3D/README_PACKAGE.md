# Vendored SINTER3D source

This directory contains the SINTER3D source tree needed by
`src/methods/merfish/SINTER3D/run_merfish5.py`.

The upstream project has no public VCS commit anchor in the benchmark
provenance (`method_commits.tsv: SINTER3D no_vcs`).  To keep the Git package
self-contained, the source files (not the 114 MB tutorials/data) are vendored
here.  The runner falls back to this directory when
`$MERFISH5_UPSTREAM/SINTER3D/SINTER3D-master` is absent.

The upstream tree did not include a LICENSE file.  Before redistributing this
directory outside the project, confirm the license with the SINTER3D authors.
The vendored files are unmodified except for the benchmark adapter outside this
directory.

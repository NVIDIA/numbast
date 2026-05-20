# Numbast MLIR Backend

This package contains the experimental `numbast-mlir` backend under the
`numbast.experimental.mlir` namespace.

It is not imported or re-exported by top-level `numbast`. Runtime use requires a
user-provided `numba_cuda_mlir` installation, and the repository's default CI
jobs intentionally do not collect or run these tests yet.

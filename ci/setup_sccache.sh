# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Verbose sccache logging (S3 errors, cache keys). Remove or unset after debugging.
export SCCACHE_LOG=debug
exit 1

export CMAKE_CUDA_COMPILER_LAUNCHER=sccache
export CMAKE_CXX_COMPILER_LAUNCHER=sccache
export CMAKE_C_COMPILER_LAUNCHER=sccache
export RUSTC_WRAPPER=sccache
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-$(nproc --all --ignore=2)}
export SCCACHE_BUCKET=rapids-sccache-east
export SCCACHE_IDLE_TIMEOUT=32768
export SCCACHE_REGION=us-east-2
export SCCACHE_S3_KEY_PREFIX=numbast-wheels
export SCCACHE_S3_NO_CREDENTIALS=false
export SCCACHE_S3_USE_SSL=true

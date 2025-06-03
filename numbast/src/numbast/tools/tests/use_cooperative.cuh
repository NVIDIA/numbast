// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <cooperative_groups.h>
#include <cuda/barrier>

namespace cg = cooperative_groups;

__device__ __inline__ void
_wait_on_tile(cuda::barrier<cuda::thread_scope_block> &tile) {
  auto token = tile.arrive();
  tile.wait(std::move(token));
}

extern "C" __device__ __inline__ int cta_barrier() {
  auto cta = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);
  __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
  if (threadIdx.x == 0) {
    init(&barrier, blockDim.x);
  }

  _wait_on_tile(barrier);
  return 0;
}

// Functions for regex testing - these should match pattern ".*_barrier.*"
extern "C" __device__ __inline__ int global_barrier_sync() {
  auto cta = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);
  __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
  if (threadIdx.x == 0) {
    init(&barrier, blockDim.x);
  }
  _wait_on_tile(barrier);
  return 0;
}

extern "C" __device__ __inline__ int thread_barrier_wait() {
  auto cta = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);
  __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
  if (threadIdx.x == 0) {
    init(&barrier, blockDim.x);
  }
  _wait_on_tile(barrier);
  return 0;
}

// Functions for regex testing - these should match pattern ".*_sync.*"
extern "C" __device__ __inline__ int grid_sync_all() {
  auto cta = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);
  __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
  if (threadIdx.x == 0) {
    init(&barrier, blockDim.x);
  }
  _wait_on_tile(barrier);
  return 0;
}

extern "C" __device__ __inline__ int block_sync_threads() {
  auto cta = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);
  __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
  if (threadIdx.x == 0) {
    init(&barrier, blockDim.x);
  }
  _wait_on_tile(barrier);
  return 0;
}

// Functions that should NOT match any patterns - these should not be
// cooperative
extern "C" __device__ __inline__ int regular_function() {
  return threadIdx.x + blockIdx.x;
}

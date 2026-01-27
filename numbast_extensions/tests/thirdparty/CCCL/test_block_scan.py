# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

from numbast import MemoryShimWriter

from numba import cuda
from numba.cuda.types import int32

from .conftest import make_bindings, requires_cuda_13


@requires_cuda_13
class TestBlockScan:
    """Tests for CUB BlockScan operations."""

    @pytest.fixture(scope="class")
    def block_scan_bindings(self, block_scan_header):
        """Create BlockScan bindings."""
        shim_writer = MemoryShimWriter(f"#include <{block_scan_header}>")

        BlockScan = make_bindings(
            block_scan_header,
            shim_writer,
            "cub::BlockScan",
            {"BlockScan": {"InclusiveSum": {"output": "out_return"}}},
        )

        if BlockScan is None:
            pytest.skip("Failed to create BlockScan bindings")

        return BlockScan, shim_writer

    def test_inclusive_sum(self, block_scan_bindings):
        """Test BlockScan InclusiveSum operation."""
        BlockScan, shim_writer = block_scan_bindings

        num_threads_per_block = 32

        @cuda.jit(link=shim_writer.links())
        def try_block_scan(d_input, d_output):
            tid = cuda.threadIdx.x

            # Instantiate BlockScan for a 1D block
            block_scan_t = BlockScan(
                T=int32,
                BLOCK_DIM_X=num_threads_per_block,
                ALGORITHM="cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING",
                BLOCK_DIM_Y=1,
                BLOCK_DIM_Z=1,
            )

            block_scan = block_scan_t()

            x = d_input[tid]

            # Inclusive scan (sum) across the block
            out = block_scan.InclusiveSum(x)

            # Write the output back to global memory
            d_output[tid] = out

        d_input = cuda.to_device(
            np.arange(1, 1 + num_threads_per_block, dtype=np.int32)
        )
        d_output = cuda.device_array(num_threads_per_block, dtype=np.int32)

        try_block_scan[1, num_threads_per_block](d_input, d_output)
        cuda.synchronize()

        result = d_output.copy_to_host()
        expected = np.cumsum(
            np.arange(1, 1 + num_threads_per_block, dtype=np.int32)
        )

        np.testing.assert_array_equal(result, expected)

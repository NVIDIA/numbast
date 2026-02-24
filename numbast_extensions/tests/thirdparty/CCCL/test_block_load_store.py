# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import cffi

from numbast import MemoryShimWriter

from numba import cuda
from numba.cuda.types import int32

from .conftest import make_bindings, requires_cuda_13


@requires_cuda_13
class TestBlockLoadStore:
    """Tests for CUB BlockLoad and BlockStore operations."""

    @pytest.fixture(scope="class")
    def block_load_store_bindings(self, block_load_header, block_store_header):
        """Create BlockLoad and BlockStore bindings."""
        shim_writer = MemoryShimWriter(
            f"#include <{block_load_header}>\n#include <{block_store_header}>"
        )

        BlockLoad = make_bindings(
            block_load_header,
            shim_writer,
            "cub::BlockLoad",
            {"BlockLoad": {"Load": {"dst_items": "out_ptr"}}},
        )

        BlockStore = make_bindings(
            block_store_header,
            shim_writer,
            "cub::BlockStore",
            {"BlockStore": {"Store": {"items": "out_ptr"}}},
        )

        if BlockLoad is None:
            pytest.skip("Failed to create BlockLoad bindings")
        if BlockStore is None:
            pytest.skip("Failed to create BlockStore bindings")

        return BlockLoad, BlockStore, shim_writer

    def test_block_load_store_direct(self, block_load_store_bindings):
        """Test BlockLoad and BlockStore with DIRECT algorithm."""
        BlockLoad, BlockStore, shim_writer = block_load_store_bindings

        ffi = cffi.FFI()
        num_threads_per_block = 32
        items_per_thread = 1

        @cuda.jit(link=shim_writer.links())
        def try_block_load(d_input, d_output):
            block_load = BlockLoad(
                T=int32,
                BLOCK_DIM_X=num_threads_per_block,
                ITEMS_PER_THREAD=items_per_thread,
                ALGORITHM="cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT",
                BLOCK_DIM_Y=1,
                BLOCK_DIM_Z=1,
            )

            block_store = BlockStore(
                T=int32,
                BLOCK_DIM_X=num_threads_per_block,
                ITEMS_PER_THREAD=items_per_thread,
                ALGORITHM="cub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT",
                BLOCK_DIM_Y=1,
                BLOCK_DIM_Z=1,
            )

            thread_data = cuda.local.array(shape=items_per_thread, dtype=int32)

            d_input_ptr = ffi.from_buffer(d_input)
            d_output_ptr = ffi.from_buffer(d_output)
            thread_data_ptr = ffi.from_buffer(thread_data)

            block_load.Load(d_input_ptr, thread_data_ptr)

            cuda.syncthreads()

            block_store.Store(d_output_ptr, thread_data_ptr)

        input_data = np.arange(32, 64, dtype=np.int32)
        d_input = cuda.to_device(input_data)
        d_output = cuda.device_array(32, dtype=np.int32)

        try_block_load[1, 32](d_input, d_output)
        cuda.synchronize()

        result = d_output.copy_to_host()
        expected = input_data

        np.testing.assert_array_equal(result, expected)

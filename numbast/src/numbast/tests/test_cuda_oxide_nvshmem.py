# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in parity exercise for an NVSHMEM source/build tree."""

import json
import os
from pathlib import Path

import pytest

from numbast.tools.binding_config import CudaOxideConfig
from numbast.tools.cuda_oxide_binding_generator import (
    generate_cuda_oxide_bindings,
)

NVSHMEM_SOURCE_DIR = os.environ.get("NUMBAST_NVSHMEM_SOURCE_DIR")
pytestmark = pytest.mark.skipif(
    not NVSHMEM_SOURCE_DIR,
    reason="set NUMBAST_NVSHMEM_SOURCE_DIR for the NVSHMEM parity exercise",
)


def test_nvshmem_public_c_device_surface(tmp_path):
    source = Path(NVSHMEM_SOURCE_DIR)
    include = source / "src" / "include"
    symbol_inventory = os.environ.get("NUMBAST_NVSHMEM_SYMBOL_INVENTORY")
    ltoir = os.environ.get("NUMBAST_NVSHMEM_LTOIR", "libnvshmem_device.ltoir")

    cuda_oxide = {
        "LTOIR Inputs": [ltoir],
        "Bypass Parse Errors": True,
        "Type Aliases": {
            "nvshmem_team_t": "int32_t",
            "nvshmemx_team_t": "nvshmem_team_t",
            "nvshmemx_region_handle_t": "uint64_t",
            "nvshmemx_region_attrs_t": "nvshmemx_region_attrs",
        },
    }
    if symbol_inventory:
        cuda_oxide["Symbol Inventory"] = symbol_inventory

    config = CudaOxideConfig(
        {
            "Backend": "cuda-oxide",
            "Entry Point": str(
                source
                / "contrib"
                / "nvshmem4rust"
                / "generator"
                / "entry_point.h"
            ),
            "File List": [
                str(include / "device" / "nvshmem_coll_defines.cuh"),
                str(include / "device" / "nvshmem_defines.h"),
                str(include / "device" / "nvshmemx_coll_defines.cuh"),
                str(include / "device" / "nvshmemx_defines.h"),
                str(include / "device_host" / "nvshmem_types.h"),
            ],
            "GPU Arch": [os.environ.get("NUMBAST_NVSHMEM_GPU_ARCH", "sm_80")],
            "Clang Include Paths": [str(include)],
            "Predefined Macros": ["NVSHMEM_BUILD_LTOIR_LIBRARY"],
            "Exclude": {"Function": ["atomicAdd_system"]},
            "Skip Prefix": "nvshmemi_",
            "CUDA Oxide": cuda_oxide,
        }
    )
    rust_path, manifest_path = generate_cuda_oxide_bindings(config, tmp_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    symbols = {item["native_name"] for item in manifest["symbols"]}

    assert len(symbols) > 2_000
    assert all(name.startswith(("nvshmem_", "nvshmemx_")) for name in symbols)
    assert "nvshmem_int_p" in symbols
    assert "nvshmemx_int_put_block" in symbols
    cuda_aliases = {
        item["name"]: item for item in manifest["types"]["cuda_abi_aliases"]
    }
    assert cuda_aliases["double2"] == {
        "name": "double2",
        "rust_type": "[u128; 1]",
        "size": 16,
        "alignment": 16,
    }
    assert manifest["types"]["records"] == [
        {
            "name": "nvshmemx_region_attrs",
            "size": 64,
            "alignment": 4,
            "storage_type": "[u32; 16]",
            "fields": [["hints", "uint32_t"], ["reserved", "char[60]"]],
        }
    ]
    smem_enum = next(
        item
        for item in manifest["types"]["enums"]
        if item["name"] == "nvshmemx_smem_amount_t"
    )
    assert smem_enum["rust_underlying_type"] == "u32"
    compatibility = manifest["compatibility"]
    modern_only = compatibility["modern_nvvm_required_symbols"]
    assert modern_only
    assert set(modern_only).issubset(symbols)
    architecture = int(config.gpu_arch[0].split("_", 1)[1].rstrip("a"))
    assert compatibility["selected_arch_supports_all_symbols"] is (
        architecture >= 100
    )
    assert 'unsafe extern "C"' in Path(rust_path).read_text(encoding="utf-8")
    if symbol_inventory:
        assert manifest["symbol_verification"]["status"] == "verified"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ast_canopy import parse_declarations_from_source

from numbast.utils import make_function_shim


def test_make_function_shim_preserves_multidimensional_array_pointer(tmp_path):
    header = tmp_path / "matrix.cuh"
    header.write_text(
        "__device__ void get_matrix_3x4(float out[3][4]);\n",
        encoding="utf-8",
    )

    decls = parse_declarations_from_source(str(header), [str(header)], "sm_50")
    func_decl = decls.functions[0]

    shim = make_function_shim(
        "shim",
        func_decl.name,
        func_decl.return_type.unqualified_non_ref_type_name,
        func_decl.params,
    )

    assert "float (**out)[4]" in shim
    assert "get_matrix_3x4(*out);" in shim

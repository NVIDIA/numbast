# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from numba import cuda


def test_multiple_decls(make_binding):
    res1 = make_binding("multiple_decls.cuh", {}, {})
    binding = res1["bindings"]
    src = res1["src"]

    pat = r"def _lower_\w+_nbst\(shim_stream, shim_obj\):"
    matches = re.findall(pat, src)
    assert len(matches) == 1, "Expected exactly one lower function"

    lower_func_name = matches[0]
    assert "_lower__Z3fooii_nbst" in lower_func_name, (
        "Expected _lower__Z3fooii_nbst"
    )

    foo = binding["foo"]

    @cuda.jit
    def kernel():
        foo(1, 2)

    kernel[1, 1]()

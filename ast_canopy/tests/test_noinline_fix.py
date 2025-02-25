# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import redirect_stdout
from io import StringIO

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def sample_noinline_macro(data_folder):
    return data_folder / "noinline_macro_fix.cu"


def test_noinline_macro(sample_noinline_macro):
    srcstr = str(sample_noinline_macro)

    buf = StringIO()
    with redirect_stdout(buf):
        parse_declarations_from_source(srcstr, [srcstr], "sm_80", verbose=True)

    assert "error: use of undeclared identifier 'noinline'" not in buf.getvalue()

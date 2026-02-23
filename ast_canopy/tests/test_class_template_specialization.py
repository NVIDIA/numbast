# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def sample_ctsd(data_folder):
    return data_folder / "sample_ctsd.cu"


def test_ctpsd_unparsed_in_structs(sample_ctsd):
    srcstr = str(sample_ctsd)

    decls = parse_declarations_from_source(srcstr, [srcstr], "sm_80")
    ctsd = decls.class_template_specializations

    # At this stage, assert that the ctpsd is not parsed.
    assert len(ctsd) == 1
    assert ctsd[0].specialized_name == "BlockScan<int, 128>"

    assert len(ctsd[0].fields) == 0
    assert len(ctsd[0].methods) == 2
    assert len(ctsd[0].templated_methods) == 0

    # FIXME: Should be 0, CTSD creates an implicit Record for the underlying struct.
    # It's actually the same record as the one specialized.
    assert len(ctsd[0].nested_records) == 1

    assert len(ctsd[0].nested_class_templates) == 0

    # C++ standards says that an empty class has non-zero size.
    assert ctsd[0].sizeof_ != 0
    assert ctsd[0].alignof_ != 0

    assert len(ctsd[0].actual_template_arguments) == 2
    assert ctsd[0].actual_template_arguments[0] == "int"
    assert ctsd[0].actual_template_arguments[1] == "128"

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def sample_ctpsd(data_folder):
    return data_folder / "sample_ctpsd.cu"


def test_ctpsd_unparsed_in_structs(sample_ctpsd):
    srcstr = str(sample_ctpsd)

    decls = parse_declarations_from_source(srcstr, [srcstr], "sm_80")
    ct = decls.class_templates

    # At this stage, assert that the ctpsd is not parsed.
    assert len(ct) == 1
    assert ct[0].record.name == "foo"
    assert len(ct[0].template_parameters) == 2

import os

from ast_canopy import parse_declarations_from_source


def test_include_cccl_headers(data_folder):
    path = os.path.join(data_folder, "sample_cccl_include.cu")

    parse_declarations_from_source(
        path,
        [path],
        "sm_80",
    )

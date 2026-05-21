import pytest

import os

from ast_canopy import parse_declarations_from_source
from numba_cuda_mlir import cuda


@pytest.fixture(scope="session")
def decl_of():
    def _path(file_name: str):
        return os.path.join(os.path.dirname(__file__), "data", file_name)

    def _decl(file_name: str):
        path = _path(file_name)
        major, minor = cuda.get_current_device().compute_capability
        return parse_declarations_from_source(
            path, [path], f"sm_{major}{minor}"
        ), path

    return _decl

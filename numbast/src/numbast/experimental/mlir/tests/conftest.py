import pytest

import os

import ast_canopy


@pytest.fixture(scope="session")
def decl_of():
    def _path(file_name: str):
        return os.path.join(os.path.dirname(__file__), "data", file_name)

    def _decl(file_name: str):
        path = _path(file_name)
        return ast_canopy.parse_decl_only_from_src_current_sm(path), path

    return _decl

import ast_canopy


def test_import():
    ver = ast_canopy.__version__
    assert "\n" not in ver  # version string should be clean

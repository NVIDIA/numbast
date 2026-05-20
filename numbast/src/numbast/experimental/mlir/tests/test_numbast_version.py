import numbast


def test_import():
    ver = numbast.__version__
    assert "\n" not in ver

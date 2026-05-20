import os
import subprocess
import sys

import pytest


@pytest.mark.skip(reason="Add config item for backend types.")
def test_demo():
    dir = os.path.dirname(__file__)
    demo = os.path.join(dir, "demo.py")
    subprocess.run([sys.executable, demo], check=True, cwd=dir)

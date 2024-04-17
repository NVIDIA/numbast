import os
import subprocess


def test_demo():
    dir = os.path.dirname(__file__)
    demo = os.path.join(dir, "demo.py")
    subprocess.run(["python", demo], check=True, cwd=dir)

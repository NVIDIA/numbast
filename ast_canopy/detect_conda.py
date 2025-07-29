#!/usr/bin/env python3
import os
import sys

# Detect if current environment is in conda
is_conda_env = os.path.exists(os.path.join(sys.prefix, "conda-meta"))

if is_conda_env:
    print("true")
else:
    print("false")

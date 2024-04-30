#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

apt update --yes

apt install lsb-release wget software-properties-common gnupg zstd gcc-12
g++-12 libedit-dev -y

wget https://apt.llvm.org/llvm.sh
bash llvm.sh 18

apt-get install libclang-18-dev -y

update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 10
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 10


python -m pip install git+https://github.com/rapidsai/pynvjitlink.git@v0.2.2
python -m pip install ml-dtypes


# missing clang++
ln -s /usr/bin/clang++-18 /usr/bin/clang++

# Install AST_Canopy, Numbast and extensions
pip install ast_canopy/
pip install numbast/
pip install numba_extensions/bf16

# Run tests
pytest \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-numbast.xml" \
    numba_extensions/tests/test_torch_bf16.py

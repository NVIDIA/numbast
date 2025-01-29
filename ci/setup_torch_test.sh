#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

apt update --yes

# ARM PyTorch binaries are not well-supported.  As such, we
# need to add LLVM/GCC12/etc manually because NGC-ARM-PyTorch image does not
# ship with these dependencies for building numbast/bf16.
apt install lsb-release wget software-properties-common gnupg zstd gcc-12 g++-12 libedit-dev -y

wget https://apt.llvm.org/llvm.sh
bash llvm.sh 18

apt-get install libclang-18-dev -y

update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 10
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 10


python -m pip install pynvjitlink-cu12
python -m pip install ml-dtypes


# missing clang++
ln -s /usr/bin/clang++-18 /usr/bin/clang++

# Install AST_Canopy, Numbast and extensions
ast_canopy/build.sh
pip install numbast/

# bf16 is now in numbast_extensions.bf16
pip install numbast_extensions/

# Run tests
pytest \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-numbast.xml" \
    numbast_extensions/tests/test_torch_bf16.py

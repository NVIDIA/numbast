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


python -m pip install git+https://github.com/rapidsai/pynvjitlink.git@v0.2.2
python -m pip install ml-dtypes


# missing clang++
ln -s /usr/bin/clang++-18 /usr/bin/clang++

# Install AST_Canopy, Numbast and extensions
# FIXME: Current build system cannot auto install libastcanopy.so
# into system's lib path. While this file is packaged in the wheel,
# it needs to be moved manually. To be fixed by a new build system.
ast_canopy/build.sh
pip install numbast/
pip install numba_extensions/bf16

# Run tests
pytest \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-numbast.xml" \
    numba_extensions/tests/test_torch_bf16.py

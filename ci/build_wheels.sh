#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

LLVM_PREFIX=${LLVM_PREFIX:-${GITHUB_WORKSPACE}/llvm-install}

echo "=========================================="
echo "Building project with LLVM"
echo "LLVM prefix: ${LLVM_PREFIX}"
echo "=========================================="

# Set environment variables for ast_canopy build
if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
  export CMAKE_PREFIX_PATH="${LLVM_PREFIX}:${CMAKE_PREFIX_PATH}"
else
  export CMAKE_PREFIX_PATH="${LLVM_PREFIX}"
fi

export CMAKE_INSTALL_PREFIX="$(pwd)/install"

# Build the ast_canopy C++ library
cd "${GITHUB_WORKSPACE}/ast_canopy"
echo "Building ast_canopy..."
./build.sh --llvm-linkage=STATIC --no-install
cmake --install cpp/build
echo "Build completed successfully!"

# Now build the python wheels
export CMAKE_PREFIX_PATH="${CMAKE_INSTALL_PREFIX}:${CMAKE_PREFIX_PATH}"
for py_ver in 3.10 3.11 3.12 3.13; do
  python"${py_ver}" -m pip wheel -w dist -v --disable-pip-version-check .
done

# Ensure auditwheeel can see the ast_canopy library before repairing the wheels
export LD_LIBRARY_PATH="${CMAKE_INSTALL_PREFIX}/lib64"
auditwheel repair -w ../final-dist dist/*.whl

# Now build the numbast python wheels
cd ../numbast
python3.13 -m pip wheel -w dist -v \
  --disable-pip-version-check \
  --find-links=../final-dist \
  --pre \
  .
cp dist/numbast*.whl ../final-dist/

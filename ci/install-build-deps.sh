#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set -euo pipefail

NINJA_VERSION=1.13.1
SCCACHE_VERSION=0.10.0
OS_ARCH=$(uname -m)

if [[ ! -d build-deps ]]; then
  echo "Downloading build dependencies"
  mkdir -p build-deps
  cd build-deps
  if [[ "${OS_ARCH}" == "x86_64" ]]; then
    curl -sL https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/ninja-linux.zip -o ninja.zip
  elif [[ "${OS_ARCH}" == "aarch64" ]]; then
    curl -sL https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/ninja-linux-aarch64.zip -o ninja.zip
  else
    echo "Unsupported architecture: ${OS_ARCH}"
    exit 1
  fi
  curl -sL https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-${OS_ARCH}-unknown-linux-musl.tar.gz -o sccache.tar.gz
  cd ..
fi

unzip build-deps/ninja.zip
tar -xf build-deps/sccache.tar.gz

mv ninja /usr/local/bin/
mv sccache-v${SCCACHE_VERSION}-${OS_ARCH}-unknown-linux-musl/sccache /usr/local/bin/

echo "Installed ninja and sccache"

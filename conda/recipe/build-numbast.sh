#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -exuo pipefail

pushd numbast
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NUMBAST="${PKG_VERSION}"
${PYTHON} -m pip install . -vv --no-deps --no-build-isolation
popd

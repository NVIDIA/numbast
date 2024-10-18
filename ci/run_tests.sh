#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run tests
pytest \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-ast_canopy.xml" \
    ast_canopy/

pytest \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-numbast.xml" \
    numbast/

pytest \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-numbast_extensions.xml" \
    numbast_extensions/

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

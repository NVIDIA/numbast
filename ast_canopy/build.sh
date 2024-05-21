#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

$PYTHON -m pip install ast_canopy/ -vv && \
    mv $SP_DIR/libastcanopy.so $PREFIX/lib

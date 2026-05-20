// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

struct Bar {
  Bar() = default;
};

Bar __device__ __forceinline__ operator+(const Bar &a, const Bar &b) {
  return Bar();
}

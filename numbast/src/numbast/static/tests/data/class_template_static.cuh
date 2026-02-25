// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

template <typename T> struct TemplateBox {
  T value;

  __device__ TemplateBox(T v) : value(v) {}
  __device__ T get() const { return value; }
};

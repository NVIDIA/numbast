// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

int add(int a, int b) { return a + b; }

float scale(float value, float factor) { return value * factor; }

void set_value(int *out, int value) { *out = value; }

double __host__ host_offset(double x, double offset) { return x + offset; }

int __host__ __device__ add_host_device(int a, int b) { return a + b; }

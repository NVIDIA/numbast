// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

struct PlainStruct {
  int a;
  float b;
};

union PlainUnion {
  int as_int;
  float as_float;
};

class PlainClass {
public:
  int a;
};

struct WithNested {
  int tag;
  union Payload {
    int as_int;
    double as_double;
  } payload;
};

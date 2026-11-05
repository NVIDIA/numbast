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

// All three kinds nested under one parent, so a single parse can show that
// nesting preserves the kind rather than flattening it to one answer.
struct WithNested {
  int tag;
  union NestedUnion {
    int as_int;
    double as_double;
  } nested_union;
  struct NestedStruct {
    int a;
  } nested_struct;
  class NestedClass {
  public:
    int a;
  } nested_class;
};

// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <stdint.h>

typedef int32_t team_t;
typedef team_t team_alias_t;
typedef uint64_t region_handle_t;
typedef const int const_int_t;
typedef int *int_ptr_t;
typedef int_ptr_t pointer_alias_t;
typedef int int_array_t[4];
typedef void (*callback_t)(int);

enum Status { success = 0 };
typedef Status status_t;

struct Record {
  int value;
};
typedef Record record_t;
typedef record_t record_alias_t;

union Storage {
  int value;
  float other;
};
typedef Storage storage_t;

// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <stdint.h>

constexpr int BASE = 6;
const int GLOBAL_COUNT = BASE * 7;
static const unsigned long long MAX_U64 = 18446744073709551615ULL;
constexpr bool ENABLED = true;
const bool DISABLED = false;
constexpr double RATIO = 1.2345678901234567;
const float SCALE = 1.25f;
typedef int32_t handle_t;
constexpr handle_t DEFAULT_HANDLE = -1;
enum Status { ready = 2 };
const Status DEFAULT_STATUS = ready;
namespace library {
inline constexpr int NAMESPACE_VALUE = 8;
}
int mutable_global = 3;
volatile const int VOLATILE_VALUE = 4;
extern const int EXTERNAL_VALUE;
int runtime_value();
const int RUNTIME_VALUE = runtime_value();
const char TEXT[] = "ignored";
constexpr const int *NULL_POINTER = nullptr;
struct Object {
  int member;
};
constexpr Object OBJECT = {1};
struct ClassScope {
  static constexpr int MEMBER = 9;
  static const int OUT_OF_LINE;
};
const int ClassScope::OUT_OF_LINE = 12;

template <class T> constexpr int TEMPLATE_VALUE = sizeof(T);
void locals() {
  constexpr int LOCAL = 10;
  static const int STATIC_LOCAL = 11;
}

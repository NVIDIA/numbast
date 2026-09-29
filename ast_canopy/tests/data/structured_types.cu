// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

struct WordStorage {
  unsigned short value;
};

typedef WordStorage word_t;

enum Status : unsigned char { success, failure };

struct StructuredTypes {
  const int *pointer_to_const;
  int *const const_pointer;
  int *array_of_pointers[4];
  int (*pointer_to_array)[4];
  int (*callback)(float);
  word_t *alias;
  Status status;
};

void references(int &left, int &&right);

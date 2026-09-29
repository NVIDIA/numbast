// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

struct CudaOxideTypes {
  const int *pointer_to_const;
  int *const const_pointer;
  int *array_of_pointers[4];
  int (*pointer_to_array)[4];
};

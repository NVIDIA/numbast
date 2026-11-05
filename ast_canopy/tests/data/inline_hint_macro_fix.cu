// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#define NVSHMEMI_DEVICE_INLINE __inline_hint__

__device__ NVSHMEMI_DEVICE_INLINE void inline_hint_void() {}

__device__ NVSHMEMI_DEVICE_INLINE int inline_hint_int() { return 1; }

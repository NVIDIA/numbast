// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * @brief Writes x + 1 into the provided output reference.
 *
 * @param[out] out Reference that will be assigned the value `x + 1`.
 * @param[in] x Input integer whose value is incremented by one and stored in `out`.
 */

void __device__ add_out(int &out, int x) { out = x + 1; }

/**
 * @brief Stores (x + 2) into the output reference and returns (x + 3).
 *
 * @param out Reference that will be assigned the value x + 2.
 * @param x Input value used to compute the stored and returned results.
 * @return int The value x + 3.
 */
int __device__ add_out_ret(int &out, int x) {
  out = x + 2;
  return x + 3;
}

/**
 * @brief Computes the value five greater than the referenced integer.
 *
 * @param x Input integer passed by reference; the function does not modify it.
 * @return int The value of `x + 5`.
 */
int __device__ add_in_ref(int &x) { return x + 5; }

/**
 * @brief Increments the referenced integer by a specified amount.
 *
 * @param x Reference to the integer that will be increased in place.
 * @param delta Amount to add to `x`.
 */
void __device__ add_inout_ref(int &x, int delta) { x += delta; }

static __device__ const float4 transform_rows[3] = {
    {1.0f, 2.0f, 3.0f, 4.0f},
    {5.0f, 6.0f, 7.0f, 8.0f},
    {9.0f, 10.0f, 11.0f, 12.0f},
};

const float4 *__device__ get_transform(int handle) {
  (void)handle;
  return transform_rows;
}

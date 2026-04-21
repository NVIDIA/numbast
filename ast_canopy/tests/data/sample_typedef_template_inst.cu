// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

// Minimal repro for the typedef-to-template-instantiation crash in
// ast_canopy/cpp/src/typedef.cpp.
//
// The typedef matcher builds a map (record_id_to_name) from record IDs
// to names that the record matcher captured. For a typedef whose
// underlying type is a class template instantiation, the underlying
// record's ID is NOT in that map (class template specialisations are
// captured on a separate matcher list). The old code called
// record_id_to_name->at(id), which threw std::out_of_range and aborted
// parsing of the entire header.
//
// The fix falls back to find() with a safe default (the record's own
// name from Clang) when the ID is missing.

#pragma once

template <typename T, int N> struct Storage {
  T data[N];
};

// Both typedefs point at template instantiations whose IDs are not in
// record_id_to_name. Prior to the fix, parsing these lines threw.
typedef Storage<float, 3> Vec3fStorage;
typedef Storage<double, 4> Vec4dStorage;

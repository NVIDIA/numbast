// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

// Minimal repro for the typedef-to-template-instantiation crash in
// ast_canopy/cpp/src/typedef.cpp.
//
// The typedef matcher builds a map (record_id_to_name) from record IDs
// to names that the record/class-template-specialization matchers captured.
// For a typedef whose underlying type is a class template instantiation, the
// underlying record's ID was missing from that map because class template
// specialisations are captured on a separate matcher list. The old code called
// record_id_to_name->at(id), which threw std::out_of_range and aborted parsing
// of the entire header.
//
// The fix registers class template specialization IDs in the shared map and
// keeps the typedef lookup defensive for other unregistered record-like types.

#pragma once

template <typename T, int N> struct Storage {
  T data[N];
};

// Both typedefs point at template instantiations. Prior to the fix, their
// underlying record IDs were not in record_id_to_name and parsing these lines
// threw.
typedef Storage<float, 3> Vec3fStorage;
typedef Storage<double, 4> Vec4dStorage;

// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include <ast_canopy/source_location.hpp>

namespace ast_canopy {

SourceLocation::SourceLocation() : _is_valid(false) {}

SourceLocation::SourceLocation(const std::string &file_name,
                               const unsigned int line,
                               const unsigned int column, const bool is_valid)
    : _file_name(file_name), _line(line), _column(column), _is_valid(is_valid) {
}

} // namespace ast_canopy

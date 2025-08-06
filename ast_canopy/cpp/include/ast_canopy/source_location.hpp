// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

#include <string>

namespace ast_canopy {

class SourceLocation {
public:
  SourceLocation();
  SourceLocation(const std::string &file_name, const unsigned int line,
                 const unsigned int column, const bool is_valid);

  bool is_valid() const { return _is_valid; }
  std::string file_name() const { return _file_name; }
  unsigned int line() const { return _line; }
  unsigned int column() const { return _column; }

private:
  bool _is_valid;
  std::string _file_name;
  unsigned int _line;
  unsigned int _column;
};

}; // namespace ast_canopy

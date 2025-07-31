// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#pragma once

#include <stdexcept>

namespace ast_canopy {

class ParseError : public std::runtime_error {
public:
  ParseError(std::string &err) : std::runtime_error(err) {}
};

} // namespace ast_canopy

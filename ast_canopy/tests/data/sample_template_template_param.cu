// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

// Minimal repro for the TemplateTemplateParmDecl crash in
// ast_canopy/cpp/src/template_param.cpp.
//
// Prior to the fix, the TemplateParam constructor for
// TemplateTemplateParmDecl threw std::runtime_error, which aborted
// parsing of any class template that used a template template parameter.

#pragma once

template <typename T> struct SimpleContainer {
  T item;
};

// Class template with a template template parameter.
// Parsing this declaration used to throw inside TemplateParam(TTPD).
template <typename T, template <typename> class Container> struct Adapter {
  Container<T> value;
};

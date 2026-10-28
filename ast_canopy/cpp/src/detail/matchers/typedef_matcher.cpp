// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on

#include "matchers.hpp"

#ifndef NDEBUG
#include <iostream>
#endif

namespace ast_canopy {

namespace detail {

TypedefMatcher::TypedefMatcher(traverse_ast_payload *payload)
    : payload(payload) {}

void TypedefMatcher::run(const MatchFinder::MatchResult &Result) {
  const TypedefDecl *TD = Result.Nodes.getNodeAs<clang::TypedefDecl>("typedef");
  std::string file_name = source_filename_from_decl(TD);

  if (std::any_of(payload->files_to_retain->begin(),
                  payload->files_to_retain->end(),
                  [&file_name](const std::string &file_to_retain) {
                    return file_name == file_to_retain;
                  }))

  {
#ifndef NDEBUG
    std::string source_range =
        TD->getSourceRange().printToString(*Result.SourceManager);
    std::cout << source_range << std::endl;
    std::cout << std::endl;
#endif
    // Every typedef in a retained file is reported, whatever it aliases.
    //
    // Which typedefs matter is the consumer's policy, not this matcher's: a
    // binding generator usually wants the canonical type a name resolves to,
    // while a dialect generator needs the public name itself, because a C
    // library's typedefs *are* its type vocabulary -- `CUstream` is the API
    // and `CUstream_st` is an implementation detail that exists only so the
    // pointer has something to point at.
    //
    // Restricting this to `isStructureOrClassType()` dropped exactly the two
    // forms a C API is built from: `typedef struct Foo_st *Foo;`, whose
    // underlying type is a pointer rather than a record, and
    // `typedef enum Foo_enum { ... } Foo;`, whose underlying type is a tag
    // but not a record. Both were discarded with no record that they existed.
    payload->decls->typedefs.push_back(Typedef(TD, payload->record_id_to_name));
  }
}

} // namespace detail
} // namespace ast_canopy

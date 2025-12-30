// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// This file is intentionally a *test fixture* for `qual_name` behavior.
//
// `qual_name` in ast_canopy is derived from Clang's
// `Decl::getQualifiedNameAsString()` (with small stability tweaks for anonymous
// records). These examples are meant to serve as executable documentation.

// --- 1) No namespace (global scope) -----------------------------------------

enum GlobalE { GA = 1 };

struct GlobalS {
  int m(int x) { return x; }
};

typedef GlobalS GlobalAlias;

int global_f(int a) { return a; }

template <typename T> T global_tf(T x) { return x; }

template <typename T> struct GlobalTpl {
  T v;
};

// --- 2) Anonymous record created in a C style typedef -----------------------
//
// In Clang, the underlying RecordDecl for `typedef struct { ... } Name;`
// is anonymous (no "tag" name). Clang typically reports an empty record name.
// ast_canopy then falls back to `unnamed<ID>` using Clang's internal Decl ID so
// downstream always has something printable (not stable across runs).

typedef struct {
  int a;
  int b;
} CStyleAnon;

// --- 3) Anonymous namespace (namespace { ... }) -----------------------------
//
// Clang typically prints anonymous namespaces as `(anonymous namespace)` in
// qualified names. We rely on that string as the user-visible marker.

namespace {
struct AnonNS_S {
  int m(int x) { return x; }
};

int anon_ns_f(int x) { return x; }
} // namespace

// --- 4) Nested namespaces ---------------------------------------------------

namespace ns1 {
namespace ns2 {

enum E { A = 1 };

struct S {
  int m(int x) { return x; }
};

typedef S Alias;

int f(int a) { return a; }

template <typename T> T tf(T x) { return x; }

template <typename T> struct Tpl {
  T v;
};

} // namespace ns2
} // namespace ns1

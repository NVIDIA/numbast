// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on
class Foo {
  void Bar1(); // Private
public:
  void Bar2(); // Public
private:
  void Bar3(); // Private
protected:
  void Bar4(); // Protected
};

struct Bar {
  void Bar1(); // Public
public:
  void Bar2(); // Public
private:
  void Bar3(); // Private
protected:
  void Bar4(); // Protected
};

struct Baz {
private:
  int x;

public:
  int y;

protected:
  int z;
};

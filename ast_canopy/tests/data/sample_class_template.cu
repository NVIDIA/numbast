enum class E { A, B, C };

template <typename T, E e> struct Foo {
  Foo() {}

  template <typename U> E bar(T t, U u) { return e; }

  void baz() {}

  T t;
};

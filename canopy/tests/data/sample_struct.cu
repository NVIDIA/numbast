struct alignas(16) foo {
  float2 a;

  foo(float2 a_) : a(a_) {}

  foo(foo &&other) : a(other.a) { other.a = float2(); }

  double add(double a, double b) { return a + b; }

  operator float2() { return a; }

  // unary
  foo operator-() const { return foo(float2{-a.x, -a.y}); }

  // binary
  foo operator+(const foo &other) const {
    return foo(float2{a.x + other.a.x, a.y + other.a.y});
  }

  // compound assignment
  foo &operator+=(const foo &other) {
    a.x += other.a.x;
    a.y += other.a.y;
    return *this;
  }

  // comparison
  bool operator==(const foo &other) const {
    return a.x == other.a.x && a.y == other.a.y;
  }

  // subscript
  float &operator[](int i) { return i == 0 ? a.x : a.y; }

  // callable
  float operator()(float x) { return a.x * x + a.y; }

  // copy assignment operator
  foo &operator=(const foo &other) {
    a = other.a;
    return *this;
  }

  template <typename T> T bar() {}
};

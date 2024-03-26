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

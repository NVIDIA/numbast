// Minimal reproduction cases for ast_canopy crashes on complex template
// hierarchies. Each section targets a specific crash root cause.

#pragma once

// ============================================================
// Case 1: CRTP with dependent types (mangleName crash)
// A class template method has dependent parameter/return types.
// The Itanium name mangler segfaults on these if invoked directly.
// ============================================================

template <typename Derived> struct CRTPBase {
  __device__ Derived &derived() { return static_cast<Derived &>(*this); }
  __device__ const Derived &derived() const {
    return static_cast<const Derived &>(*this);
  }
  // Method with dependent return type
  __device__ Derived add(const Derived &other) const { return derived(); }
};

template <typename Scalar> struct Vec3 : public CRTPBase<Vec3<Scalar>> {
  Scalar x, y, z;
  __host__ __device__ Vec3() : x(0), y(0), z(0) {}
  __host__ __device__ Vec3(Scalar a, Scalar b, Scalar c) : x(a), y(b), z(c) {}
};

// ============================================================
// Case 2: Typedef to a class template instantiation
// The underlying record may not be in record_id_to_name if it
// comes from a different file or is a template instantiation.
// ============================================================

template <typename T, int N> struct Storage {
  T data[N];
};

typedef Storage<float, 3> Vec3fStorage;
typedef Storage<double, 4> Vec4dStorage;

// ============================================================
// Case 3: Template template parameter
// TemplateTemplateParmDecl must be handled without crashing.
// ============================================================

template <typename T, template <typename> class Container> struct Adapter {
  Container<T> value;
};

template <typename T> struct SimpleContainer {
  T item;
};

// ============================================================
// Case 4: Dependent types in function signatures
// Function templates with complex dependent return types.
// ============================================================

template <typename T>
__device__ typename T::value_type extract_value(const T &container);

// ============================================================
// Case 5: Multiple inheritance CRTP (deep hierarchy)
// Tests that deeply nested dependent types don't crash.
// ============================================================

template <typename Derived> struct Level1 {
  __device__ void method1() {}
};

template <typename Derived> struct Level2 : public Level1<Derived> {
  __device__ Derived &self() { return static_cast<Derived &>(*this); }
};

template <typename Scalar> struct Concrete : public Level2<Concrete<Scalar>> {
  Scalar val;
  __host__ __device__ Concrete() : val(0) {}
};

// ============================================================
// Case 6: Non-type template parameter expressions
// Class template specializations with expressions as args.
// ============================================================

template <typename T, int N, bool Aligned = (N > 4)> struct AlignedStorage {
  T data[N];
};

// Force instantiation of specialization with expression arg
struct TestInst {
  AlignedStorage<float, 3> small; // Aligned = false
  AlignedStorage<float, 8> large; // Aligned = true
};

// ============================================================
// Concrete functions using the above types (for files_to_retain)
// ============================================================

__device__ float vec3_dot(Vec3<float> a, Vec3<float> b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

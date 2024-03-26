enum class E { A, B, C };

// Type Template, Non-type template
template <typename T, int N, E e> void __device__ foo(T t) {}

// Min required arg == 0?
template <typename T = int> void __device__ bar(T t) {}

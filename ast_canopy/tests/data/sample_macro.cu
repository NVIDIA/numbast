#define MAKE_TYPENAME_FUNC(TYPE)                                               \
  TYPE __device__ forty_two_##TYPE() { return TYPE(42); }

MAKE_TYPENAME_FUNC(int)
MAKE_TYPENAME_FUNC(float)
MAKE_TYPENAME_FUNC(double)

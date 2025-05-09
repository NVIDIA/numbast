template <typename A, typename B> struct foo {};

template <typename T> struct foo<T, T *> {};

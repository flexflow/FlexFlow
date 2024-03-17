#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_HASH_UTILS_CORE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_HASH_UTILS_CORE_H

#include <functional>

namespace FlexFlow {

// tuple hashing pulled from
// https://www.variadic.xyz/2018/01/15/hashing-stdpair-and-stdtuple/
template <class T>
inline void hash_combine(std::size_t &seed, T const &v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <class T, class... Ts>
inline void hash_combine(std::size_t &seed, T const &v, Ts... rest) {
  hash_combine(seed, v);
  hash_combine(seed, rest...);
}

} // namespace FlexFlow

#endif

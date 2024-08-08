#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_HASH_UTILS_CORE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_HASH_UTILS_CORE_H

#include <functional>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace FlexFlow {

template <class T>
std::size_t get_std_hash(T const &v) {
  std::hash<T> hasher;
  return hasher(v);
}

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

template <typename T>
void unordered_hash_combine(std::size_t &seed, T const &t) {
  seed += get_std_hash(t);
}

template <typename T>
void unordered_container_hash(std::size_t &seed, T const &t) {
  hash_combine(seed, t.size());
  size_t total = 0;
  for (auto const &v : t) {
    unordered_hash_combine(total, v);
  }
  hash_combine(seed, total);
}

template <typename It>
void iter_hash(std::size_t &seed, It start, It end) {
  hash_combine(seed, std::distance(start, end));
  for (; start < end; start++) {
    hash_combine(seed, *start);
  }
}

} // namespace FlexFlow

#endif

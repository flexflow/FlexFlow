#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNORDERED_SET_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNORDERED_SET_OF_H

#include "utils/hash/pair.h"
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

template <typename C, typename T = typename C::value_type>
std::unordered_set<T> unordered_set_of(C const &c) {
  return std::unordered_set<T>{c.cbegin(), c.cend()};
}

template <typename K, typename V>
std::unordered_set<std::pair<K, V>>
    unordered_set_of(std::unordered_map<K, V> const &m) {
  std::unordered_set<std::pair<K, V>> result;
  for (auto const &[k, v] : m) {
    result.insert({k, v});
  }
  return result;
}

} // namespace FlexFlow

#endif

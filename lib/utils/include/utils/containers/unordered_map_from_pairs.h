#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNORDERED_MAP_FROM_PAIRS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNORDERED_MAP_FROM_PAIRS_H

#include <unordered_map>

namespace FlexFlow {

template <typename C, typename K = typename C::value_type::first_type, typename V = typename C::value_type::second_type>
std::unordered_map<K, V>
  unordered_map_from_pairs(C const &c) {
  return std::unordered_map<K, V>(c.cbegin(), c.cend());
}

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_INVERT_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_INVERT_MAP_H

#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

template <typename C>
auto invert_map(C const &m) {
  std::unordered_map<typename C::mapped_type,
                     std::unordered_set<typename C::key_type>>
      m_inv;
  for (auto const &[key, value] : m) {
    m_inv[value].insert(key);
  }
  return m_inv;
}
} // namespace FlexFlow

#endif

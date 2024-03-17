#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPECLASS_MONOID_INSTANCES_UNORDERED_MAP_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPECLASS_MONOID_INSTANCES_UNORDERED_MAP_H

#include "utils/algorithms/typeclass/monoid/monoid.h"
#include "utils/backports/type_identity.h"
#include <unordered_map>

namespace FlexFlow {

template <typename K, typename V>
struct take_left_monoid {
  static std::unordered_map<K, V> mempty() {
    return {};
  }
  static void mappend_inplace(std::unordered_map<K, V> &lhs,
                              std::unordered_map<K, V> const &rhs) {
    for (auto const &[k, v] : rhs) {
      if (lhs.count(k) == 0) {
        lhs[k] = v;
      }
    }
  }
};

template <typename K, typename V>
struct take_right_monoid {
  static std::unordered_map<K, V> mempty() {
    return {};
  }
  static void mappend_inplace(std::unordered_map<K, V> &lhs,
                              std::unordered_map<K, V> const &rhs) {
    for (auto const &[k, v] : rhs) {
      lhs[k] = v;
    }
  }
};

template <typename K, typename V, typename Instance = default_monoid_t<V>>
struct mappend_values_monoid {
  static std::unordered_map<K, V> mempty() {
    return {};
  }
  static void mappend_inplace(std::unordered_map<K, V> &lhs,
                              std::unordered_map<K, V> const &rhs) {
    for (auto const &[k, v] : rhs) {
      if (lhs.count(k) == 0) {
        lhs[k] = v;
      } else {
        ::FlexFlow::mappend_inplace<V, Instance>(lhs.at(k), v);
      }
    }
  }
};

} // namespace FlexFlow

#endif

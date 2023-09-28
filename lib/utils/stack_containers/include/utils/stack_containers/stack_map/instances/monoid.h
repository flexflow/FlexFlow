#ifndef _FLEXFLOW_LIB_UTILS_STACK_CONTAINERS_INCLUDE_UTILS_STACK_CONTAINERS_STACK_MAP_INSTANCES_MONOID_H
#define _FLEXFLOW_LIB_UTILS_STACK_CONTAINERS_INCLUDE_UTILS_STACK_CONTAINERS_STACK_MAP_INSTANCES_MONOID_H

#include "utils/algorithms/typeclass/monoid/monoid.h"
#include "utils/stack_containers/stack_map/stack_map.h"
#include "utils/backports/type_identity.h"

namespace FlexFlow {

template <typename K, typename V, std::size_t MAXSIZE>
struct take_left_monoid {
  using M = stack_map<K, V, MAXSIZE>;

  static M mempty() { return {}; }
  static void mappend_inplace(M &lhs, M const &rhs) {
    for (auto const &[k, v] : rhs) {
      if (lhs.count(k) == 0) {
        lhs[k] = v;
      }
    }
  }
};

template <typename K, typename V, std::size_t MAXSIZE>
struct take_right_monoid {
  using M = stack_map<K, V, MAXSIZE>;

  static M mempty() { return {}; }
  static void mappend_inplace(M &lhs, M const &rhs) {
    for (auto const &[k, v] : rhs) {
      lhs[k] = v;
    }
  }
};

template <typename K, typename V, std::size_t MAXSIZE, typename Instance = default_monoid_t<V>>
struct mappend_values_monoid {
  using M = stack_map<K, V, MAXSIZE>;

  static M mempty() { return {}; }
  static void mappend_inplace(M &lhs, M const &rhs) {
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

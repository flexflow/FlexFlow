#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_INSTANCES_UNORDERED_SET_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_INSTANCES_UNORDERED_SET_H

#include "utils/algorithms/typeclass/monoid/monoid.h"
#include <unordered_set>
#include "utils/backports/type_identity.h"

namespace FlexFlow {

template <typename T>
struct unordered_set_monoid { 
  static std::unordered_set<T> mempty() { return {}; }
  static void mappend_inplace(std::unordered_set<T> &acc, std::unordered_set<T> const &val) {
    for (T const &t : val) {
      acc.insert(t);
    }
  }
};

template <typename T>
struct default_monoid<std::unordered_set<T>> : type_identity<unordered_set_monoid<T>> {};

} // namespace FlexFlow

#endif

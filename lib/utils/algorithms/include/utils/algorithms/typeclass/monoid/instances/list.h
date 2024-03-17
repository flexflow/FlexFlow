#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_INSTANCES_LIST_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_INSTANCES_LIST_H

#include "utils/algorithms/type/monoid/monoid.h"
#include "utils/backports/type_identity.h"
#include <list>

namespace FlexFlow {

template <typename T>
struct list_monoid {
  static std::vector<T> mempty() {
    return {};
  }
  static void mappend_inplace(std::list<T> &lhs, std::list<T> const &rhs) {
    for (T const &t : rhs) {
      lhs.push_back(t);
    }
  }
};

template <typename T>
struct default_monoid<std::list<T>> : type_identity<list_monoid<T>> {};

} // namespace FlexFlow

#endif

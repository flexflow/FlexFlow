#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_INSTANCES_VECTOR_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_INSTANCES_VECTOR_H

#include "utils/algorithms/typeclass/monoid/monoid.h"
#include "utils/backports/type_identity.h" 

namespace FlexFlow {

template <typename T>
struct vector_monoid { 
  static std::vector<T> mempty() { return {}; }
  static void mappend_inplace(std::vector<T> &lhs, std::vector<T> const &rhs) { 
    lhs.reserve(lhs.size() + std::distance(rhs.begin(), rhs.end()));
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
  }
};

template <typename T>
struct default_monoid<std::vector<T>> : type_identity<vector_monoid<T>> {};

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_STACK_CONTAINERS_INCLUDE_UTILS_STACK_CONTAINERS_STACK_VECTOR_INSTANCES_MONOID_H
#define _FLEXFLOW_LIB_UTILS_STACK_CONTAINERS_INCLUDE_UTILS_STACK_CONTAINERS_STACK_VECTOR_INSTANCES_MONOID_H

#include "utils/algorithms/typeclass/monoid/monoid.h"
#include "utils/backports/type_identity.h" 
#include "utils/stack_containers/stack_vector/stack_vector.h"

namespace FlexFlow {

template <typename T, std::size_t MAXSIZE>
struct stack_vector_monoid { 
  using M = stack_vector<T, MAXSIZE>;

  static M mempty() { return {}; }
  static void mappend_inplace(M &lhs, M const &rhs) { 
    for (auto const &v : rhs) {
      lhs.push_back(v);
    }
  }
};

template <typename T, std::size_t MAXSIZE>
struct default_monoid<stack_vector<T, MAXSIZE>> : type_identity<stack_vector_monoid<T, MAXSIZE>> {};

} // namespace FlexFlow

#endif

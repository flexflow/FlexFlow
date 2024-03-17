#ifndef _FLEXFLOW_LIB_UTILS_STACK_CONTAINERS_INCLUDE_UTILS_STACK_CONTAINERS_STACK_STRING_INSTANCES_MONOID_H
#define _FLEXFLOW_LIB_UTILS_STACK_CONTAINERS_INCLUDE_UTILS_STACK_CONTAINERS_STACK_STRING_INSTANCES_MONOID_H

#include "utils/algorithms/typeclass/monoid/monoid.h"
#include "utils/backports/type_identity.h"
#include <cstddef>

namespace FlexFlow {

template <typename Char, std::size_t MAXSIZE>
struct stack_basic_string_monoid {
  using M = stack_basic_string<Char, MAXSIZE>;

  static M mempty() {
    return {};
  }
  static void mappend_inplace(M &lhs, M const &rhs) {
    lhs += rhs;
  }
};

template <typename Char, std::size_t MAXSIZE>
struct default_monoid_t<stack_basic_string<Char, MAXSIZE>>
    : type_identity<stack_basic_string_monoid<Char, MAXSIZE>> {}

} // namespace FlexFlow

#endif

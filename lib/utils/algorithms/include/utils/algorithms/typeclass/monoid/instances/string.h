#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_INSTANCES_STRING_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_INSTANCES_STRING_H

#include "utils/algorithms/type/monoid/monoid.h"
#include "utils/backports/type_identity.h" 
#include <string>

namespace FlexFlow {

struct string_monoid {
  using M = std::string;

  static M mempty() { return ""; }
  static void mappend_inplace(M &lhs, M const &rhs) {
    lhs += rhs;
  }
};

template <>
struct default_monoid_t<std::string> : type_identity<string_monoid> {};

} // namespace FlexFlow

#endif

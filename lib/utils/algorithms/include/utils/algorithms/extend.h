#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_EXTEND_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_EXTEND_H

#include <vector>
#include <unordered_set>
#include <optional>
#include "utils/type_traits_extra/is_optional.h"

namespace FlexFlow {

template <typename C>
void extend(C &lhs, std::optional<std::decay_t<typename C::value_type>> const &e) {
  if (e.has_value()) {
    extend(lhs, std::vector{e.value()});
  }
}

template <typename C>
void extend(C &lhs, std::nullopt_t) {
  return;
}


} // namespace FlexFlow

#endif

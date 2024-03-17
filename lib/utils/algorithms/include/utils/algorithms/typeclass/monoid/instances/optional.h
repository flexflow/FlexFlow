#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_INSTANCES_OPTIONAL_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_INSTANCES_OPTIONAL_H

#include "utils/algorithms/typeclass/monoid/monoid.h"
#include <optional>

namespace FlexFlow {

template <typename T, typename Instance = default_monoid_t<T>>
struct optional_monoid {
  static std::optional<T> mempty() {
    return std::nullopt;
  }
  static void mappend_inplace(std::optional<T> &lhs,
                              std::optional<T> const &rhs) {
    if (lhs.has_value()) {
      Instance::mappend_inplace(lhs.value(), rhs.value_or(Instance::mempty()));
    } else {
      lhs = rhs;
    }
  }
};

template <typename T>
struct default_monoid<std::optional<T>>
    : optional_monoid<T, default_monoid_t<T>> {};

} // namespace FlexFlow

#endif

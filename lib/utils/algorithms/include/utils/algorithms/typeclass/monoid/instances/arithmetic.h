#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_ARITHMETIC_INSTANCE_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_ARITHMETIC_INSTANCE_H

#include "utils/algorithms/type/monoid/monoid.h"
#include <type_traits>

namespace FlexFlow {

template <typename T, typename Enable = void>
struct addition_monoid {};

template <typename T>
struct addition_monoid<T, std::enable_if_t<std::is_arithmetic_v<T>>> {
  static T mempty() {
    return static_cast<T>(0);
  }
  static void mappend_inplace(T &lhs, T const &rhs) {
    lhs += rhs;
  }
};

template <typename T>
struct is_commutative_monoid<addition_monoid<T>> : std::true_type {};

template <typename T, typename Enable = void>
struct product_monoid {};

template <typename T>
struct product_monoid<T, std::enable_if_t<std::is_arithmetic_v<T>>> {
  static T mempty() {
    return static_cast<T>(1);
  }
  static void mappend_inplace(T &lhs, T const &rhs) {
    lhs *= rhs;
  }
};

template <typename T>
struct is_commutative_monoid<product_monoid<T>> : std::true_type {};

} // namespace FlexFlow

#endif

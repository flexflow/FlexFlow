#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_H

#include "utils/type_traits_extra/is_ordered.h"
#include "utils/type_traits_extra/iterator.h"
#include <iterator>
#include <type_traits>

namespace FlexFlow {

template <typename T, typename Enable = void>
struct default_monoid {};

template <typename T>
using default_monoid_t = typename default_monoid<T>::type;

template <typename T, typename Instance>
struct is_valid_monoid_instance
    : std::conjunction<
          std::is_same<std::invoke_result_t<decltype(Instance::mempty)>, T>,
          std::is_same<std::invoke_result_t<decltype(Instance::mappend_inplace),
                                            T &,
                                            T const &>,
                       void>> {};

template <typename T, typename Instance>
inline constexpr bool is_valid_monoid_instance_v =
    is_valid_monoid_instance<T, Instance>::value;

template <typename Instance, typename Enable = void>
struct is_commutative_monoid : std::false_type {};

template <typename Instance>
inline constexpr bool is_commutative_monoid_v =
    is_commutative_monoid<Instance>::value;

template <typename T, typename Instance = default_monoid_t<T>>
T mempty() {
  static_assert(is_valid_monoid_instance_v<T, Instance>);

  return Instance::mempty();
}

template <typename T, typename Instance = default_monoid_t<T>>
void mappend_inplace(T &accum, T const &val) {
  static_assert(is_valid_monoid_instance_v<T, Instance>);

  Instance::mappend_inplace(accum, val);
}

template <typename T, typename Instance = default_monoid_t<T>>
auto mappend(T const &lhs, T const &rhs)
    -> std::enable_if_t<std::is_copy_constructible_v<T>, T> {
  static_assert(is_valid_monoid_instance_v<T, Instance>);

  T result = lhs;
  mappend_inplace<T, Instance>(result, rhs);
  return result;
}

// concatmap

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_NUMBERS_IS_EQUAL_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_NUMBERS_IS_EQUAL_H

#include <cstddef>
#include <type_traits>

namespace FlexFlow {

template <size_t L, size_t R>
struct is_equal : std::false_type {};

template <size_t N>
struct is_equal<N, N> : std::true_type {};

template <typename T>
inline constexpr bool is_equal_v = is_equal<T>::value;

} // namespace FlexFlow

#endif

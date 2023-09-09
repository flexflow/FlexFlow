#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_NEQ_COMPARABLE_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_NEQ_COMPARABLE_H

#include <type_traits>

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_neq_comparable : std::false_type {};

template <typename T>
struct is_neq_comparable<
    T,
    std::void_t<decltype((bool)(std::declval<T>() != std::declval<T>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_neq_comparable_v = is_neq_comparable<T>::value;


} // namespace FlexFlow

#endif

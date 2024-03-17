#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPES_ARE_ALL_SAME_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPES_ARE_ALL_SAME_H

#include <type_traits>

namespace FlexFlow {

template <typename... Ts>
struct types_are_all_same;

template <typename... Ts>
inline constexpr bool types_are_all_same_v = types_are_all_same<Ts...>::value;

template <typename... Ts>
struct types_are_all_same : std::false_type {};

template <>
struct types_are_all_same<> : std::true_type {};

template <typename T>
struct types_are_all_same<T> : std::true_type {};

template <typename Head, typename Next, typename... Rest>
struct types_are_all_same<Head, Next, Rest...>
    : std::conjunction<std::is_same<Head, Next>,
                       types_are_all_same<Head, Rest...>> {};

} // namespace FlexFlow

#endif

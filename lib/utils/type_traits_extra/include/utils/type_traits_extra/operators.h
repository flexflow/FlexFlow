#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_OPERATORS_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_OPERATORS_H

#include <type_traits>

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_plusable : std::false_type {};

template <typename T>
struct is_plusable<
    T,
    std::void_t<decltype((T)(std::declval<T>() + std::declval<T>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_plusable_v = is_plusable<T>::value;

template <typename T, typename Enable = void>
struct is_minusable : std::false_type {};

template <typename T>
struct is_minusable<
    T,
    std::void_t<decltype((T)(std::declval<T>() - std::declval<T>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_minusable_v = is_minusable<T>::value;

template <typename T, typename Enable = void>
struct is_timesable : std::false_type {};

template <typename T>
struct is_timesable<
    T,
    std::void_t<decltype((T)(std::declval<T>() * std::declval<T>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_timesable_v = is_timesable<T>::value;

} // namespace FlexFlow

#endif

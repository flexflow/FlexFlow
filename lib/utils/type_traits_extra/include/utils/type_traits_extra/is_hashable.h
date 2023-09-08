#ifndef _FLEXFLOW_UTILS_TYPE_TRAITS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_HASHABLE_H
#define _FLEXFLOW_UTILS_TYPE_TRAITS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_HASHABLE_H

#include <functional>
#include <type_traits>

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_hashable : std::false_type {};

template <typename T>
struct is_hashable<
    T,
    std::void_t<decltype((size_t)(std::declval<std::hash<T>>()(std::declval<T>())))>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_hashable_v = is_hashable<T>::value;

#define CHECK_HASHABLE(...)                                                    \
  static_assert(is_hashable<__VA_ARGS__>::value,                               \
                #__VA_ARGS__ " should be hashable (but is not)");


}

#endif

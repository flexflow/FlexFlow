#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_EQUAL_COMPARABLE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_EQUAL_COMPARABLE_H

#include <type_traits>

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_equal_comparable : std::false_type {};

template <typename T>
struct is_equal_comparable<
    T,
    std::enable_if_t<
        std::is_same_v<decltype(std::declval<T>() == std::declval<T>()), bool>>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_equal_comparable_v = is_equal_comparable<T>::value;

} // namespace FlexFlow

#endif

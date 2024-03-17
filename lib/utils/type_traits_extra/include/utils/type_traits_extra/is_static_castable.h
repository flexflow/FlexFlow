#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_IS_STATIC_CASTABLE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_IS_STATIC_CASTABLE_H

#include <type_traits>

namespace FlexFlow {

template <typename From, typename To, typename Enable = void>
struct is_static_castable : std::false_type {};

template <typename From, typename To>
struct is_static_castable<
    From,
    To,
    std::void_t<decltype(static_cast<To>(std::declval<From>()))>>
    : std::true_type {};

template <typename From, typename To>
inline constexpr bool is_static_castable_v =
    is_static_castable<From, To>::value;

} // namespace FlexFlow

#endif

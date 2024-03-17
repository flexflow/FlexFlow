#ifndef _FLEXFLOW_UTILS_TYPE_TRAITS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_STREAMABLE_H
#define _FLEXFLOW_UTILS_TYPE_TRAITS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_STREAMABLE_H

#include <iostream>
#include <type_traits>

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_streamable : std::false_type {};

template <typename T>
struct is_streamable<T, std::void_t<decltype(std::cout << std::declval<T>())>>
    : std::true_type {};

} // namespace FlexFlow

#endif

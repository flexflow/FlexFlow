#ifndef _FLEXFLOW_LIB_UTILS_STRONG_TYPEDEF_INCLUDE_UTILS_STRONG_TYPEDEF_IS_STRONG_TYPEDEF_H
#define _FLEXFLOW_LIB_UTILS_STRONG_TYPEDEF_INCLUDE_UTILS_STRONG_TYPEDEF_IS_STRONG_TYPEDEF_H

#include <type_traits>
#include "underlying_type.h"

namespace FlexFlow {

template <typename T, typename Enable = void> struct is_strong_typedef : std::false_type {};

template <typename T>
struct is_strong_typedef<T, std::void_t<underlying_type_t<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_strong_typedef_v = is_strong_typedef<T>::value;

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_ORDERED_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_ORDERED_H

#include <vector>

namespace FlexFlow {

template <typename T>
struct is_ordered : std::false_type {};
template <typename T>
struct is_ordered<std::vector<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_ordered_v = is_ordered<T>::value;

} // namespace FlexFlow

#endif

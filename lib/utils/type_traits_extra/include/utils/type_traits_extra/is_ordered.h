#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_ORDERED_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_ORDERED_H

namespace FlexFlow {

template <typename T>
struct is_ordered;

template <typename T>
inline constexpr bool is_ordered_v = is_ordered<T>::value;

} // namespace FlexFlow

#endif

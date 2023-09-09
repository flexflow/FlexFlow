#ifndef _FLEXFLOW_LIB_UTILS_SMART_PTRS_INCLUDE_UTILS_SMART_PTRS_IS_CLONABLE_H
#define _FLEXFLOW_LIB_UTILS_SMART_PTRS_INCLUDE_UTILS_SMART_PTRS_IS_CLONABLE_H

#include <type_traits>

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_clonable : std::false_type {};

template <typename T>
struct is_clonable<T, std::void_t<decltype(std::declval<T>().clone())>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_clonable_v = is_clonable<T>::value;

} // namespace FlexFlow

#endif

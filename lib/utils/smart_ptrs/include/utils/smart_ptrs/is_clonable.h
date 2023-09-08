#ifndef _FLEXFLOW_LIB_UTILS_SMART_PTRS_INCLUDE_UTILS_SMART_PTRS_IS_CLONABLE_H
#define _FLEXFLOW_LIB_UTILS_SMART_PTRS_INCLUDE_UTILS_SMART_PTRS_IS_CLONABLE_H

#include <functional>

namespace FlexFlow {

template <typename T, typename Enable>
struct is_clonable : std::false_type {};

template <typename T>
struct is_clonable<T, std::void_t<decltype(std::declval<T>().clone())>>
    : std::true_type {};


} // namespace FlexFlow

#endif

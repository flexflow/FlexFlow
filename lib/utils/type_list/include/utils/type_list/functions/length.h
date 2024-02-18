#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_LENGTH_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_LENGTH_H

#include "utils/type_list/type_list.h"

namespace FlexFlow {

template <typename T> struct type_list_length { };

template <typename T>
inline constexpr int type_list_length_v = type_list_length<T>::value;


} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_HEAD_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_HEAD_H

#include "type_list.h"
#include "utils/backports/type_identity.h" 

namespace FlexFlow {

template <typename T>
struct type_list_get_head { };

template <typename Head, typename... Tail>
struct type_list_get_head<type_list<Head, Tail...>> : type_identity<Head> { };

template <typename T>
using type_list_get_head_t = typename type_list_get_head<T>::type;


} // namespace FlexFlow

#endif

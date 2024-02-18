#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_GET_INIT_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_GET_INIT_H

#include "utils/type_list/type_list.h"
#include "utils/backports/type_identity.h"
#include "prepend.h"

namespace FlexFlow {

template <typename T>
struct type_list_get_init { };

template <typename Head, typename... Tail>
struct type_list_get_init<type_list<Head, Tail...>> : type_list_prepend<Head, type_list_get_init<type_list<Tail...>>> { };

template <typename Last>
struct type_list_get_init<type_list<Last>> : type_identity<type_list<>> { };

template <typename T>
using type_list_get_init_t = typename type_list_get_init<T>::type;

} // namespace FlexFlow

#endif

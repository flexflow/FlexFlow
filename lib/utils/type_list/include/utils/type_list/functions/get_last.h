#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_GET_LAST_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_GET_LAST_H

#include "utils/type_list/type_list.h"
#include "utils/backports/type_identity.h"

namespace FlexFlow {

template <typename T>
struct type_list_get_last { };

template <typename Head, typename... Tail>
struct type_list_get_last<type_list<Head, Tail...>> : type_list_get_last<type_list<Tail...>> { };

template <typename Last>
struct type_list_get_last<type_list<Last>> : type_identity<Last> { };

template <typename T>
using type_list_get_last_t = typename type_list_get_last<T>::type;


} // namespace FlexFlow

#endif

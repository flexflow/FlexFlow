#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_GET_TAIL_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_GET_TAIL_H

#include "utils/backports/type_identity.h"
#include "utils/type_list/type_list.h"

namespace FlexFlow {

template <typename T>
struct type_list_get_tail {};

template <typename Head, typename... Tail>
struct type_list_get_tail<type_list<Head, Tail...>>
    : type_identity<type_list<Tail...>> {};

template <typename T>
using type_list_get_tail_t = typename type_list_get_tail<T>::type;

} // namespace FlexFlow

#endif

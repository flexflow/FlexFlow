#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_INDEXING_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_INDEXING_H

#include "type_list.h"
#include "utils/type_traits_extra/type_functions/indexing.h"
#include "utils/backports/type_identity.h"
#include "prepend.h"

namespace FlexFlow {

template <typename Head, typename... Tail>
struct get_head<type_list<Head, Tail...>> : type_identity<Head> { };

template <typename Head, typename... Tail>
struct get_tail<type_list<Head, Tail...>> : type_identity<type_list<Tail...>> { };

template <typename Head, typename... Tail>
struct get_last<type_list<Head, Tail...>> : get_last<type_list<Tail...>> { };

template <typename Last>
struct get_last<type_list<Last>> : type_identity<Last> { };

template <typename Head, typename... Tail>
struct get_init<type_list<Head, Tail...>> : prepend<Head, get_init<type_list<Tail...>>> { };

template <typename Last>
struct get_init<type_list<Last>> : type_identity<type_list<>> { };

} // namespace FlexFlow

#endif

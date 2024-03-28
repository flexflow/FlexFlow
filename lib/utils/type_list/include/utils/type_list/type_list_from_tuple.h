#ifndef _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_TYPE_LIST_FROM_TUPLE_H
#define _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_TYPE_LIST_FROM_TUPLE_H

#include "utils/type_list/type_list.h"
#include <tuple>
#include "utils/backports/type_identity.h"

namespace FlexFlow {

template <typename T>
struct type_list_from_tuple { };

template <typename... Ts>
struct type_list_from_tuple<std::tuple<Ts...>> : type_identity<type_list<Ts...>> {};

template <typename T>
using type_list_from_tuple_t = typename type_list_from_tuple<T>::type;

} // namespace FlexFlow

#endif

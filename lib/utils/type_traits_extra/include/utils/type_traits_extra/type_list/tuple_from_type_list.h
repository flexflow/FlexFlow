#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_FROM_TYPE_LIST_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_FROM_TYPE_LIST_H

#include "type_list.h"
#include <tuple>
#include "utils/backports/type_identity.h"

namespace FlexFlow {

template <typename T> struct tuple_from_type_list;

template <typename T>
using tuple_from_type_list_t = typename tuple_from_type_list<T>::type;

template <typename... Ts> 
struct tuple_from_type_list<type_list<Ts...>> : type_identity<std::tuple<Ts...>> {};

} // namespace FlexFlow

#endif

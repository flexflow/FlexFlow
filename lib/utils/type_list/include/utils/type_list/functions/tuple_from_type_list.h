#ifndef _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_FUNCTIONS_FROM_TYPE_LIST_H
#define _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_FUNCTIONS_FROM_TYPE_LIST_H

#include "utils/type_list/type_list.h"
#include "utils/backports/type_identity.h"
#include <tuple>

namespace FlexFlow {

template <typename T>
struct tuple_from_type_list;

template <typename... Ts>
struct tuple_from_type_list<type_list<Ts...>> 
  : type_identity<std::tuple<Ts...>> { };

template <typename T>
using tuple_from_type_list_t = typename tuple_from_type_list<T>::type;

} // namespace FlexFlow

#endif

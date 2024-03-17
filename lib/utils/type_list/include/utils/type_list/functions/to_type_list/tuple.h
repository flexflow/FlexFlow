#ifndef _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_TYPE_TUPLE_H
#define _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_TYPE_TUPLE_H

#include "utils/type_list/type_list.h"
#include "utils/type_list/functions/to_type_list.h"
#include <tuple>
#include "utils/backports/type_identity.h"

namespace FlexFlow {

template <typename... Ts>
struct to_type_list<std::tuple<Ts...>> 
  : type_identity<type_list<Ts...>> { };

} // namespace FlexFlow

#endif

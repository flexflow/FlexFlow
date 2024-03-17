#ifndef _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_FUNCTIONS_TO_TYPE_LIST_VARIANT_H
#define _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_FUNCTIONS_TO_TYPE_LIST_VARIANT_H

#include "utils/type_list/type_list.h"
#include "utils/type_list/functions/to_type_list.h"
#include "utils/backports/type_identity.h"
#include <variant>

namespace FlexFlow {

template <typename... Ts>
struct to_type_list<std::variant<Ts...>> 
  : type_identity<type_list<Ts...>> { };

} // namespace FlexFlow

#endif

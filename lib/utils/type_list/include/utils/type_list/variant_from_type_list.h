#ifndef _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_FUNCTIONS_VARIANT_FROM_TYPE_LIST_H
#define _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_FUNCTIONS_VARIANT_FROM_TYPE_LIST_H

#include "utils/backports/type_identity.h"
#include "utils/type_list/type_list.h"
#include <variant>

namespace FlexFlow {

template <typename T>
struct variant_from_type_list;

template <typename... Ts>
struct variant_from_type_list<type_list<Ts...>>
    : type_identity<std::variant<Ts...>> {};

template <typename T>
using variant_from_type_list_t = typename variant_from_type_list<T>::type;

} // namespace FlexFlow

#endif

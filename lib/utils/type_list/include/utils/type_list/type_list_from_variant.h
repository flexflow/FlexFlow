#ifndef _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_TYPE_LIST_FROM_VARIANT_H
#define _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_TYPE_LIST_FROM_VARIANT_H

#include "utils/type_list/type_list.h"
#include <variant>
#include "utils/backports/type_identity.h"

namespace FlexFlow {

template <typename T>
struct type_list_from_variant { };

template <typename... Ts>
struct type_list_from_variant<std::variant<Ts...>> : type_identity<type_list<Ts...>> {};

template <typename T>
using type_list_from_variant_t = typename type_list_from_variant<T>::type;

} // namespace FlexFlow

#endif

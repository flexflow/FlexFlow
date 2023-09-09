#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_VARIANT_FROM_TYPE_LIST_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_VARIANT_FROM_TYPE_LIST_H

#include "type_list.h"
#include <variant>
#include "utils/backports/type_identity.h"

namespace FlexFlow {

template <typename T> struct variant_from_type_list;

template <typename T>
using variant_from_type_list_t = typename variant_from_type_list<T>::type;

template <typename... Ts> 
struct variant_from_type_list<type_list<Ts...>> : type_identity<std::variant<Ts...>> {};

} // namespace FlexFlow

#endif

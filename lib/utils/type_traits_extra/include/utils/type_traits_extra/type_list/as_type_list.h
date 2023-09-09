#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_AS_TYPE_LIST_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_AS_TYPE_LIST_H

#include <utility>
#include <tuple>
#include <variant>
#include "utils/backports/type_identity.h"
#include "type_list.h"

namespace FlexFlow {

template <typename T, typename Enable = void>
struct as_type_list;

template <typename T>
using as_type_list_t = typename as_type_list<T>::type;

template <typename... Ts>
struct as_type_list<std::tuple<Ts...>> : type_identity<type_list<Ts...>> {};

template <typename... Ts>
struct as_type_list<std::variant<Ts...>> : type_identity<type_list<Ts...>> {};

template <typename T1, typename T2>
struct as_type_list<std::pair<T1, T2>> : type_identity<type_list<T1, T2>> {};

} // namespace FlexFlow

#endif

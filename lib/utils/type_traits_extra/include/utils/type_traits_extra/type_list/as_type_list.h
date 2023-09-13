#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_AS_TYPE_LIST_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_AS_TYPE_LIST_H

#include <utility>
#include <tuple>
#include <variant>
#include "utils/backports/type_identity.h"
#include "type_list.h"

namespace FlexFlow {

template <typename T, typename Enable = void>
struct as_type_list_impl { };

template <typename... Ts>
struct as_type_list_impl<std::tuple<Ts...>> : type_identity<type_list<Ts...>> {};

template <typename... Ts>
struct as_type_list_impl<std::variant<Ts...>> : type_identity<type_list<Ts...>> {};

template <typename T1, typename T2>
struct as_type_list_impl<std::pair<T1, T2>> : type_identity<type_list<T1, T2>> {};

template <typename T>
struct as_type_list : as_type_list_impl<std::decay_t<T>> { };

template <typename T>
using as_type_list_t = typename as_type_list<T>::type;


} // namespace FlexFlow

#endif

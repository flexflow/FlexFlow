#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_CONCAT_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_CONCAT_H

#include "utils/backports/type_identity.h"
#include "utils/type_list/type_list.h"
#include <type_traits>

namespace FlexFlow {

template <typename...>
struct type_list_concat_impl {};

template <typename... Ts1, typename... Ts2, typename... Rest>
struct type_list_concat_impl<type_list<Ts1...>, type_list<Ts2...>, Rest...>
    : type_list_concat_impl<type_list<Ts1..., Ts2...>, Rest...> {};

template <typename... Ts1>
struct type_list_concat_impl<type_list<Ts1...>>
    : type_identity<type_list<Ts1...>> {};

template <typename... Ts>
struct type_list_concat : type_list_concat_impl<std::decay_t<Ts>...> {};

template <typename... Ts>
using type_list_concat_t = typename type_list_concat<Ts...>::type;

} // namespace FlexFlow

#endif

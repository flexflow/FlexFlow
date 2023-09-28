#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_CONTAINS_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_CONTAINS_H

#include "type_list.h"
#include <type_traits>
#include "indexing.h"

namespace FlexFlow {

template <typename T, typename TypeList>
struct type_list_contains_impl {};

template <typename T>
struct type_list_contains_impl<T, type_list<>> : std::false_type {};

template <typename T, typename Head, typename... Tail>
struct type_list_contains_impl<T, type_list<Head, Tail...>> : std::disjunction<std::is_same<T, Head>, type_list_contains_impl<T, type_list<Tail...>>> { };

template <typename T, typename TypeList>
struct type_list_contains : type_list_contains_impl<T, std::decay_t<TypeList>> {};

template <typename T, typename TypeList>
inline constexpr bool type_list_contains_v = type_list_contains<T, TypeList>::value;

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_IS_LIST_INITIALIZABLE_FROM_TUPLE_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_IS_LIST_INITIALIZABLE_FROM_TUPLE_H

#include "utils/type_traits_extra/is_list_initializable.h"
#include "utils/type_list/type_list.h"

namespace FlexFlow {

template <typename T, typename ArgList>
struct is_list_initializable_from_type_list;

template <typename T, typename... Args>
struct is_list_initializable_from_type_list<T, type_list<Args...>>
    : is_list_initializable<T, Args...> {};

template <typename T, typename ArgList>
inline constexpr bool is_list_initializable_from_type_list_v =
    is_list_initializable_from_type_list<T, ArgList>::value;

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_IS_TYPE_LIST_H
#define _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_IS_TYPE_LIST_H

#include "type_list.h"
#include <type_traits>

namespace FlexFlow {

template <typename T> struct is_type_list : std::false_type { };

template <typename... Ts> struct is_type_list<type_list<Ts...>> : std::true_type { };

template <typename T>
inline constexpr bool is_type_list_v = is_type_list<T>::value;

} // namespace FlexFlow

#endif

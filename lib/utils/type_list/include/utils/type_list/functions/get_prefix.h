#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_GET_PREFIX_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_GET_PREFIX_H

#include "type_list.h"
#include "length.h"
#include <type_traits>
#include "get_init.h"
#include "utils/type_traits_extra/metafunction/repeat.h"
#include "normalize_index.h"

namespace FlexFlow {

template <typename T, int End, typename Enable = void> struct type_list_get_prefix { };

template <int End, typename... Ts> 
struct type_list_get_prefix<type_list<Ts...>, End> 
  : metafunction_repeat<type_list_get_init, normalize_index_v<type_list_length_v<type_list<Ts...>>, (type_list_length_v<type_list<Ts...>> - End)>, type_list<Ts...>> { };

template <typename T, int End> 
using type_list_get_prefix_t = typename type_list_get_prefix<T, End>::type;

} // namespace FlexFlow

#endif

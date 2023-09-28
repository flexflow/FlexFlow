#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_GET_SUFFIX_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_GET_SUFFIX_H

#include "type_list.h"
#include "get_tail.h"
#include "utils/type_traits_extra/metafunction/repeat.h" 
#include "length.h"
#include "normalize_index.h"

namespace FlexFlow {

template <typename T, int Start, typename Enable = void> 
struct type_list_get_suffix { };

template <int Start, typename... Ts>
struct type_list_get_suffix<type_list<Ts...>, Start> 
  : metafunction_repeat<type_list_get_tail, normalize_index_v<type_list<Ts...>, Start>, type_list<Ts...>> { };

template <typename T, int Start>
using type_list_get_suffix_t = typename type_list_get_suffix<T, Start>::type;

} // namespace FlexFlow

#endif

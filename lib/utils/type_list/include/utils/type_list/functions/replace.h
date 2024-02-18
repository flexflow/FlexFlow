#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_REPLACE_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_REPLACE_H

#include "utils/type_list/type/variant.h"
#include "indexing.h"
#include "concat.h"

namespace FlexFlow {

template <int Idx, typename NewValue, typename TypeList> struct type_list_replace_element;

template <int Idx, typename NewValue, typename... Ts>
struct type_list_replace_element<Idx, NewValue, type_list<Ts...>> :
  variant_from_type_list<
    type_list_concat_t<
      type_list_get_prefix_t<type_list<Ts...>, Idx>,
      type_list<NewValue>,
      type_list_get_suffix_t<type_list<Ts...>, Idx+1>
    >
  > { };

template <int Idx, typename NewValue, typename TypeList>
using type_list_replace_element_t = typename type_list_replace_element<Idx, NewValue, TypeList>::type;

} // namespace FlexFlow

#endif

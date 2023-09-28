#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_SLICE_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_SLICE_H

#include "utils/ff_exceptions/type_function_not_implemented.h"
#include "length.h"
#include "get_suffix.h"
#include "get_prefix.h"

namespace FlexFlow {

template <typename T, int Start, int End> struct type_list_slice_impl { };

template <int Start, int End, typename... Ts>
struct type_list_slice_impl<type_list<Ts...>, Start, End> : 
  type_list_get_suffix<type_list_get_prefix<type_list<Ts...>, End>, Start> { };

template <typename T, int Start = 0, int End = type_list_length_v<T>> struct type_list_slice { };

template <int Start, int End, typename... Ts>
struct type_list_slice<type_list<Ts...>, Start, End>
  : type_list_slice_impl<
      type_list<Ts...>,
      normalize_index_v<type_list_length_v<type_list<Ts...>>, Start>,
      normalize_index_v<type_list_length_v<type_list<Ts...>>, End>> { };

template <typename T, int Start = 0, int End = type_list_length_v<T>>
using type_list_slice_t = typename type_list_slice<T, Start, End>::type;

} // namespace FlexFlow

#endif

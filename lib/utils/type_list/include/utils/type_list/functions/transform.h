#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_APPLY_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_APPLY_H

#include "type_list.h"

namespace FlexFlow {

template <template <typename...> typename Metafunction, typename ArgList> struct type_list_apply { };

template <template <typename...> typename Metafunction, typename... Args> 
struct type_list_apply<Metafunction, type_list<Args...>>
  : Metafunction<Args...> { };

template <template <typename...> typename Metafunction, typename ArgList> 
using type_list_apply_t = typename type_list_apply<Metafunction, ArgList>::type;

template <template <typename...> typename Metafunction, typename ArgList>
inline constexpr auto type_list_apply_v = type_list_apply<Metafunction, ArgList>::value;

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_PREPEND_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_PREPEND_H

#include "utils/type_list/type_list.h"
#include "utils/backports/type_identity.h" 

namespace FlexFlow {

template <typename ToPrepend, typename List>
struct type_list_prepend { };

template <typename ToPrepend, typename... Args>
struct type_list_prepend<ToPrepend, type_list<Args...>> : type_identity<type_list<ToPrepend, Args...>> { };

template <typename ToPrepend, typename T>
using type_list_prepend_t = typename type_list_prepend<ToPrepend, T>::type;

} // namespace FlexFlow

#endif

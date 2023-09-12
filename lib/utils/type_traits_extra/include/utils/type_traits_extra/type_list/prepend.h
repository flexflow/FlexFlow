#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_PREPEND_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_PREPEND_H

#include "utils/type_traits_extra/type_functions/prepend.h"
#include "type_list.h"
#include "utils/backports/type_identity.h" 

namespace FlexFlow {

template <typename ToPrepend, typename... Args>
struct prepend<ToPrepend, type_list<Args...>> : type_identity<type_list<ToPrepend, Args...>> { };

} // namespace FlexFlow

#endif

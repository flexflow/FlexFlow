#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_AS_TYPE_LIST_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_AS_TYPE_LIST_H

#include "utils/type_traits_extra/type_list/as_type_list.h"
#include "utils/visitable/type/functions/visit_as_tuple.h"
#include "utils/visitable/type/traits/is_visitable.h"

namespace FlexFlow {

template <typename T>
struct as_type_list<T, std::enable_if_t<is_visitable_v<T>>>
    : as_type_list<visit_as_tuple_t<T>> {};

} // namespace FlexFlow

#endif

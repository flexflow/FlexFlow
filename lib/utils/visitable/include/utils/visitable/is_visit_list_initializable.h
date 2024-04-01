#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_TRAITS_IS_VISIT_LIST_INITIALIZABLE_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_TRAITS_IS_VISIT_LIST_INITIALIZABLE_H

#include "is_list_initializable_from_type_list.h"
#include "is_visitable.h"
#include "utils/visitable/type_list_from_visitable.h"

namespace FlexFlow {

template <typename T>
concept visit_list_initializable = is_visitable_v<T> && is_list_initializable_from_type_list_v<T, type_list_from_visitable_t<T>>;

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_TYPE_TRAITS_IS_ONLY_VISIT_LIST_INITIALIZABLE_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_TYPE_TRAITS_IS_ONLY_VISIT_LIST_INITIALIZABLE_H

#include "utils/visitable/is_visit_list_initializable.h"
#include "utils/visitable/type_list_from_visitable.h"
#include "utils/type_list/get_init.h"

namespace FlexFlow {

template <typename T>
concept is_only_visit_list_initializable = visit_list_initializable<T>
  && !is_list_initializable_from_type_list_v<T, type_list_get_init_t<type_list_from_visitable_t<T>>>;

} // namespace FlexFlow

#endif

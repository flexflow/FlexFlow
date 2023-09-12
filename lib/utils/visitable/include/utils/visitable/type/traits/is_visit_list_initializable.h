#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_TRAITS_IS_VISIT_LIST_INITIALIZABLE_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_TRAITS_IS_VISIT_LIST_INITIALIZABLE_H

#include "is_visitable.h"
#include "is_list_initializable_from_type_list.h"
#include "utils/visitable/type/functions/as_type_list.h"

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_visit_list_initializable
    : std::conjunction<is_visitable<T>,
                  is_list_initializable_from_type_list<T, as_type_list_t<T>>> {};


} // namespace FlexFlow

#endif

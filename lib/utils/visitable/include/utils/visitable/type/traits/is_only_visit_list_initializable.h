#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_TYPE_TRAITS_IS_ONLY_VISIT_LIST_INITIALIZABLE_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_TYPE_TRAITS_IS_ONLY_VISIT_LIST_INITIALIZABLE_H

#include "is_visit_list_initializable.h"
#include "utils/visitable/type/functions/as_type_list.h"
#include "utils/type_traits_extra/type_list/indexing.h"

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_only_visit_list_initializable
    : std::conjunction<is_visit_list_initializable<T>,
                  std::negation<is_list_initializable_from_type_list<
                      T,
                      get_init_t<as_type_list<T>>>>> {};


} // namespace FlexFlow

#endif

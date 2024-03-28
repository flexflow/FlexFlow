#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_TYPE_TRAITS_IS_WELL_BEHAVED_VISITABLE_TYPE_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_TYPE_TRAITS_IS_WELL_BEHAVED_VISITABLE_TYPE_H

#include "is_visit_list_initializable.h"
#include "is_visitable.h"
#include "utils/type_traits_extra/biconditional.h"
#include "utils/type_traits_extra/is_well_behaved_value_type.h"
#include "utils/type_traits_extra/numbers/is_equal.h"
#include "utils/visitable/field_count.h"

namespace FlexFlow {

template <typename T>
struct is_well_behaved_visitable_type
    : std::conjunction<is_visitable<T>,
                       is_well_behaved_value_type<T>,
                       is_visit_list_initializable<T>,
                       biconditional<std::bool_constant<field_count_v<T> == 0>,
                                     std::is_default_constructible<T>>> {};

} // namespace FlexFlow

#endif

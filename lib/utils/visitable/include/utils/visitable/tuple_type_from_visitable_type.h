#ifndef _FLEXFLOW_UTILS_VISITABLE_INCLUDE_VISITABLE_AS_TUPLE_H
#define _FLEXFLOW_UTILS_VISITABLE_INCLUDE_VISITABLE_AS_TUPLE_H

#include "utils/visitable/field_count.h"
#include "utils/type_traits_extra/type_functions/prepend.h"
#include "utils/visitable/required.h"
#include "visit_struct/visit_struct.hpp"
#include <type_traits>
#include "utils/type_list/tuple_from_type_list.h"
#include "utils/visitable/type_list_from_visitable.h"

namespace FlexFlow {

template <typename T>
using tuple_type_from_visitable_type = tuple_from_type_list<type_list_from_visitable_t<T>>;

template <typename T>
using tuple_type_from_visitable_type_t = typename tuple_type_from_visitable_type<T>::type;

} // namespace FlexFlow

#endif

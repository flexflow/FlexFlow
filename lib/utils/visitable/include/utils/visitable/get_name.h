#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_TYPE_FUNCTIONS_GET_NAME_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_TYPE_FUNCTIONS_GET_NAME_H

#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

template <typename T>
char const *visitable_get_name() {
  return ::visit_struct::get_name<T>();
}

} // namespace FlexFlow

#endif

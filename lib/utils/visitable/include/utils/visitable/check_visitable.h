#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_CHECK_VISITABLE_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_CHECK_VISITABLE_H

#include "utils/visitable/is_visitable.h"
#include "utils/type_traits_extra/debug_print_type.h"

namespace FlexFlow {

#define CHECK_VISITABLE(...)                                                   \
  static_assert(::FlexFlow::is_visitable_v<__VA_ARGS__>,                       \
                #__VA_ARGS__ " should be visitable (but is not)");

} // namespace FlexFlow

#endif

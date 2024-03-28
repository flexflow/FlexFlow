#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATORS_LESSTHAN_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATORS_LESSTHAN_H

#include "utils/type_traits_extra/is_lt_comparable.h"
#include "utils/visitable/visitable_elements_satisfy.h"
#include "utils/visitable/is_visitable.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

template <typename T>
auto operator<(T const &lhs, T const &rhs) -> std::enable_if_t<
    std::conjunction_v<is_visitable<T>, elements_satisfy<is_lt_comparable, T>>,
    bool> {
  return visit_lt(lhs, rhs);
}

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATOR_NOTEQUALS_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATOR_NOTEQUALS_H

#include <type_traits>
#include "utils/visitable/is_visitable.h"
#include "utils/type_traits_extra/is_neq_comparable.h"
#include "utils/visitable/visitable_elements_satisfy.h"

namespace FlexFlow {

template <typename T>
auto operator!=(T const &lhs, T const &rhs) -> std::enable_if_t<
    std::conjunction_v<is_visitable<T>, visitable_elements_satisfy<is_neq_comparable, T>>,
    bool> {
  return visit_neq(lhs, rhs);
}

} // namespace FlexFlow

#endif

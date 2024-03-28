#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATORS_NEQ_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATORS_NEQ_H

#include "utils/type_traits_extra/is_neq_comparable.h"
#include "utils/visitable/visitable_elements_satisfy.h"
#include "utils/visitable/is_visitable.h"
#include "utils/visitable/check_visitable.h"

namespace FlexFlow {

struct neq_visitor {
  bool result = false;

  template <typename T>
  void operator()(char const *, T const &t1, T const &t2) {
    result |= (t1 != t2);
  }
};

template <typename T>
bool visit_neq(T const &lhs, T const &rhs) {
  CHECK_VISITABLE(T);
  static_assert(visitable_elements_satisfy<is_neq_comparable, T>::value,
                "Values must be comparable via operator!=");

  neq_visitor vis;
  visit_struct::for_each(lhs, rhs, vis);
  return vis.result;
}

} // namespace FlexFlow

#endif

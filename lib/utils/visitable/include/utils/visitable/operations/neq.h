#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATORS_NEQ_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATORS_NEQ_H

#include "utils/visitable/type/traits/is_visitable.h"
#include "utils/type_traits_extra/is_neq_comparable.h"
#include "utils/visitable/type/traits/elements_satisfy.h"

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
  static_assert(is_visitable<T>::value, "Type must be visitable");
  static_assert(elements_satisfy<is_neq_comparable, T>::value,
                "Values must be comparable via operator!=");

  neq_visitor vis;
  visit_struct::for_each(lhs, rhs, vis);
  return vis.result;
}


template <typename T>
auto operator!=(T const &lhs, T const &rhs) -> std::enable_if_t<
    std::conjunction_v<is_visitable<T>, elements_satisfy<is_neq_comparable, T>>,
    bool> {
  return visit_neq(lhs, rhs);
}


} // namespace FlexFlow

#endif

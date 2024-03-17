#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATORS_LESSTHAN_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATORS_LESSTHAN_H

#include "utils/type_traits_extra/is_lt_comparable.h"
#include "utils/visitable/type/traits/elements_satisfy.h"
#include "utils/visitable/type/traits/is_visitable.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct lt_visitor {
  bool result = true;

  template <typename T>
  void operator()(char const *, T const &t1, T const &t2) {
    result = result && (t1 < t2);
  }
};

template <typename T>
bool visit_lt(T const &t1, T const &t2) {
  static_assert(is_visitable<T>::value, "Type must be visitable");
  static_assert(elements_satisfy<is_lt_comparable, T>::value,
                "Values must be comparable via operator<");

  lt_visitor vis;
  visit_struct::for_each(t1, t2, vis);
  return vis.result;
}

template <typename T>
auto operator<(T const &lhs, T const &rhs) -> std::enable_if_t<
    std::conjunction_v<is_visitable<T>, elements_satisfy<is_lt_comparable, T>>,
    bool> {
  return visit_lt(lhs, rhs);
}

} // namespace FlexFlow

#endif

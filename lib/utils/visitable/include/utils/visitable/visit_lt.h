#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_VISIT_LT_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_VISIT_LT_H

#include "utils/visitable/check_visitable.h"
#include "utils/visitable/visitable_elements_satisfy.h"
#include "utils/type_traits_extra/is_lt_comparable.h"

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
  CHECK_VISITABLE(T);
  static_assert(visitable_elements_satisfy<is_lt_comparable, T>::value,
                "Values must be comparable via operator<");

  lt_visitor vis;
  visit_struct::for_each(t1, t2, vis);
  return vis.result;
}

} // namespace FlexFlow

#endif

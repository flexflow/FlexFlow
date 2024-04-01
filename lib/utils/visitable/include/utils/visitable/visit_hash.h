#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATIONS_HASH_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATIONS_HASH_H

#include "utils/type_traits_extra/is_hashable.h"
#include "utils/visitable/visitable_elements_satisfy.h"
#include "utils/visitable/is_visitable.h"
#include "visit_struct/visit_struct.hpp"
#include "utils/visitable/check_visitable.h"
#include <functional>

namespace FlexFlow {

struct hash_visitor {
  std::size_t result = 0;

  template <typename T>
  void operator()(char const *, T const &t1) {
    hash_combine(result, t1);
  }
};

template <typename T>
std::size_t visit_hash(T const &t) {
  CHECK_VISITABLE(T);
  static_assert(visitable_elements_satisfy_v<is_hashable, T>,
                "Values must be hashable");

  hash_visitor vis;
  visit_struct::for_each(t, vis);
  return vis.result;
}

} // namespace FlexFlow

#endif

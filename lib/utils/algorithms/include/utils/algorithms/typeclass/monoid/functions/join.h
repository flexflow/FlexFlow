#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_FUNCTIONS_JOIN_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_FUNCTIONS_JOIN_H

#include "utils/algorithms/type/monoid/monoid.h"

namespace FlexFlow {

template <typename C, typename Instance = default_monoid_t<element_type_t<C>>>
auto mjoin(C const &c, element_type_t<C> const &delimiter)
    -> std::enable_if_t<is_ordered_v<C>, element_type_t<C>> {
  using T = element_type_t<C>;
  static_assert(is_valid_monoid_instance_v<T, Instance>);

  T result = mempty<T, Instance>();
  bool first = true;
  for (T const &t : c) {
    if (!first) {
      mappend_inplace<T, Instance>(result, delimiter);
    }
    mappend_inplace<T, Instance>(result, t);
    first = false;
  }

  return result;
}

} // namespace FlexFlow

#endif

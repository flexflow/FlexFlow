#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_FUNCTIONS_CONCATMAP_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_FUNCTIONS_CONCATMAP_H

#include "utils/algorithms/type/monoid/monoid.h"

namespace FlexFlow {

template <typename F, 
          typename C, 
          typename Instance = monoid_t<std::invoke_result_t<F, element_type_t<C>>>>
auto concatMap(C const &c, F const &f) 
  -> std::invoke_result_t<F, element_type_t<C>>
{
  using In = element_type_t<C>;
  using Out = std::invoke_result_t<F, In>;

  Out result = Instance::mempty();
  for (In const &in : c) {
    Instance::mappend_inplace(result, f(in));
  }
  return result;
}

} // namespace FlexFlow

#endif

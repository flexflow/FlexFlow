#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_FUNCTIONS_MPRODUCT_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_FUNCTIONS_MPRODUCT_H

#include "utils/algorithms/type/monoid/functions/mconcat.h"
#include "utils/algorithms/type/monoid/instances/arithmetic.h"
#include "utils/algorithms/type/monoid/monoid.h"

namespace FlexFlow {

template <typename C>
auto mproduct(C const &c)
    -> std::enable_if_t<is_ordered_v<C>, element_type_t<C>> {
  using T = element_type_t<C>;
  return mconcat<T, product_monoid_t<T>>(c);
}

template <typename C, typename F>
auto mproductWhere(C const &c, F const &f) -> std::enable_if_t<
    is_ordered_v<C> &&
        is_static_castable_v<std::invoke_result_t<F, element_type_t<C>>, bool>,
    element_type_t<C>> {
  using T = element_type_t<C>;
  using T = element_type_t<C>;
  return mconcatWhere<T, product_monoid_t<T>>(c, f);
}

} // namespace FlexFlow

#endif

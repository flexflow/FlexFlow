#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_FUNCTIONS_MCONCAT_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_MONOID_FUNCTIONS_MCONCAT_H

#include "utils/algorithms/type/monoid/monoid.h"
#include "utils/type_traits_extra/is_static_castable.h"

namespace FlexFlow {

template <typename C, typename F, typename Instance = default_monoid_t<element_type_t<C>>>
auto mconcatWhere(C const &c)
  -> std::enable_if_t<(is_ordered_v<C> || is_commutative_monoid_v<Instance>) && is_static_castable_v<std::invoke_result_t<F, element_type_t<C> const &>, bool>, element_type_t<C>>
{
  using T = element_type_t<C>;
  T result = mempty<T, Instance>();
  for (T const &t : c) {
    if (static_cast<bool>(f(t))) {
      mappend_inplace<T, Instance>(result, t);
    }
  }
  return result;
}

template <typename C, typename Instance = default_monoid_t<element_type_t<C>>>
auto mconcat(C const &c) 
  -> std::enable_if_t<is_ordered_v<C>, element_type_t<C>>
{
  return mconcatWhere<Instance>(c, [](element_type_t<C> const &) { return true; });
}


} // namespace FlexFlow

#endif

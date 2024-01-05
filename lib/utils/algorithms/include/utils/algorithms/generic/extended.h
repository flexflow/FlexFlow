#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_GENERIC_EXTENDED_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_GENERIC_EXTENDED_H

#include "utils/algorithms/typeclass/functor/functor.h"
#include "utils/algorithms/typeclass/monoid/monoid.h"

namespace FlexFlow {

template <typename T1, typename T2, typename Instance = default_monoid_t<T1>, typename FunctorInstance = default_functor_t<T2>>
void extend(T1 &t1, T2 const &t2) {
  static_assert(std::is_constructible_v<T1, element_type_t<FunctorInstance>>);
  
  T1 t2_converted = mconcat(fmap<T2, FunctorInstance>(t2, [](element_type_t<FunctorInstance> const &a) { return T1{a}; }));
  mappend_inplace<T1, Instance>(t1, t2_converted);
}

template <typename T1, typename T2>
T1 extended(T1 const &t1, T2 const &t2) {
  T1 result = t1;
  extend(result, t2);
  return result;
}

} // namespace FlexFlow

#endif

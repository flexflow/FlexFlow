#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_OPERATORS_EQ_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_OPERATORS_EQ_H

#include "utils/visitable/type/traits/is_visitable.h"
#include "utils/type_traits_extra/is_equal_comparable.h"
#include "utils/visitable/type/traits/elements_satisfy.h"

namespace FlexFlow {

template <typename T>
bool visit_eq(T const &lhs, T const &rhs) {
  static_assert(is_visitable_v<T>, "Type must be visitable");
  /* static_assert(elements_satisfy<is_equal_comparable, T>::value, */
  /*               "Values must be comparable via operator=="); */

  bool result = true;
  visit_struct::for_each(lhs, rhs, [&](char const *, auto const &t1, auto const &t2) {
    result &= (t1 == t2);
  });
  return result;
}


template <typename T>
auto operator==(T const &lhs, T const &rhs)
    -> std::enable_if_t<is_visitable_v<T>, bool> {
                               /* elements_satisfy<is_equal_comparable, T>>, */
  return visit_eq(lhs, rhs);
}

/* template <typename T, typename TT> */
/* auto operator==(T const &lhs, TT const &rhs) -> std::enable_if_t< */
/*     std::conjunction_v<is_visitable<T>, std::is_convertible<TT, T>>, */
/*     bool> { */
/*   return lhs == static_cast<T>(rhs); */
/* } */

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_FUNCTOR_INSTANCES_LIST_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_FUNCTOR_INSTANCES_LIST_H

#include "utils/algorithms/typeclass/functor/functor.h"
#include "utils/backports/type_identity.h"
#include <list>

namespace FlexFlow {

template <typename T>
struct list_functor {
  using A = T;

  template <typename X>
  using F = std::list<X>;

  template <typename Func>
  static F<std::invoke_result_t<Func, A>> fmap(F<A> const &v, Func const &f) {
    using B = std::invoke_result_t<Func, A>;
    F<B> result;
    std::transform(v.cbegin(), v.cend(), std::back_inserter(result), f);
    return result;
  }

  template <typename Func,
            typename = std::enable_if_t<std::is_invocable_r_v<A, Func, A>>>
  static void fmap_inplace(F<A> &v, Func const &f) {
    std::transform(v.cbegin(), v.cend(), v.begin(), f);
  }
};

template <typename T>
struct default_functor<std::list<T>> : type_identity<list_functor<T>> {};

} // namespace FlexFlow

#endif

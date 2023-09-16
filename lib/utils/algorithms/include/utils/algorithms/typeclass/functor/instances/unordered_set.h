#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_FUNCTOR_INSTANCES_UNORDERED_SET_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_FUNCTOR_INSTANCES_UNORDERED_SET_H

#include <unordered_set>
#include "utils/algorithms/typeclass/functor/functor.h"
#include "utils/backports/type_identity.h"
#include "utils/type_traits_extra/is_hashable.h"

namespace FlexFlow {

template <typename T>
struct unordered_set_functor {
  static_assert (is_hashable_v<T>);

  using A = T;

  template <typename X>
  using F = std::unordered_set<X>;

  template <typename Func, typename = std::enable_if_t<is_hashable_v<std::invoke_result_t<Func, A>>>>
  static F<std::invoke_result_t<Func, A>> fmap(F<A> const &v, Func const &f) {
    using B = std::invoke_result_t<Func, A>;
    F<B> result;
    for (A const &a : v) {
      result.insert(f(a));
    }
    return result;
  }

  template <typename In, typename F>
  static std::vector<std::invoke_result_t<F, In>> fmap(std::vector<In> const &v, F const &f) {
    using Out = std::invoke_result_t<F, In>;
    std::unordered_set<Out> result;
    for (In const &i : v) {
      result.insert(f(i));
    }
    return result;
  }
};

template <typename T>
struct default_functor<std::unordered_set<T>, std::enable_if_t<is_hashable_v<T>>> : type_identity<unordered_set_functor<T>> { };

} // namespace FlexFlow

#endif

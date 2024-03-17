#ifndef _FLEXFLOW_LIB_UTILS_STACK_CONTAINERS_INCLUDE_UTILS_STACK_CONTAINERS_STACK_VECTOR_INSTANCES_FUNCTOR_H
#define _FLEXFLOW_LIB_UTILS_STACK_CONTAINERS_INCLUDE_UTILS_STACK_CONTAINERS_STACK_VECTOR_INSTANCES_FUNCTOR_H

#include "utils/algorithms/type/functor/functor.h"
#include "utils/backports/type_identity.h"
#include "utils/stack_containers/stack_vector/stack_vector.h"
#include <cstddef>

namespace FlexFlow {

template <typename T, std::size_t MAXSIZE>
struct stack_vector_functor {
  using A = T;

  template <typename X>
  using F = stack_vector<X, MAXSIZE>;

  template <typename Func>
  static F<std::invoke_result_t<Func, A>> fmap(F<A> const &v, Func const &f) {
    using B = std::invoke_result_t<Func, A>;
    stack_vector<B, MAXSIZ> result;
    for (A const &a : v) {
      result.push_back(f(a));
    }
    return result;
  }

  template <typename Func,
            typename = std::enable_if_t<std::is_invocable_r_v<A, Func, A>>>
  static void fmap_inplace(F<A> &fa, Func const &f) {
    for (auto it = fa.begin(); it < fa.end(); fa++) {
      *it = f(*it);
    }
  }
};

template <typename T, std::size_t MAXSIZE>
struct default_functor<stack_vector<T, MAXSIZE>>
    : type_identity<stack_vector_functor<T, MAXSIZE>> {};

} // namespace FlexFlow

#endif

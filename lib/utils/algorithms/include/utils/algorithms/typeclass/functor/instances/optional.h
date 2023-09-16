#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_FUNCTOR_INSTANCES_OPTIONAL_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_FUNCTOR_INSTANCES_OPTIONAL_H

#include <optional>
#include "utils/algorithms/typeclass/functor/functor.h"
#include "utils/backports/type_identity.h"

namespace FlexFlow {

template <typename T>
struct optional_functor {
  using A = T;

  template <typename X>
  using F = std::optional<X>;

  template <typename Func>
  static F<std::invoke_result_t<Func, A>> fmap(F<A> const &v, Func const &f) {
    using B = std::invoke_result_t<Func, A>;
    if (v.has_value()) {
      return f(v.value());
    } else {
      return std::nullopt; 
    }
  }
};

template <typename T>
struct default_functor<T> : type_identity<optional_functor<T>> {};

} // namespace FlexFlow

#endif

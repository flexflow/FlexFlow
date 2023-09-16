#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_FUNCTOR_INSTANCES_VECTOR_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_FUNCTOR_INSTANCES_VECTOR_H

#include <vector>
#include "utils/algorithms/type/functor/functor.h"
#include "utils/backports/type_identity.h" 

namespace FlexFlow {

template <typename T>
struct vector_functor {
  using A = T;

  template <typename X>
  using F = std::vector<X>;

  template <typename Func>
  static F<std::invoke_result_t<Func, A>> fmap(std::vector<A> const &v, Func const &f) {
    using B = std::invoke_result_t<Func, A>;
    std::vector<B> result;
    std::transform(v.cbegin(), v.cend(), std::back_inserter(result), f);
    return result;
  }

  template <typename Func, typename = std::enable_if_t<std::is_invocable_r_v<A, Func, A>>>
  static void fmap_inplace(std::vector<A> &v, Func const &f) {
    std::transform(v.cbegin(), v.cend(), v.begin(), f);
  }
};

template <typename T>
struct default_functor<std::vector<T>> : type_identity<vector_functor<T>> {};

template <typename A>
inline constexpr bool vector_functor_is_correct_for_input_v = is_valid_functor_instance<default_functor_t<std::vector<A>>>::value;

static_assert(vector_functor_is_correct_for_input_v<opaque_input_type_t>);
/* static_assert(vector_functor_is_correct_for_input_v<int>); */

} // namespace FlexFlow

#endif

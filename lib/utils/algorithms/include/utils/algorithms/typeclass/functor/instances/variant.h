#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPECLASS_FUNCTOR_INSTANCES_VARIANT_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPECLASS_FUNCTOR_INSTANCES_VARIANT_H

#include "utils/algorithms/typeclass/functor/functor.h"
#include "utils/backports/type_identity.h"
#include "utils/type_traits_extra/type_list/replace.h"
#include "utils/variant_extra/variant.h"
#include <variant>

namespace FlexFlow {

template <int Idx, typename... Ts>
struct variant_idx_functor {
  using A = type_list_get_element<Idx, type_list<Ts...>>;

  template <typename X>
  using F = type_list_replace_element_t<Idx, X, A>;

  template <typename Func>
  static F<std::invoke_result_t<Func, A>> fmap(F<A> const &v, Func const &f) {
    using B = std::invoke_result_t<Func, A>;
    if (std::holds_alternative<A>(v)) {
      return f(std::get<Idx>(v));
    } else {
      return cast<F<B>>(v).value();
    }
  }
};

/* template <typename ToReplace..., typename... Ts> */
/* struct variant_functor { */
/*   using A = variant_from_type_list<type_list_subtract_t<type_list<Ts...>,
 * type_list<ToReplace...>>>; */

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_UTILS_VARIANT_H
#define _FLEXFLOW_UTILS_VARIANT_H 

#include "mpark/variant.hpp"

namespace FlexFlow {

/* using mp = mpark; */

template <class ...Args>
struct variant_join_helper;

/* template <typename ...Ts> */
/* using variant = ::mpark::variant<Ts...>; */

using namespace mpark;

/* template <typename T> */
/* using optional = ::tl::optional<T>; */



/* template <typename T> */
/* using get = ::mpark::get; */

/* template <typename T> */
/* using holds_alternative = ::mpark::holds_alternative<T>; */

/* template <typename T> */
/* using visit = mpark::visit<T>; */

template <class ...Args1, class ...Args2>
struct variant_join_helper<mpark::variant<Args1...>, mpark::variant<Args2...>> {
    using type = mpark::variant<Args1..., Args2...>;
};

template <class Variant1, class Variant2>
using variant_join = typename variant_join_helper<Variant1, Variant2>::type;

template <typename Out>
struct VariantCastFunctor {
  template <typename T>
  Out operator()(T const &t) const {
    return Out(t);
  }
};

template <typename VariantOut, typename VariantIn>
VariantOut variant_cast(VariantIn const &v) {
  return visit(VariantCastFunctor<VariantOut>{}, v);
}

}

#endif 

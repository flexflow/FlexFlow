#ifndef _FLEXFLOW_UTILS_VARIANT_H
#define _FLEXFLOW_UTILS_VARIANT_H 

#include "mpark/variant.hpp"
#include "utils/optional.h"

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

// from https://en.cppreference.com/w/cpp/types/conjunction
template<class...> struct conjunction : std::true_type { };
template<class B1> struct conjunction<B1> : B1 { };
template<class B1, class... Bn>
struct conjunction<B1, Bn...>
    : std::conditional<bool(B1::value), conjunction<Bn...>, B1>::type {};



template <class ...Args1, class ...Args2>
struct variant_join_helper<mpark::variant<Args1...>, mpark::variant<Args2...>> {
    using type = mpark::variant<Args1..., Args2...>;
};

template <class Variant1, class Variant2>
using variant_join = typename variant_join_helper<Variant1, Variant2>::type;

template <typename T, typename Variant> struct is_in_variant;

template <typename T, typename ...Rest> struct is_in_variant<T, variant<T, Rest...>> : std::true_type { };
template <typename T, typename Head, typename ...Rest> struct is_in_variant<T, variant<Head, Rest...>> :
  is_in_variant<T, variant<Rest...>> { };
template <typename T> struct is_in_variant<T, variant<>> : std::false_type { };

template <typename Variant1, typename Variant2> struct is_subeq_variant;
template <typename Head, typename Variant2, typename ...Rest> struct is_subeq_variant<variant<Head, Rest...>, Variant2> 
  : conjunction<is_in_variant<Head, Variant2>, is_subeq_variant<variant<Rest...>, Variant2>> { };
template <typename Variant2> struct is_subeq_variant<variant<>, Variant2>
  : std::true_type { };

template <typename Out>
struct VariantWidenFunctor {
  template <typename T>
  Out operator()(T const &t) const {
    return Out(t);
  }
};

template <typename Out>
struct VariantNarrowFunctor {
  template <typename T>
  typename std::enable_if<is_in_variant<T, Out>::value, optional<Out>>::type operator()(T const &t) const {
    return Out(t);
  }

  template <typename T>
  typename std::enable_if<!is_in_variant<T, Out>::value, optional<Out>>::type operator()(T const &t) const {
    return nullopt; 
  }
};

template <typename VariantOut, 
          typename VariantIn,
          typename = std::enable_if<is_subeq_variant<VariantIn, VariantOut>::value>>
VariantOut widen(VariantIn const &v) {
  return visit(VariantWidenFunctor<VariantOut>{}, v);
}

template <typename VariantOut, 
          typename VariantIn,
          typename = std::enable_if<is_subeq_variant<VariantOut, VariantIn>::value>>
optional<VariantOut> narrow(VariantIn const &v) {
  return visit(VariantNarrowFunctor<VariantOut>{}, v);
}

}

#endif 

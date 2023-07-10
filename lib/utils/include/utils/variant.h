#ifndef _FLEXFLOW_UTILS_VARIANT_H
#define _FLEXFLOW_UTILS_VARIANT_H

#include "mpark/variant.hpp"
#include "utils/optional.h"
#include "utils/type_traits.h"

namespace FlexFlow {

/* using mp = mpark; */

/* template <typename ...Ts> */
/* using variant = ::mpark::variant<Ts...>; */

using namespace ::mpark;

/* template <typename T> */
/* using optional = ::tl::optional<T>; */

/* template <typename T> */
/* using get = ::mpark::get; */

/* template <typename T> */
/* using holds_alternative = ::mpark::holds_alternative<T>; */

/* template <typename T> */
/* using visit = mpark::visit<T>; */

/* template <class ...Args1, class ...Args2> */
/* struct variant_join_helper<mpark::variant<Args1...>,
 * mpark::variant<Args2...>> { */
/*     using type = mpark::variant<Args1..., Args2...>; */
/* }; */
template <template <typename, typename = void> class Cond, typename... Ts>
struct elements_satisfy_impl<Cond, void, variant<Ts...>>
    : elements_satisfy<Cond, std::tuple<Ts...>> {};

template <typename T, typename Variant>
struct is_in_variant;
template <typename T, typename... Rest>
struct is_in_variant<T, variant<T, Rest...>> : std::true_type {};
template <typename T, typename Head, typename... Rest>
struct is_in_variant<T, variant<Head, Rest...>>
    : is_in_variant<T, variant<Rest...>> {};
template <typename T>
struct is_in_variant<T, variant<>> : std::false_type {};

template <typename T, size_t Idx, typename Variant>
struct variant_idx_helper;
template <typename T, size_t Idx, typename... Rest>
struct variant_idx_helper<T, Idx, variant<T, Rest...>>
    : std::integral_constant<int, Idx> {};
template <typename T, size_t Idx, typename Head, typename... Rest>
struct variant_idx_helper<T, Idx, variant<Head, Rest...>>
    : variant_idx_helper<T, (Idx + 1), variant<Rest...>> {};

template <typename T, typename Variant>
struct index_of_type : variant_idx_helper<T, 0, Variant> {
  static_assert(is_in_variant<T, Variant>::value, "");
};

static_assert(index_of_type<int, variant<float, double, int, bool>>::value == 2,
              "");

template <typename Variant1, typename Variant2>
struct is_subeq_variant;
template <typename Head, typename Variant2, typename... Rest>
struct is_subeq_variant<variant<Head, Rest...>, Variant2>
    : conjunction<is_in_variant<Head, Variant2>,
                  is_subeq_variant<variant<Rest...>, Variant2>> {};
template <typename Variant2>
struct is_subeq_variant<variant<>, Variant2> : std::true_type {};

template <typename Variant1, typename Variant2, typename Enable = void>
struct variant_join_helper;

template <typename Head, typename... Args1, typename... Args2>
struct variant_join_helper<
    variant<Head, Args1...>,
    variant<Args2...>,
    typename std::enable_if<
        !is_in_variant<Head, variant<Args2...>>::value>::type> {
  using type = typename variant_join_helper<variant<Args1...>,
                                            variant<Head, Args2...>>::type;
};

template <typename Head, typename... Args1, typename... Args2>
struct variant_join_helper<
    variant<Head, Args1...>,
    variant<Args2...>,
    typename std::enable_if<
        is_in_variant<Head, variant<Args2...>>::value>::type> {
  using type =
      typename variant_join_helper<variant<Args1...>, variant<Args2...>>::type;
};

template <typename... Args2>
struct variant_join_helper<variant<>, variant<Args2...>> {
  using type = variant<Args2...>;
};

template <class Variant1, class Variant2>
using variant_join = typename variant_join_helper<Variant1, Variant2>::type;

template <class Variant1, typename... T>
using variant_add = variant_join<Variant1, variant<T...>>;

static_assert(
    std::is_same<variant_join<variant<int, float>, variant<float, double>>,
                 variant<int, float, double>>::value,
    "");
static_assert(std::is_same<variant_join<variant<int>, variant<float, double>>,
                           variant<int, float, double>>::value,
              "");
static_assert(
    std::is_same<variant_join<variant<int>, variant<int>>, variant<int>>::value,
    "");

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
  typename std::enable_if<is_in_variant<T, Out>::value, optional<Out>>::type
      operator()(T const &t) const {
    return Out(t);
  }

  template <typename T>
  typename std::enable_if<!is_in_variant<T, Out>::value, optional<Out>>::type
      operator()(T const &t) const {
    return nullopt;
  }
};

template <
    typename VariantOut,
    typename VariantIn,
    typename = std::enable_if<is_subeq_variant<VariantIn, VariantOut>::value>>
VariantOut widen(VariantIn const &v) {
  return visit(VariantWidenFunctor<VariantOut>{}, v);
}

template <
    typename VariantOut,
    typename VariantIn,
    typename = std::enable_if<is_subeq_variant<VariantOut, VariantIn>::value>>
optional<VariantOut> narrow(VariantIn const &v) {
  return visit(VariantNarrowFunctor<VariantOut>{}, v);
}

template <typename VariantOut, typename VariantIn>
optional<VariantOut> cast(VariantIn const &v) {
  return narrow<VariantOut>(widen<variant_join<VariantIn, VariantOut>>(v));
}

} // namespace FlexFlow

#endif

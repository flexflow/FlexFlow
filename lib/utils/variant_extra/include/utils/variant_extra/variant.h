#ifndef _FLEXFLOW_UTILS_VARIANT_H
#define _FLEXFLOW_UTILS_VARIANT_H

#include <variant>
#include <type_traits>
#include <optional>
#include <functional>
#include "type/is_in_variant.h"

namespace FlexFlow {

/* using mp = mpark; */

/* template <typename ...Ts> */
/* using variant = ::mpark::variant<Ts...>; */

/* template <typename T> */
/* using std::optional = ::tl::std::optional<T>; */

/* template <typename T> */
/* using get = ::mpark::get; */

/* template <typename T> */
/* using holds_alternative = ::mpark::holds_alternative<T>; */

/* template <typename Pack, typename... Args> */
/* struct pack_contains_all_of; */

/* template <typename HeadNeedle, typename... NeedleRest, typename... Haystack> */
/* struct pack_contains_all_of<pack<Haystack...>, HeadNeedle, NeedleRest...> */
/*     : conjunction<pack_contains_type<pack<Haystack...>, HeadNeedle>, */
/*                   pack_contains_all_of<pack<Haystack...>, NeedleRest...>> {}; */

/* template <typename... Haystack> */
/* struct pack_contains_all_of<pack<Haystack...>> : std::false_type {}; */

/* template <typename... Needles, typename... Haystack> */
/* struct pack_contains_all_of<variant<Haystack...>, Needles...> */
/*     : pack_contains_all_of<pack<Haystack...>, Needles...> {}; */

/* template <typename T> */
/* using visit = mpark::visit<T>; */

/* template <class ...Args1, class ...Args2> */
/* struct variant_join_helper<mpark::variant<Args1...>,
 * mpark::variant<Args2...>> { */
/*     using type = mpark::variant<Args1..., Args2...>; */
/* }; */

template <typename T, size_t Idx, typename Variant>
struct variant_idx_helper;
template <typename T, size_t Idx, typename... Rest>
struct variant_idx_helper<T, Idx, std::variant<T, Rest...>>
    : std::integral_constant<int, Idx> {};
template <typename T, size_t Idx, typename Head, typename... Rest>
struct variant_idx_helper<T, Idx, std::variant<Head, Rest...>>
    : variant_idx_helper<T, (Idx + 1), std::variant<Rest...>> {};

template <typename T, typename Variant>
struct index_of_type : variant_idx_helper<T, 0, Variant> {
  static_assert(is_in_variant<T, Variant>::value, "");
};

static_assert(index_of_type<int, std::variant<float, double, int, bool>>::value == 2,
              "");

template <typename Variant1, typename Variant2>
struct is_subeq_variant;
template <typename Head, typename Variant2, typename... Rest>
struct is_subeq_variant<std::variant<Head, Rest...>, Variant2>
    : std::conjunction<is_in_variant<Head, Variant2>,
                  is_subeq_variant<std::variant<Rest...>, Variant2>> {};
template <typename Variant2>
struct is_subeq_variant<std::variant<>, Variant2> : std::true_type {};

template <typename Variant1, typename Variant2, typename Enable = void>
struct variant_join_helper;

template <typename Head, typename... Args1, typename... Args2>
struct variant_join_helper<
    std::variant<Head, Args1...>,
    std::variant<Args2...>,
    typename std::enable_if<
        !is_in_variant<Head, std::variant<Args2...>>::value>::type> {
  using type = typename variant_join_helper<std::variant<Args1...>,
                                            std::variant<Head, Args2...>>::type;
};

template <typename Head, typename... Args1, typename... Args2>
struct variant_join_helper<
    std::variant<Head, Args1...>,
    std::variant<Args2...>,
    typename std::enable_if<
        is_in_variant<Head, std::variant<Args2...>>::value>::type> {
  using type =
      typename variant_join_helper<std::variant<Args1...>, std::variant<Args2...>>::type;
};

template <typename... Args2>
struct variant_join_helper<std::variant<>, std::variant<Args2...>> {
  using type = std::variant<Args2...>;
};

template <class Variant1, class Variant2>
using variant_join = typename variant_join_helper<Variant1, Variant2>::type;

template <class Variant1, typename... T>
using variant_add = variant_join<Variant1, std::variant<T...>>;

static_assert(
    std::is_same<variant_join<std::variant<int, float>, std::variant<float, double>>,
                 std::variant<int, float, double>>::value,
    "");
static_assert(std::is_same<variant_join<std::variant<int>, std::variant<float, double>>,
                           std::variant<int, float, double>>::value,
              "");
static_assert(
    std::is_same<variant_join<std::variant<int>, std::variant<int>>, std::variant<int>>::value,
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
  typename std::enable_if<is_in_variant<T, Out>::value, std::optional<Out>>::type
      operator()(T const &t) const {
    return Out(t);
  }

  template <typename T>
  typename std::enable_if<!is_in_variant<T, Out>::value, std::optional<Out>>::type
      operator()(T const &t) const {
    return std::nullopt;
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
    typename Container,
    typename VariantIn = typename Container::value_type,
    typename = std::enable_if<is_subeq_variant<VariantIn, VariantOut>::value>>
auto widen(Container const &c) -> decltype(transform(
    c, std::declval<std::function<VariantOut(VariantIn const &)>>())) {
  return transform(c, [](VariantIn const &i) { return widen<VariantOut>(i); });
}

template <
    typename VariantOut,
    typename VariantIn,
    typename = std::enable_if<is_subeq_variant<VariantOut, VariantIn>::value>>
std::optional<VariantOut> narrow(VariantIn const &v) {
  return visit(VariantNarrowFunctor<VariantOut>{}, v);
}

template <
    typename VariantOut,
    typename Container,
    typename VariantIn = typename Container::value_type,
    typename = std::enable_if<is_subeq_variant<VariantIn, VariantOut>::value>>
auto narrow(Container const &c) -> decltype(transform(
    c,
    std::declval<std::function<std::optional<VariantOut>(VariantIn const &)>>())) {
  return transform(c, [](VariantIn const &i) { return narrow<VariantOut>(i); });
}

template <typename T1,
          typename T2,
          typename... Trest,
          typename VariantIn,
          typename = std::enable_if<
              !is_subeq_variant<std::variant<T1, T2, Trest...>, VariantIn>::value>>
std::optional<std::variant<T1, T2, Trest...>> narrow(VariantIn const &v) {
  return visit(VariantNarrowFunctor<std::variant<T1, T2, Trest...>>{}, v);
}

template <typename VariantOut, typename VariantIn>
std::optional<VariantOut> cast(VariantIn const &v) {
  return narrow<VariantOut>(widen<variant_join<VariantIn, VariantOut>>(v));
}

} // namespace FlexFlow

#endif

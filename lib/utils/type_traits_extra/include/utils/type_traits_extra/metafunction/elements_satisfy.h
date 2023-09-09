#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_METAFUNCTION_ELEMENTS_SATISFY_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_METAFUNCTION_ELEMENTS_SATISFY_H

#include <type_traits>
#include "is_nary.h"
#include <variant>

namespace FlexFlow {

template <template <typename...> class Cond, typename T, typename Enable = void>
struct elements_satisfy;

template <template <typename...> class Cond, typename T>
inline constexpr bool elements_satisfy_v = elements_satisfy<Cond, T>::value;

template <template <typename...> class Cond, typename... Ts>
struct elements_satisfy_impl;

template <template <typename...> class Cond, typename Head, typename... Ts>
struct elements_satisfy_impl<Cond, Head, Ts...>
    : std::conjunction<Cond<Head>, elements_satisfy_impl<Cond, Ts...>> {};

template <template <typename...> class Cond>
struct elements_satisfy_impl<Cond> : std::true_type {};

/* template <template <typename...> class Cond, typename T> */
/* struct elements_satisfy<Cond, T, enable_if_t<!is_nary_metafunction<Cond, 1>::value>> { */
/*   static_assert(false, */
/*                 "Cannot call elements_satisfy with a metafunction with more " */
/*                 "than 1 argument"); */
/* }; */

template <template <typename...> class Cond, typename T>
struct elements_satisfy<Cond,
                        T,
                        enable_if_t<(is_nary_metafunction<Cond, 1>::value &&
                                     is_visitable<T>::value)>>
    : elements_satisfy<Cond, visit_as_tuple_t<T>> {};

template <template <typename...> class Cond, typename... Ts>
struct elements_satisfy<Cond,
                        variant<Ts...>,
                        enable_if_t<is_nary_metafunction<Cond, 1>::value>>
    : elements_satisfy_impl<Cond, Ts...> {};

template <template <typename...> class Cond, typename... Ts>
struct elements_satisfy<Cond,
                        std::tuple<Ts...>,
                        enable_if_t<is_nary_metafunction<Cond, 1>::value>>
    : elements_satisfy_impl<Cond, Ts...> {};

static_assert(
    elements_satisfy<is_equal_comparable, std::tuple<int, float>>::value, "");

/* template <template <typename...> class Cond, typename T, typename Enable> */
/* struct violating_element { */
/*   static_assert(false, */
/*                 "Cannot call violating_element with a metafunction with more " */
/*                 "than 1 argument"); */
/* }; */

}

#endif

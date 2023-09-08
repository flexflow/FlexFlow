#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_ALL_SATISFY_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_ALL_SATISFY_H

namespace FLexFlow {

template <template <typename...> class Cond, typename... Ts>
struct elements_satisfy_impl;

template <template <typename...> class Cond, typename Head, typename... Ts>
struct elements_satisfy_impl<Cond, Head, Ts...>
    : conjunction<Cond<Head>, elements_satisfy_impl<Cond, Ts...>> {};

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

template <template <typename...> class Cond, typename... Ts>
struct violating_element_impl;

template <template <typename...> class Cond, typename Head, typename... Ts>
struct violating_element_impl<Cond, Head, Ts...>
    : type_identity<
          std::conditional_t<Cond<Head>::value,
                             typename violating_element_impl<Cond, Ts...>::type,
                             Head>> {};

template <template <typename...> class Cond>
struct violating_element_impl<Cond> : type_identity<void> { };

/* template <template <typename...> class Cond, typename T, typename Enable> */
/* struct violating_element { */
/*   static_assert(false, */
/*                 "Cannot call violating_element with a metafunction with more " */
/*                 "than 1 argument"); */
/* }; */

template <template <typename...> class Cond, typename T>
struct violating_element<Cond,
                         T,
                         enable_if_t<(is_nary_metafunction<Cond, 1>::value &&
                                      is_visitable<T>::value)>>
    : violating_element<Cond, visit_as_tuple_t<T>> {};

template <template <typename...> class Cond, typename... Ts>
struct violating_element<Cond,
                         variant<Ts...>,
                         enable_if_t<is_nary_metafunction<Cond, 1>::value>>
    : violating_element_impl<Cond, Ts...> {};

template <template <typename...> class Cond, typename... Ts>
struct violating_element<Cond,
                         std::tuple<Ts...>,
                         enable_if_t<is_nary_metafunction<Cond, 1>::value>>
    : violating_element_impl<Cond, Ts...> {};

}

#endif

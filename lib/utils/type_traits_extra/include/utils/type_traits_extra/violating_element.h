#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_VIOLATING_ELEMENT_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_VIOLATING_ELEMENT_H

#include "utils/backports/type_identity.h"
#include <type_traits>

namespace FlexFlow {

template <template <typename...> class Cond, typename T, typename Enable = void>
struct violating_element;

template <template <typename...> class Cond, typename T>
using violating_element_t = typename violating_element<Cond, T>::type;

template <template <typename...> class Cond, typename... Ts>
struct violating_element_impl;

template <template <typename...> class Cond, typename Head, typename... Ts>
struct violating_element_impl<Cond, Head, Ts...>
    : type_identity<
          std::conditional_t<Cond<Head>::value,
                             typename violating_element_impl<Cond, Ts...>::type,
                             Head>> {};

template <template <typename...> class Cond>
struct violating_element_impl<Cond> : type_identity<void> {};

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

} // namespace FlexFlow

#endif

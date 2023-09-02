#ifndef _FLEXFLOW_UTILS_INCLUDE_TYPE_TRAITS_H
#define _FLEXFLOW_UTILS_INCLUDE_TYPE_TRAITS_H

#include "utils/type_traits.decl.h"
#include "utils/invoke.h"
#include "utils/metafunction.h"
#include "utils/type_traits_core.h"
#include "utils/visitable_core.h"
#include <iostream>
#include <type_traits>

namespace FlexFlow {

template <typename T>
struct is_rc_copy_virtual_compliant
    : conjunction<negation<disjunction<std::is_copy_constructible<T>,
                                       std::is_copy_assignable<T>,
                                       std::is_move_constructible<T>,
                                       std::is_move_assignable<T>>>,
                  std::has_virtual_destructor<T>> {};

template <typename T, typename Enable>
struct is_clonable : std::false_type {};

template <typename T>
struct is_clonable<T, void_t<decltype(std::declval<T>().clone())>>
    : std::true_type {};

template <typename T, typename Enable = void>
struct is_streamable : std::false_type {};

template <typename T>
struct is_streamable<T, void_t<decltype(std::cout << std::declval<T>())>>
    : std::true_type {};

template <typename T, typename Enable>
struct is_lt_comparable : std::false_type {};

template <typename T>
struct is_lt_comparable<
    T,
    void_t<decltype((bool)(std::declval<T>() < std::declval<T>()))>>
    : std::true_type {};

template <template <typename...> class Cond, typename... Ts>
struct elements_satisfy_impl;

template <template <typename...> class Cond, typename Head, typename... Ts>
struct elements_satisfy_impl<Cond, Head, Ts...>
    : conjunction<Cond<Head>, elements_satisfy_impl<Cond, Ts...>> {};

template <template <typename...> class Cond>
struct elements_satisfy_impl<Cond> : std::true_type {};

template <template <typename...> class Cond, typename T, typename Enable>
struct elements_satisfy {
  static_assert(!is_nary_metafunction<Cond, 1>::value,
                "Cannot call elements_satisfy with a metafunction with more "
                "than 1 argument");
};

template <template <typename...> class Cond, typename T>
struct elements_satisfy<Cond,
                        T,
                        enable_if_t<(is_nary_metafunction<Cond, 1>::value &&
                                     is_visitable<T>::value)>>
    : elements_satisfy<Cond, visit_as_tuple_t<T>> {};

template <template <typename...> class Cond, typename... Ts>
struct elements_satisfy<Cond,
                        std::tuple<Ts...>,
                        enable_if_t<is_nary_metafunction<Cond, 1>::value>>
    : elements_satisfy_impl<Cond, Ts...> {};

static_assert(
    elements_satisfy<is_equal_comparable, std::tuple<int, float>>::value, "");

template <typename... Ts> struct types_are_all_same : std::false_type {};

template <> struct types_are_all_same<> : std::true_type {};

template <typename T> struct types_are_all_same<T> : std::true_type {};

template <typename Head, typename Next, typename... Rest>
struct types_are_all_same<Head, Next, Rest...> : conjunction<std::is_same<Head, Next>, types_are_all_same<Head, Rest...>> {};

} // namespace FlexFlow

#endif

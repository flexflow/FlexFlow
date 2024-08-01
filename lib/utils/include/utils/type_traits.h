#ifndef _FLEXFLOW_UTILS_INCLUDE_TYPE_TRAITS_H
#define _FLEXFLOW_UTILS_INCLUDE_TYPE_TRAITS_H

#include "utils/metafunction.h"
#include "utils/type_traits_core.h"
#include "utils/visitable_core.h"
#include <iostream>
#include <type_traits>

namespace FlexFlow {

#define DEBUG_PRINT_TYPE(...)                                                  \
  using Hello =                                                                \
      typename __VA_ARGS__ ::some_type_field_that_probably_will_never_exist

#define RC_COPY_VIRTUAL_MSG                                                    \
  "https://isocpp.github.io/CppCoreGuidelines/"                                \
  "CppCoreGuidelines#Rc-copy-virtual"

#define CHECK_RC_COPY_VIRTUAL_COMPLIANT(...)                                   \
  static_assert(                                                               \
      !std::is_copy_constructible<__VA_ARGS__>::value,                         \
      #__VA_ARGS__                                                             \
      " should not be copy-constructible. See " RC_COPY_VIRTUAL_MSG);          \
  static_assert(!std::is_copy_assignable<__VA_ARGS__>::value,                  \
                #__VA_ARGS__                                                   \
                " should not be copy-assignable. See " RC_COPY_VIRTUAL_MSG);   \
  static_assert(                                                               \
      !std::is_move_constructible<__VA_ARGS__>::value,                         \
      #__VA_ARGS__                                                             \
      " should not be move-constructible. See " RC_COPY_VIRTUAL_MSG);          \
  static_assert(!std::is_move_assignable<__VA_ARGS__>::value,                  \
                #__VA_ARGS__                                                   \
                " should not be move-assignable. See " RC_COPY_VIRTUAL_MSG);   \
  static_assert(std::has_virtual_destructor<__VA_ARGS__>::value,               \
                #__VA_ARGS__                                                   \
                " should have a virtual destructor. See " RC_COPY_VIRTUAL_MSG)

#define CHECK_NOT_ABSTRACT(...)                                                \
  static_assert(                                                               \
      !std::is_abstract<__VA_ARGS__>::value,                                   \
      #__VA_ARGS__                                                             \
      " should not be abstract (are you missing a virtual method override?)");

template <typename T>
struct is_rc_copy_virtual_compliant
    : conjunction<negation<disjunction<std::is_copy_constructible<T>,
                                       std::is_copy_assignable<T>,
                                       std::is_move_constructible<T>,
                                       std::is_move_assignable<T>>>,
                  std::has_virtual_destructor<T>> {};

template <typename T, typename Enable = void>
struct is_clonable : std::false_type {};

template <typename T>
struct is_clonable<T, void_t<decltype(std::declval<T>().clone())>>
    : std::true_type {};

template <typename T, typename Enable = void>
struct is_streamable : std::false_type {};

template <typename T>
struct is_streamable<T, void_t<decltype(std::cout << std::declval<T>())>>
    : std::true_type {};

template <template <typename...> class Cond, typename... Ts>
struct elements_satisfy_impl;

template <template <typename...> class Cond, typename Head, typename... Ts>
struct elements_satisfy_impl<Cond, Head, Ts...>
    : conjunction<Cond<Head>, elements_satisfy_impl<Cond, Ts...>> {};

template <template <typename...> class Cond>
struct elements_satisfy_impl<Cond> : std::true_type {};

template <template <typename...> class Cond, typename T, typename Enable = void>
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

} // namespace FlexFlow

#endif

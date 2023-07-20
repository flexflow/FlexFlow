#ifndef _FLEXFLOW_UTILS_INCLUDE_TYPE_TRAITS_H
#define _FLEXFLOW_UTILS_INCLUDE_TYPE_TRAITS_H

#include "utils/invoke.h"
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

template <typename T>
struct is_rc_copy_virtual_compliant
    : conjunction<negation<disjunction<std::is_copy_constructible<T>,
                                       std::is_copy_assignable<T>,
                                       std::is_move_constructible<T>,
                                       std::is_move_assignable<T>>>,
                  std::has_virtual_destructor<T>> {};

template <typename T, typename Enable = void>
struct is_streamable : std::false_type {};

template <typename T>
struct is_streamable<T, void_t<decltype(std::cout << std::declval<T>())>>
    : std::true_type {};

template <typename T, typename Enable = void>
struct is_equal_comparable : std::false_type {};

template <typename T>
struct is_equal_comparable<
    T,
    void_t<decltype((bool)(std::declval<T>() == std::declval<T>()))>>
    : std::true_type {};

template <typename T, typename Enable = void>
struct is_neq_comparable : std::false_type {};

template <typename T>
struct is_neq_comparable<
    T,
    void_t<decltype((bool)(std::declval<T>() != std::declval<T>()))>>
    : std::true_type {};

template <typename T, typename Enable = void>
struct is_lt_comparable : std::false_type {};

template <typename T>
struct is_lt_comparable<
    T,
    void_t<decltype((bool)(std::declval<T>() < std::declval<T>()))>>
    : std::true_type {};

template <typename T, typename Enable = void>
struct is_hashable : std::false_type {};

template <typename T>
struct is_hashable<
    T,
    void_t<decltype((size_t)(std::declval<std::hash<T>>()(std::declval<T>())))>>
    : std::true_type {};

template <template <typename, typename = void> class Cond,
          typename Enable,
          typename... Ts>
struct elements_satisfy_impl;

template <template <typename, typename = void> class Cond, typename T>
struct elements_satisfy : elements_satisfy_impl<Cond, void, T> {};

template <template <typename, typename = void> class Cond, typename T>
struct elements_satisfy_impl<
    Cond,
    typename std::enable_if<is_visitable<T>::value>::type,
    T> : elements_satisfy<Cond, visit_as_tuple_t<T>> {};

template <template <typename, typename = void> class Cond,
          typename Head,
          typename... Ts>
struct elements_satisfy<Cond, std::tuple<Head, Ts...>>
    : conjunction<Cond<Head>, elements_satisfy<Cond, std::tuple<Ts...>>> {};

template <template <typename, typename = void> class Cond>
struct elements_satisfy<Cond, std::tuple<>> : std::true_type {};

static_assert(
    elements_satisfy<is_equal_comparable, std::tuple<int, float>>::value, "");

template <typename T>
using is_default_constructible = std::is_default_constructible<T>;

template <typename T>
using is_copy_constructible = std::is_copy_constructible<T>;

template <typename T>
using is_move_constructible = std::is_move_constructible<T>;

template <typename T>
using is_copy_assignable = std::is_copy_assignable<T>;

template <typename T>
using is_move_assignable = std::is_move_assignable<T>;

template <typename T>
struct is_well_behaved_value_type_no_hash
    : conjunction<is_equal_comparable<T>,
                  is_neq_comparable<T>,
                  is_copy_constructible<T>,
                  is_move_constructible<T>,
                  is_copy_assignable<T>,
                  is_move_assignable<T>> {};

#define CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(...)                               \
  static_assert(is_copy_constructible<__VA_ARGS__>::value,                     \
                #__VA_ARGS__ " should be copy-constructible");                 \
  static_assert(is_move_constructible<__VA_ARGS__>::value,                     \
                #__VA_ARGS__ " should be move-constructible");                 \
  static_assert(is_copy_assignable<__VA_ARGS__>::value,                        \
                #__VA_ARGS__ " should be copy-assignable");                    \
  static_assert(is_move_assignable<__VA_ARGS__>::value,                        \
                #__VA_ARGS__ " should be move-assignable")

#define CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(...)                             \
  CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(__VA_ARGS__);                            \
  static_assert(is_equal_comparable<__VA_ARGS__>::value,                       \
                #__VA_ARGS__ " should support operator==");                    \
  static_assert(is_neq_comparable<__VA_ARGS__>::value,                         \
                #__VA_ARGS__ " should support operator!=");

template <typename T>
struct is_well_behaved_value_type
    : conjunction<is_well_behaved_value_type_no_hash<T>, is_hashable<T>> {};

#define CHECK_WELL_BEHAVED_VALUE_TYPE(...)                                     \
  CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(__VA_ARGS__);                          \
  static_assert(is_hashable<__VA_ARGS__>::value,                               \
                #__VA_ARGS__ " should support std::hash")

} // namespace FlexFlow

#endif

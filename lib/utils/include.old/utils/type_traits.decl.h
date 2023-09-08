#ifndef _FLEXFLOW_UTILS_INCLUDE_TYPE_TRAITS_DECL_H
#define _FLEXFLOW_UTILS_INCLUDE_TYPE_TRAITS_DECL_H

#include "utils/invoke.h"
#include "utils/metafunction.h"
#include "utils/type_traits_core.h"
#include "utils/visitable_core.h"
#include <iostream>
#include <type_traits>

namespace FlexFlow {

#define DEBUG_PRINT_TYPE(...)                                                  \
  using debug_print_fake_type = __VA_ARGS__;                                   \
  using debug_print_fake_type_2 =                                              \
      typename debug_print_fake_type::the_type_that_is_causing_the_failure


#define CHECK_NOT_ABSTRACT(...)                                                \
  static_assert(                                                               \
      !std::is_abstract<__VA_ARGS__>::value,                                   \
      #__VA_ARGS__                                                             \
      " should not be abstract (are you missing a virtual method override?)");

template <typename T, typename Enable = void>
struct is_clonable;
template <typename T>
inline constexpr bool is_clonable_v = is_clonable<T>::value;

template <typename T, typename Enable = void>
struct is_lt_comparable;
template <typename T>
inline constexpr bool is_lt_comparable_v = is_lt_comparable<T>::value;

template <template <typename...> class Cond, typename T, typename Enable = void>
struct elements_satisfy;
template <template <typename...> class Cond, typename T>
inline constexpr bool elements_satisfy_v = elements_satisfy<Cond, T>::value;

template <template <typename...> class Cond, typename T, typename Enable = void>
struct violating_element;
template <template <typename...> class Cond, typename T>
using violating_element_t = typename violating_element<Cond, T>::type;

template <typename T>
struct THE_FAILING_ELEMENT_IS : std::is_same<T, void> {
}; // for helping out with static assertion messages

template <template <typename...> typename Cond, typename T>
using pretty_elements_satisfy =
    THE_FAILING_ELEMENT_IS<violating_element_t<Cond, T>>;

template <typename... Ts>
struct types_are_all_same;
template <typename... Ts>
inline constexpr bool types_are_all_same_v = types_are_all_same<Ts...>::value;

} // namespace FlexFlow

#endif

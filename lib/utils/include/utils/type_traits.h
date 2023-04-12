#ifndef _FLEXFLOW_UTILS_INCLUDE_TYPE_TRAITS_H
#define _FLEXFLOW_UTILS_INCLUDE_TYPE_TRAITS_H

#include <type_traits>
#include <iostream>
#include "utils/visitable_core.h"
#include "utils/type_traits_core.h"

namespace FlexFlow {

#define RC_COPY_VIRTUAL_MSG "https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rc-copy-virtual"

template <typename T>
struct is_rc_copy_virtual_compliant 
  : conjunction<
      negation<disjunction<
        std::is_copy_constructible<T>,
        std::is_copy_assignable<T>,
        std::is_move_constructible<T>,
        std::is_move_assignable<T>
      >>,
      std::has_virtual_destructor<T>
    > 
{ };

template< typename... Ts >
struct make_void { typedef void type; };
 
template< typename... Ts >
using void_t = typename make_void<Ts...>::type;

template <typename T, typename Enable = void>
struct is_streamable : std::false_type { };

template <typename T>
struct is_streamable<T, void_t<decltype(std::cout << std::declval<T>())>>
  : std::true_type { };

template <typename T, typename Enable = void>
struct is_equal_comparable : std::false_type { };

template <typename T>
struct is_equal_comparable<T, void_t<decltype((bool)(std::declval<T>() == std::declval<T>()))>>
  : std::true_type { };

template <typename T, typename Enable = void>
struct is_neq_comparable : std::false_type { };

template <typename T>
struct is_neq_comparable<T, void_t<decltype((bool)(std::declval<T>() != std::declval<T>()))>>
  : std::true_type { };

template <typename T, typename Enable = void>
struct is_lt_comparable : std::false_type { };

template <typename T>
struct is_lt_comparable<T, void_t<decltype((bool)(std::declval<T>() < std::declval<T>()))>>
  : std::true_type { };

template <typename T, typename Enable = void>
struct is_hashable : std::false_type { };

template <typename T>
struct is_hashable<T, void_t<decltype((size_t)(std::declval<std::hash<T>>()(std::declval<T>())))>>
  : std::true_type { };

template <template <typename, typename = void> typename Cond, 
          typename T, 
          typename Enable = void> 
struct elements_satisfy;

template <template <typename, typename = void> typename Cond, typename T>
struct elements_satisfy<Cond, T, typename std::enable_if<is_visitable<T>::value>::type>
  : elements_satisfy<Cond, visit_as_tuple<T>> { };

template <template <typename, typename = void> typename Cond, typename Head, typename ...Ts>
struct elements_satisfy<Cond, std::tuple<Head, Ts...>>
  : conjunction<Cond<Head>, elements_satisfy<Cond, std::tuple<Ts...>>> { };

template <template <typename, typename = void> typename Cond>
struct elements_satisfy<Cond, std::tuple<>> : std::true_type { };

static_assert(elements_satisfy<is_equal_comparable, std::tuple<int, float>>::value, "");

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

}

#endif

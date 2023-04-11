#ifndef _FLEXFLOW_UTILS_INCLUDE_TYPE_TRAITS_H
#define _FLEXFLOW_UTILS_INCLUDE_TYPE_TRAITS_H

#include <type_traits>
#include <iostream>
#include "utils/visitable.h"

namespace FlexFlow {

template <bool b>
using bool_constant = std::integral_constant<bool, b>;

// from https://en.cppreference.com/w/cpp/types/conjunction
template<class...> struct conjunction : std::true_type { };
template<class B1> struct conjunction<B1> : B1 { };
template<class B1, class... Bn>
struct conjunction<B1, Bn...>
    : std::conditional<bool(B1::value), conjunction<Bn...>, B1>::type {};

// from https://en.cppreference.com/w/cpp/types/negation
template<class B>
struct negation : bool_constant<!bool(B::value)> { };

// from https://en.cppreference.com/w/cpp/types/disjunction
template<class...> struct disjunction : std::false_type { };
template<class B1> struct disjunction<B1> : B1 { };
template<class B1, class... Bn>
struct disjunction<B1, Bn...>
    : std::conditional<bool(B1::value), B1, disjunction<Bn...>>::type  { };

template <typename LHS, typename RHS>
struct implies
    : disjunction<RHS, negation<LHS>> { };

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

template <template <typename, typename = void> typename Cond, typename T, typename Enable = void> struct elements_satisfy;

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

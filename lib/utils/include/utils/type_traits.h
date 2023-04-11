#ifndef _FLEXFLOW_UTILS_INCLUDE_TYPE_TRAITS_H
#define _FLEXFLOW_UTILS_INCLUDE_TYPE_TRAITS_H

#include <type_traits>

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

}

#endif

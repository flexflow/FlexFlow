#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_CORE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_CORE_H

#include <type_traits>

namespace FlexFlow {
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template <bool b>
using bool_constant = std::integral_constant<bool, b>;

// from https://en.cppreference.com/w/cpp/types/conjunction
template <class...>
struct conjunction : std::true_type {};
template <class B1>
struct conjunction<B1> : B1 {};
template <class B1, class... Bn>
struct conjunction<B1, Bn...>
    : std::conditional<bool(B1::value), conjunction<Bn...>, B1>::type {};

// from https://en.cppreference.com/w/cpp/types/negation
template <class B>
struct negation : bool_constant<!bool(B::value)> {};

// from https://en.cppreference.com/w/cpp/types/disjunction
template <class...>
struct disjunction : std::false_type {};
template <class B1>
struct disjunction<B1> : B1 {};
template <class B1, class... Bn>
struct disjunction<B1, Bn...>
    : std::conditional<bool(B1::value), B1, disjunction<Bn...>>::type {};

template <typename LHS, typename RHS>
struct implies : disjunction<RHS, negation<LHS>> {};

template <class T>
struct type_identity {
  using type = T;
};

template <int idx>
struct infinite_recursion {
  using type = typename infinite_recursion<(idx + 1)>::type;
};

template <bool Cond, typename True, typename False>
using conditional_t = typename std::conditional<Cond, True, False>::type;

template <typename L, typename R>
struct is_equal : bool_constant<(L::value == R::value)> {};

template <typename L, typename R>
struct biconditional : bool_constant<(bool(L::value) == bool(R::value))> {};

template <typename... Ts>
struct make_void {
  typedef void type;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

template <typename T, typename Enable = void, typename... Args>
struct is_list_initializable_impl : std::false_type {};

template <typename T, typename... Args>
struct is_list_initializable_impl<T,
                                  void_t<decltype(T{std::declval<Args>()...})>,
                                  Args...> : std::true_type {};

template <typename T, typename... Args>
using is_list_initializable = is_list_initializable_impl<T, void, Args...>;

static_assert(is_list_initializable<int, int>::value, "");

static_assert(
    std::is_same<
        conditional_t<false, infinite_recursion<0>, type_identity<bool>>::type,
        bool>::value,
    "");
/* static_assert(std::is_same<typename if_then_else<true, int, bool>::type,
 * bool>::value, ""); */

template <typename From, typename To, typename Enable = void>
struct is_static_castable : std::false_type {};

template <typename From, typename To>
struct is_static_castable<
    From,
    To,
    void_t<decltype(static_cast<To>(std::declval<From>()))>> : std::true_type {
};

template <typename C, typename Tag>
struct supports_iterator_tag
    : std::is_base_of<Tag,
                      typename std::iterator_traits<
                          typename C::iterator>::iterator_category> {};

#define CHECK_SUPPORTS_ITERATOR_TAG(TAG, ...)                                  \
  static_assert(supports_iterator_tag<__VA_ARGS__, TAG>::value,                \
                #__VA_ARGS__ " does not support required iterator tag " #TAG);

} // namespace FlexFlow

#endif

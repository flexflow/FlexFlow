#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_CORE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_CORE_H

#include <iterator>
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

template <typename... Args>
struct pack {};

template <typename T, typename... Args>
struct pack_contains_type;

template <typename T, typename Head, typename... Rest>
struct pack_contains_type<T, Head, Rest...>
    : disjunction<std::is_same<T, Head>, pack_contains_type<T, Rest...>> {};

template <typename T, typename... Args>
struct pack_contains_type<T, pack<Args...>> : pack_contains_type<T, Args...> {};

template <typename T>
struct pack_contains_type<T> : std::false_type {};

static_assert(pack_contains_type<int, float, double, int, char>::value, "");
static_assert(!pack_contains_type<int, float, double, char>::value, "");

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

template <typename T, typename Enable = void>
struct is_equal_comparable : std::false_type {};

template <typename T>
struct is_equal_comparable<
    T,
    void_t<decltype(std::declval<T>() == std::declval<T>())>>
    /* std::is_same< */
    /*   decltype(), */
    /*   bool */
    /* >::value */
    /* >> */
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

template <typename T>
inline constexpr bool is_hashable_v = is_hashable<T>::value;

template <typename T, typename Enable = void>
struct is_plusable : std::false_type {};

template <typename T>
struct is_plusable<T,
                   void_t<decltype((T)(std::declval<T>() + std::declval<T>()))>>
    : std::true_type {};

static_assert(is_plusable<int>::value, "");

template <typename T, typename Enable = void>
struct is_minusable : std::false_type {};

template <typename T>
struct is_minusable<
    T,
    void_t<decltype((T)(std::declval<T>() - std::declval<T>()))>>
    : std::true_type {};

template <typename T, typename Enable = void>
struct is_timesable : std::false_type {};

template <typename T>
struct is_timesable<
    T,
    void_t<decltype((T)(std::declval<T>() * std::declval<T>()))>>
    : std::true_type {};

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

#define CHECK_HASHABLE(...)                                                    \
  static_assert(is_hashable<__VA_ARGS__>::value,                               \
                #__VA_ARGS__ " should be hashable (but is not)");

#define CHECK_WELL_BEHAVED_VALUE_TYPE(...)                                     \
  CHECK_WELL_BEHAVED_VALUE_TYPE_NO_HASH(__VA_ARGS__);                          \
  CHECK_HASHABLE(__VA_ARGS__)

} // namespace FlexFlow

#endif

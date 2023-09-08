#ifndef _FLEXFLOW_UTILS_TYPE_TRAITS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_LIST_INITIALIZABLE_H
#define _FLEXFLOW_UTILS_TYPE_TRAITS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_LIST_INITIALIZABLE_H

#include <type_traits>

namespace FlexFlow {

template <typename T, typename Enable = void, typename... Args>
struct is_list_initializable_impl : std::false_type {};

template <typename T, typename... Args>
struct is_list_initializable_impl<T,
                                  std::void_t<decltype(T{std::declval<Args>()...})>,
                                  Args...> : std::true_type {};

template <typename T, typename... Args>
using is_list_initializable = is_list_initializable_impl<T, void, Args...>;

template <typename T, typename... Args>
inline constexpr bool is_list_initializable_v = is_list_initializable<T, Args...>::value;

}

#endif

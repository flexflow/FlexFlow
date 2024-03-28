#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_METAFUNCTION_NUM_ARGS_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_METAFUNCTION_NUM_ARGS_H

#include <type_traits>

namespace FlexFlow {

template <template <typename...> class Cond,
          typename Enable = void,
          typename... Args>
struct typelevel_function_num_args {
  static constexpr int value =
      typelevel_function_num_args<Cond, Enable, int, Args...>::value;
};

template <template <typename...> class Cond, typename... Args>
struct typelevel_function_num_args<
    Cond,
    std::void_t<decltype(std::declval<Cond<Args...>>())>,
    Args...> : std::integral_constant<int, (sizeof...(Args))> {};

template <template <typename...> class Cond, typename... Args>
inline constexpr int typelevel_function_num_args_v =
    typelevel_function_num_args<Cond, Args...>::value;

} // namespace FlexFlow

#endif

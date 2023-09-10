#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_METAFUNCTION_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_METAFUNCTION_H

#include "type_traits_core.h"

namespace FlexFlow {

template <class, class = void>
struct has_type_member : std::false_type {};

template <class T>
struct has_type_member<T, void_t<typename T::type>> : std::true_type {};

template <template <typename...> class Cond,
          typename Enable = void,
          typename... Args>
struct metafunction_num_args {
  static constexpr int value =
      metafunction_num_args<Cond, Enable, int, Args...>::value;
};

template <template <typename...> class Cond, typename... Args>
struct metafunction_num_args<Cond,
                             void_t<decltype(std::declval<Cond<Args...>>())>,
                             Args...>
    : std::integral_constant<int, (sizeof...(Args))> {};

/* template <template <typename...> class Func, int LIMIT, typename...Args> */
/* struct metafunction_num_args<Func, LIMIT, enable_if_t<(LIMIT == 0)>, Args...>
 * { */
/*   static_assert(false, "error"); */
/* }; */

template <template <typename...> class Func, int N, typename Enable = void>
struct is_nary_metafunction : std::false_type {};

template <template <typename...> class Func, int N>
struct is_nary_metafunction<
    Func,
    N,
    enable_if_t<(metafunction_num_args<Func>::value == N)>> : std::true_type {};

template <template <typename...> class Func, typename Enable, typename... Args>
struct internal_invoke_metafunction;

template <template <typename...> class Func, typename... Args>
struct internal_invoke_metafunction<
    Func,
    typename std::enable_if<(metafunction_num_args<Func>::value ==
                             (sizeof...(Args)))>::type,
    Args...> : Func<Args...> {};

template <template <typename...> class Func, typename... Args>
using invoke_metafunction = internal_invoke_metafunction<Func, void, Args...>;

} // namespace FlexFlow

#endif

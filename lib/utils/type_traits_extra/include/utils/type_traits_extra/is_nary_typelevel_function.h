#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_METAFUNCTION_IS_NARY_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_METAFUNCTION_IS_NARY_H

#include "typelevel_function_num_args.h"

namespace FlexFlow {

template <template <typename...> class Func, int N, typename Enable = void>
struct is_nary_typelevel_function : std::false_type {};

template <template <typename...> class Func, int N>
struct is_nary_typelevel_function<
    Func,
    N,
    std::enable_if_t<(typelevel_function_num_args<Func>::value == N)>>
    : std::true_type {};

template <template <typename...> class Func, int N>
inline constexpr bool is_nary_typelevel_function_v =
    is_nary_typelevel_function<Func, N>::value;

} // namespace FlexFlow

#endif

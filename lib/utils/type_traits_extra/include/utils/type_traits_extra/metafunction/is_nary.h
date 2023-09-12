#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_METAFUNCTION_IS_NARY_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_METAFUNCTION_IS_NARY_H

#include "num_args.h"

namespace FlexFlow {

template <template <typename...> class Func, int N, typename Enable = void>
struct is_nary_metafunction : std::false_type {};

template <template <typename...> class Func, int N>
struct is_nary_metafunction<
    Func,
    N,
    std::enable_if_t<(metafunction_num_args<Func>::value == N)>> : std::true_type {};

template <template <typename...> class Func, int N>
inline constexpr bool is_nary_metafunction_v = is_nary_metafunction<Func, N>::value;


} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_METAFUNCTION_REPEAT_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_METAFUNCTION_REPEAT_H

#include <type_traits>
#include "is_nary.h"
#include "utils/backports/type_identity.h"

namespace FlexFlow {

template <template <typename...> class Func, int N, typename Arg, typename Enable = void>
struct metafunction_repeat_impl { };

template <template <typename...> class Func, int N, typename Arg>
struct metafunction_repeat_impl<Func, N, Arg>
  : metafunction_repeat_impl<Func, N, typename Func<Arg>::type> { };

template <template <typename...> class Func, typename Arg>
struct metafunction_repeat_impl<Func, 0, Arg>
  : type_identity<Arg> { };

template <template <typename...> class Func, int N, typename Arg, typename Enable = void>
struct metafunction_repeat { };

template <template <typename...> class Func, int N, typename Arg>
struct metafunction_repeat<Func, N, Arg, std::enable_if_t<is_nary_metafunction_v<Func, 1> && (N >= 0)>> 
  : metafunction_repeat_impl<Func, N, Arg> { };

template <template <typename...> class Func, int N, typename Arg>
using metafunction_repeat_t = typename metafunction_repeat<Func, N, Arg>::type;

} // namespace FlexFlow

#endif

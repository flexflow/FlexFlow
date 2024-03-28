#ifndef _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_FUNCTIONS_GET_H
#define _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_FUNCTIONS_GET_H

#include "utils/compile_time_sequence/sequence.h"
#include <type_traits>
#include <fmt/fmt.h>
#include "utils/ff_exceptions/mk_runtime_error.h"

namespace FlexFlow {

/* template <typename F, int... S> */
/* auto seq_get(F const &f, int i, seq<S...> const &s) */
/*     -> decltype(f(std::declval<std::integral_constant<int, 0>>())); */

template <typename F, int X, int... S>
auto seq_get(F const &f, int i, seq<X, S...> const &s)
    -> decltype(f(std::declval<std::integral_constant<int, 0>>())) {
  if (X == i) {
    return f(std::integral_constant<int, X>{});
  } else {
    return seq_get(f, i, seq<S...>{});
  }
};

template <typename F>
auto seq_get(F const &f, int i, seq<> const &)
    -> decltype(f(std::declval<std::integral_constant<int, 0>>())) {
  throw mk_runtime_error(fmt::format("Failed seq_get for index {}", i));
}

} // namespace FlexFlow

#endif

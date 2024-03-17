#ifndef _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_FOR_EACH_H
#define _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_FOR_EACH_H

#include "utils/compile_time_sequence/sequence.h"
#include <type_traits>

namespace FlexFlow {

template <typename F, int X, int... S>
void seq_for_each(F const &f, seq<X, S...> const &) {
  f(std::integral_constant<int, X>{});
  seq_for_each(f, seq<S...>{});
}

template <typename F>
void seq_for_each(F const &f, seq<> const &) {}

} // namespace FlexFlow

#endif

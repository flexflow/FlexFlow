#ifndef _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_TAIL_H
#define _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_TAIL_H

#include "utils/compile_time_sequence/sequence.h"

namespace FlexFlow {

template <typename Seq>
struct seq_tail;

template <int X, int... S>
struct seq_tail<seq<X, S...>> {
  using type = seq<S...>;
};

template <>
struct seq_tail<seq<>> {
  using type = seq<>;
};

template <typename Seq>
using seq_tail_t = typename seq_tail<Seq>::type;

} // namespace FlexFlow

#endif

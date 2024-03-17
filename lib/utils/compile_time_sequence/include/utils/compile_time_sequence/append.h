#ifndef _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_APPEND_H
#define _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_APPEND_H

#include "utils/compile_time_sequence/sequence.h"

namespace FlexFlow {

template <typename Seq, int ToAppend>
struct seq_append;

template <int X, int... S>
struct seq_append<seq<S...>, X> {
  using type = seq<S..., X>;
};

template <typename Seq, int ToAppend>
using seq_append_t = typename seq_append<Seq, ToAppend>::type;

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_PREPEND_H
#define _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_PREPEND_H

#include "utils/compile_time_sequence/sequence.h"

namespace FlexFlow {

template <int X, typename Seq>
struct seq_prepend;

template <int X, int... S>
struct seq_prepend<X, seq<S...>> {
  using type = seq<X, S...>;
};

template <int X, typename Seq>
using seq_prepend_t = typename seq_prepend<X, Seq>::type;

} // namespace FlexFlow

#endif

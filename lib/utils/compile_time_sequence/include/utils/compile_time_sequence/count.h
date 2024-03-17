#ifndef _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_COUNT_H
#define _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_COUNT_H

#include "utils/compile_time_sequence/sequence.h"
#include "utils/compile_time_sequence/append.h"

namespace FlexFlow {

template <int n>
struct seq_count;

template <int n>
struct seq_count {
  using type = typename seq_append<typename seq_count<(n - 1)>::type, n>::type;
};

template <>
struct seq_count<-1> {
  using type = seq<>;
};

template <int n>
using seq_count_t = typename seq_count<n>::type;

} // namespace FlexFlow

#endif

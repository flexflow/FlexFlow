#ifndef _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_ENUMERATE_ARGS_H
#define _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_ENUMERATE_ARGS_H

#include "utils/compile_time_sequence/count.h"

namespace FlexFlow {

template <typename... Args>
struct seq_enumerate_args;

template <typename... Args>
struct seq_enumerate_args {
  using type = seq_count_t<(int)(sizeof...(Args)) - 1>;
};

template <typename... Args>
using seq_enumerate_args_t = typename seq_enumerate_args<Args...>::type;

} // namespace FlexFlow

#endif

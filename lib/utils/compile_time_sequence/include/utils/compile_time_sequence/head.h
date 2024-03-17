#ifndef _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_HEAD_H
#define _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_HEAD_H

#include "utils/compile_time_sequence/sequence.h"
#include <type_traits>

namespace FlexFlow {

template <typename Seq>
struct seq_head;

template <int X, int... S>
struct seq_head<seq<X, S...>> : std::integral_constant<int, X> {};

template <>
struct seq_head<seq<>> : std::integral_constant<int, -1> {};

template <typename Seq>
inline constexpr int seq_head_v = seq_head<Seq>::value;

} // namespace FlexFlow

#endif

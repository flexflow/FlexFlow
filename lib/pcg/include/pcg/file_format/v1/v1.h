#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_H

#include "graphs.h"
#include "pcg/computation_graph.h"

namespace FlexFlow {

template <typename T,
          typename std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T to_v1(T const &t) {
  return t;
}

template <typename V1T, typename... Args>
V1T to_v1(variant<Args...> const &v) {
  NOT_IMPLEMENTED();
}

template <typename V1T, typename T>
optional<V1T> to_v1(optional<T> const &t) {
  if (t.has_value()) {
    return to_v1(t.value());
  } else {
    return nullopt;
  }
}

template <typename T,
          typename std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T to_v1(req<T> const &t) {
  return T(t);
}

} // namespace FlexFlow

#endif

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

template <typename T,
          typename std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
T from_v1(T const &vt) {
  return vt;
}

std::string to_v1(std::string const &s);
std::string from_v1(std::string const &vs);

template <typename V1T, typename T>
optional<V1T> to_v1(optional<T> const &t) {
  if (t.has_value()) {
    return to_v1(t.value());
  } else {
    return nullopt;
  }
}

template <typename T, typename V1T>
optional<T> from_v1(optional<V1T> const &vt) {
  if (vt.has_value()) {
    return from_v1(vt.value());
  } else {
    return nullopt;
  }
}

} // namespace FlexFlow

#endif

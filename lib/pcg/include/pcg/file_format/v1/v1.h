#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_H

#include "graphs.h"
#include "pcg/computation_graph.h"

namespace FlexFlow {

template <typename V1T, typename T>
optional<V1T> to_v1(optional<T> const &t) {
  if (t.has_value()) {
    return to_v1(t.value());
  } else {
    return nullopt;
  }
}

} // namespace FlexFlow

#endif

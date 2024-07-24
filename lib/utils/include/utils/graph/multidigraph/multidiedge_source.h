#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_MULTIDIEDGE_SOURCE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_MULTIDIGRAPH_MULTIDIEDGE_SOURCE_H

#include "utils/graph/multidigraph/multidiedge.dtg.h"

namespace FlexFlow {

struct MultiDiEdgeSource {
public:
  MultiDiEdgeSource();

  MultiDiEdge new_multidiedge();
private:
  static size_t next_available_multidiedge_id;
};

} // namespace FlexFlow

#endif

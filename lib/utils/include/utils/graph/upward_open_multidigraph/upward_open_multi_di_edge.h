#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UPWARD_OPEN_MULTIDIGRAPH_UPWARD_OPEN_MULTI_DI_EDGE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UPWARD_OPEN_MULTIDIGRAPH_UPWARD_OPEN_MULTI_DI_EDGE_H

#include "utils/graph/open_multidigraph/open_multi_di_edge.dtg.h"
#include "utils/graph/upward_open_multidigraph/upward_open_multi_di_edge.dtg.h"

namespace FlexFlow {

OpenMultiDiEdge open_multidiedge_from_upward_open(UpwardOpenMultiDiEdge const &);

} // namespace FlexFlow

#endif

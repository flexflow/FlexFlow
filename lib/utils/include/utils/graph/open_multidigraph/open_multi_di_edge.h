#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_MULTIDIGRAPH_OPEN_MULTI_DI_EDGE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_MULTIDIGRAPH_OPEN_MULTI_DI_EDGE_H

#include "utils/graph/open_multidigraph/open_multi_di_edge.dtg.h"

namespace FlexFlow {

bool is_input_edge(OpenMultiDiEdge const &);
bool is_output_edge(OpenMultiDiEdge const &);
bool is_standard_edge(OpenMultiDiEdge const &);

} // namespace FlexFlow

#endif

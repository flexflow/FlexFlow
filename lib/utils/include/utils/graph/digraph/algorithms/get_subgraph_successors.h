#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_SUBGRAPH_SUCCESSORS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_SUBGRAPH_SUCCESSORS_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

std::unordered_set<Node> get_subgraph_successors(DiGraphView const &, 
                                                 std::unordered_set<Node> const &);

} // namespace FlexFlow

#endif

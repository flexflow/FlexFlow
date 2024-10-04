#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_HAS_EDGE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_HAS_EDGE_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

bool digraph_has_edge(DiGraphView const &, DirectedEdge const &);

} // namespace FlexFlow

#endif

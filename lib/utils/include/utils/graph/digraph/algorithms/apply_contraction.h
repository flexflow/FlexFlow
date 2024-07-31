#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_APPLY_CONTRACTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_APPLY_CONTRACTION_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

DiGraphView apply_contraction(DiGraphView const &g,
                              std::unordered_map<Node, Node> const &nodes);

} // namespace FlexFlow

#endif

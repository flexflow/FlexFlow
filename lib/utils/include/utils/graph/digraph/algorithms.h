#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_H

#include "utils/graph/digraph/digraph.h"

namespace FlexFlow {

std::unordered_set<DirectedEdge> get_edges(DiGraphView const &);
std::unordered_set<Node> get_sources(DiGraphView const &);
std::unordered_set<Node> get_sinks(DiGraphView const &);

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_PREDECESSORS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_PREDECESSORS_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

std::unordered_map<Node, std::unordered_set<Node>> get_predecessors(DiGraphView const &);
std::unordered_set<Node> get_predecessors(DiGraphView const &, Node const &);
std::unordered_map<Node, std::unordered_set<Node>>
    get_predecessors(DiGraphView const &, std::unordered_set<Node> const &);

} // namespace FlexFlow

#endif

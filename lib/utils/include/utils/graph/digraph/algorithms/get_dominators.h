#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_DOMINATORS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_GET_DOMINATORS_H

#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

std::unordered_map<Node, std::unordered_set<Node>>
    get_dominators(DiGraphView const &);
std::unordered_set<Node> get_dominators(DiGraphView const &, Node const &);
std::unordered_set<Node> get_dominators(DiGraphView const &,
                                        std::unordered_set<Node> const &);

} // namespace FlexFlow

#endif

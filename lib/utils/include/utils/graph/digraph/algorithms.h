#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_H

#include "utils/graph/digraph/digraph.h"

namespace FlexFlow {

std::unordered_set<Node> get_nodes(DiGraph const &);
std::unordered_set<DirectedEdge> get_edges(DirectedEdge const &);
std::unordered_set<DirectedEdge> get_incoming_edges(DiGraph const &, Node const &);
std::unordered_set<DirectedEdge> get_outgoing_edges(DiGraph const &, Node const &);
std::unordered_set<Node> get_sources(DiGraph const &);
std::vector<Node> get_topological_ordering(DiGraph const &);

} // namespace FlexFlow

#endif

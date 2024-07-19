#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DIGRAPH_ALGORITHMS_H

#include "utils/graph/digraph/digraph.h"

namespace FlexFlow {

std::unordered_set<DirectedEdge> get_edges(DiGraphView const &);
std::unordered_set<DirectedEdge> get_incoming_edges(DiGraphView const &,
                                                    Node const &);
std::unordered_map<Node, std::unordered_set<DirectedEdge>>
    get_incoming_edges(DiGraphView const &, std::unordered_set<Node> const &);
std::unordered_set<DirectedEdge> get_outgoing_edges(DiGraphView const &,
                                                    Node const &);
std::unordered_map<Node, std::unordered_set<DirectedEdge>>
    get_outgoing_edges(DiGraphView const &, std::unordered_set<Node> const &);
std::unordered_set<Node> get_sources(DiGraphView const &);
std::unordered_set<Node> get_sinks(DiGraphView const &);
std::vector<Node> get_topological_ordering(DiGraphView const &);
std::optional<bool> is_acyclic(DiGraphView const &);

DiGraphView flipped(DiGraphView const &g);

std::unordered_set<Node> get_predecessors(DiGraphView const &, Node const &);
std::unordered_map<Node, std::unordered_set<Node>>
    get_predecessors(DiGraphView const &, std::unordered_set<Node> const &);

} // namespace FlexFlow

#endif

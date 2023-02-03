#ifndef _FLEXFLOW_UTILS_GRAPH_ALGORITHMS_H
#define _FLEXFLOW_UTILS_GRAPH_ALGORITHMS_H

#include "multidigraph.h"
#include "digraph.h"
#include "undirected.h"
#include <vector>
#include <unordered_map>

namespace FlexFlow {
namespace utils {
namespace graph {

std::unordered_set<Node> get_nodes(IMultiDiGraphView const &);

std::unordered_set<multidigraph::Edge> get_edges(IMultiDiGraphView const &);
std::unordered_set<digraph::Edge> get_edges(IDiGraphView const &);
std::unordered_set<undirected::Edge> get_edges(IUndirectedGraphView const &);

std::unordered_set<multidigraph::Edge> get_incoming_edges(IMultiDiGraphView const &, std::unordered_set<Node> const &);
std::unordered_set<digraph::Edge> get_incoming_edges(IDiGraphView const &, std::unordered_set<Node> const &);
std::unordered_set<undirected::Edge> get_incoming_edges(IUndirectedGraphView const &, std::unordered_set<Node> const &);

std::unordered_map<Node, std::unordered_set<Node>> get_predecessors(IMultiDiGraphView const &, std::unordered_set<Node> const &);
std::unordered_set<Node, std::unordered_set<Node>> get_predecessors(IDiGraphView const &);

bool is_acyclic(IMultiDiGraphView const &, std::unordered_set<Node> const &);
bool is_acyclic(IDiGraphView const &);

std::vector<Node> topo_sort(IMultiDiGraphView const &);
std::vector<Node> topo_sort(IDiGraphView const &);

std::unordered_map<Node, std::unordered_set<std::size_t>> dominators(IMultiDiGraphView const &);

}
}
}

#endif 

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

std::unordered_set<Node> get_nodes(IMultiDiGraph const &);
std::unordered_set<multidigraph::Edge> get_edges(IMultiDiGraph const &);
std::unordered_set<digraph::Edge> get_edge(IDiGraph const &);
std::unordered_set<undirected::Edge> get_edge(IUndirectedGraph const &);

std::unordered_set<multidigraph::Edge> get_incoming_edges(IMultiDiGraph const &);
std::unordered_set<digraph::Edge> get_incoming_edges(IDiGraph const &);
std::unordered_set<undirected::Edge> get_incoming_edges(IUndirectedGraph const &);

std::unordered_set<Node> get_predecessors(IMultiDiGraph const &);
std::unordered_set<Node> get_predecessors(IDiGraph const &);

std::vector<Node> topo_sort(IMultiDiGraph const &);
std::vector<Node> topo_sort(IDiGraph const &);

std::unordered_map<Node, std::unordered_set<std::size_t>> dominators(IMultiDiGraph const &);

}
}
}

#endif 

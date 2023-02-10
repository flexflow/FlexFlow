#ifndef _FLEXFLOW_UTILS_GRAPH_ALGORITHMS_H
#define _FLEXFLOW_UTILS_GRAPH_ALGORITHMS_H

#include "node.h"
#include "multidigraph.h"
#include "digraph.h"
#include "undirected.h"
#include <vector>
#include <unordered_map>

namespace FlexFlow {
namespace utils {

std::vector<Node> add_nodes(IGraph &, int);
std::unordered_set<Node> get_nodes(IGraphView const &);

std::size_t num_nodes(IGraphView const &);

std::unordered_set<MultiDiEdge> get_edges(IMultiDiGraphView const &);
std::unordered_set<DirectedEdge> get_edges(IDiGraphView const &);
std::unordered_set<UndirectedEdge> get_edges(IUndirectedGraphView const &);

std::unordered_set<MultiDiEdge> get_incoming_edges(IMultiDiGraphView const &, Node const &);
std::unordered_set<DirectedEdge> get_incoming_edges(IDiGraphView const &, Node const &);

std::unordered_set<MultiDiEdge> get_incoming_edges(IMultiDiGraphView const &, std::unordered_set<Node>);
std::unordered_set<DirectedEdge> get_incoming_edges(IDiGraphView const &, std::unordered_set<Node> const &);

std::unordered_set<MultiDiEdge> get_outgoing_edges(IMultiDiGraphView const &, std::unordered_set<Node> const &);
std::unordered_set<DirectedEdge> get_outgoing_edges(IDiGraphView const &, std::unordered_set<Node> const &);
std::unordered_set<UndirectedEdge> get_outgoing_edges(IUndirectedGraphView const &, std::unordered_set<Node> const &);

std::unordered_map<Node, std::unordered_set<Node>> get_predecessors(IMultiDiGraphView const &, std::unordered_set<Node> const &);
std::unordered_map<Node, std::unordered_set<Node>> get_predecessors(IDiGraphView const &, std::unordered_set<Node> const &);

std::unordered_set<Node> get_sources(IDiGraphView const &);
std::unordered_set<Node> get_sinks(IDiGraphView const &);

bool is_acyclic(IMultiDiGraphView const &, std::unordered_set<Node> const &);
tl::optional<bool> is_acyclic(IDiGraphView const &);

std::vector<Node> topo_sort(IMultiDiGraphView const &);
std::vector<Node> topo_sort(IDiGraphView const &);

std::unordered_map<Node, std::unordered_set<std::size_t>> dominators(IMultiDiGraphView const &);

/* std::vector<Node> boundary_dfs_ordering(IDiGraphView const &, std::unordered_set<Node> const &starting_points); */
std::vector<Node> dfs_ordering(IDiGraphView const &, std::unordered_set<Node> const &starting_points);
std::vector<Node> unchecked_dfs_ordering(IDiGraphView const &, std::unordered_set<Node> const &starting_points);

}
}

#endif 

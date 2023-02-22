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

void remove_node(IMultiDiGraph &, Node const &);
void remove_node(IDiGraph &, Node const &);
void remove_node(IUndirectedGraph &, Node const &);

std::size_t num_nodes(IGraphView const &);

void add_edges(IMultiDiGraph &, std::vector<MultiDiEdge> const &);
void add_edges(IDiGraph &, std::vector<DirectedEdge> const &);
void add_edges(IUndirectedGraph &, std::vector<UndirectedEdge> const &);

bool contains_edge(IMultiDiGraph const &, MultiDiEdge const &);
bool contains_edge(IDiGraph const &, DirectedEdge const &);
bool contains_edge(IUndirectedGraph const &, UndirectedEdge const &);

void remove_edges(IMultiDiGraph &, std::unordered_set<MultiDiEdge> const &);
void remove_edges(IDiGraph &, std::unordered_set<DirectedEdge> const &);
void remove_edges(IUndirectedGraph &, std::vector<UndirectedEdge> const &);

std::unordered_set<MultiDiEdge> get_edges(IMultiDiGraphView const &);
std::unordered_set<DirectedEdge> get_edges(IDiGraphView const &);
std::unordered_set<UndirectedEdge> get_edges(IUndirectedGraphView const &);

std::unordered_set<MultiDiEdge> get_incoming_edges(IMultiDiGraphView const &, Node const &);
std::unordered_set<DirectedEdge> get_incoming_edges(IDiGraphView const &, Node const &);
std::unordered_set<MultiDiEdge> get_incoming_edges(IMultiDiGraphView const &, std::unordered_set<Node>);
std::unordered_set<DirectedEdge> get_incoming_edges(IDiGraphView const &, std::unordered_set<Node> const &);

std::unordered_set<MultiDiEdge> get_outgoing_edges(IMultiDiGraphView const &, Node const &);
std::unordered_set<MultiDiEdge> get_outgoing_edges(IMultiDiGraphView const &, std::unordered_set<Node> const &);

std::unordered_set<DirectedEdge> get_outgoing_edges(IDiGraphView const &, Node const &);
std::unordered_set<DirectedEdge> get_outgoing_edges(IDiGraphView const &, std::unordered_set<Node> const &);

std::unordered_set<UndirectedEdge> get_node_edges(IUndirectedGraphView const &, Node const &);
std::unordered_set<UndirectedEdge> get_node_edges(IUndirectedGraphView const &, std::unordered_set<Node> const &);

std::unordered_set<Node> get_predecessors(IMultiDiGraphView const &, Node const &);
std::unordered_set<Node> get_predecessors(IDiGraphView const &, Node const &);
std::unordered_map<Node, std::unordered_set<Node>> get_predecessors(IMultiDiGraphView const &, std::unordered_set<Node> const &);
std::unordered_map<Node, std::unordered_set<Node>> get_predecessors(IDiGraphView const &, std::unordered_set<Node> const &);

std::unordered_set<Node> get_sources(IDiGraphView const &);
std::unordered_set<Node> get_sinks(IDiGraphView const &);

bool is_acyclic(IMultiDiGraphView const &, std::unordered_set<Node> const &);
tl::optional<bool> is_acyclic(IDiGraphView const &);

std::unordered_map<Node, std::unordered_set<Node>> dominators(IMultiDiGraphView const &);
std::unordered_map<Node, std::unordered_set<Node>> dominators(IDiGraphView const &);

/* std::vector<Node> boundary_dfs_ordering(IDiGraphView const &, std::unordered_set<Node> const &starting_points); */
std::vector<Node> dfs_ordering(IDiGraphView const &, std::unordered_set<Node> const &starting_points);
std::vector<Node> unchecked_dfs_ordering(IDiGraphView const &, std::unordered_set<Node> const &starting_points);
std::vector<Node> bfs_ordering(IDiGraphView const &, std::unordered_set<Node> const &starting_points);
std::vector<Node> topological_ordering(IDiGraphView const &);
std::vector<Node> topological_ordering(IMultiDiGraphView const &);
std::vector<Node> unchecked_topological_ordering(IDiGraphView const &);


}
}

#endif 

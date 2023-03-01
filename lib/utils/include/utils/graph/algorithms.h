#ifndef _FLEXFLOW_UTILS_GRAPH_ALGORITHMS_H
#define _FLEXFLOW_UTILS_GRAPH_ALGORITHMS_H

#include "node.h"
#include "multidigraph.h"
#include "digraph.h"
#include "undirected.h"
#include <vector>
#include <unordered_map>
#include "utils/dot_file.h"
#include "utils/containers.h"
#include "views.h"

namespace FlexFlow {
namespace utils {

std::vector<Node> add_nodes(IGraph &, int);
std::unordered_set<Node> get_nodes(IGraphView const &);

std::unordered_set<Node> query_nodes(IGraphView const &, std::unordered_set<Node> const &);

void remove_node(IMultiDiGraph &, Node const &);
void remove_node(IDiGraph &, Node const &);
void remove_node(IUndirectedGraph &, Node const &);

void remove_node_if_unused(IMultiDiGraph &, Node const &);
void remove_node_if_unused(IDiGraph &, Node const &);
void remove_node_if_unused(IUndirectedGraph &, Node const &);

void contract_node(IMultiDiGraph &, Node const &);
void contract_node(IDiGraph &, Node const &);
void contract_node(IUndirectedGraph &, Node const &);

std::size_t num_nodes(IGraphView const &);
bool empty(IGraphView const &);

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

std::unordered_set<UndirectedEdge> get_node_edges(IUndirectedGraphView const &, Node const &);

std::unordered_set<MultiDiEdge> get_incoming_edges(IMultiDiGraphView const &, Node const &);
std::unordered_set<DirectedEdge> get_incoming_edges(IDiGraphView const &, Node const &);
std::unordered_set<MultiDiEdge> get_incoming_edges(IMultiDiGraphView const &, std::unordered_set<Node>);
std::unordered_set<DirectedEdge> get_incoming_edges(IDiGraphView const &, std::unordered_set<Node> const &);

std::unordered_map<std::size_t, MultiDiEdge> get_incoming_edges_by_idx(IMultiDiGraphView const &, Node const &);
std::unordered_map<std::size_t, MultiDiEdge> get_outgoing_edges_by_idx(IMultiDiGraphView const &, Node const &);

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
std::unordered_set<Node> get_sources(IMultiDiGraphView const &);

std::unordered_set<Node> get_sinks(IDiGraphView const &);
std::unordered_set<Node> get_sinks(IMultiDiGraphView const &);

bool is_acyclic(IMultiDiGraphView const &, std::unordered_set<Node> const &);
tl::optional<bool> is_acyclic(IDiGraphView const &);
tl::optional<bool> is_acyclic(IMultiDiGraphView const &);

std::unordered_map<Node, std::unordered_set<Node>> get_dominators(IMultiDiGraphView const &);
std::unordered_map<Node, std::unordered_set<Node>> get_dominators(IDiGraphView const &);
std::unordered_set<Node> get_dominators(IDiGraphView const &, Node const &);
std::unordered_set<Node> get_dominators(IDiGraphView const &, std::unordered_set<Node> const &);

std::unordered_map<Node, std::unordered_set<Node>> get_post_dominators(IMultiDiGraphView const &);
std::unordered_map<Node, std::unordered_set<Node>> get_post_dominators(IDiGraphView const &);
std::unordered_map<Node, tl::optional<Node>> get_imm_dominators(IMultiDiGraphView const &);
std::unordered_map<Node, tl::optional<Node>> get_imm_dominators(IDiGraphView const &);
std::unordered_map<Node, tl::optional<Node>> get_imm_post_dominators(IMultiDiGraphView const &);
std::unordered_map<Node, tl::optional<Node>> get_imm_post_dominators(IDiGraphView const &);
tl::optional<Node> get_imm_post_dominator(IDiGraphView const &, Node const &);
tl::optional<Node> get_imm_post_dominator(IMultiDiGraphView const &, Node const &);
tl::optional<Node> get_imm_post_dominator(IDiGraphView const &, std::unordered_set<Node> const &);

/* std::vector<Node> boundary_dfs_ordering(IDiGraphView const &, std::unordered_set<Node> const &starting_points); */
std::vector<Node> get_dfs_ordering(IDiGraphView const &, std::unordered_set<Node> const &starting_points);
std::vector<Node> get_unchecked_dfs_ordering(IDiGraphView const &, std::unordered_set<Node> const &starting_points);
std::vector<Node> get_bfs_ordering(IDiGraphView const &, std::unordered_set<Node> const &starting_points);
std::vector<Node> get_topological_ordering(IDiGraphView const &);
std::vector<Node> get_topological_ordering(IMultiDiGraphView const &);
std::vector<Node> get_unchecked_topological_ordering(IDiGraphView const &);

std::vector<std::unordered_set<Node>> get_weakly_connected_components(IMultiDiGraphView const &);
std::vector<std::unordered_set<Node>> get_weakly_connected_components(IDiGraphView const &);
std::vector<std::unordered_set<Node>> get_connected_components(IUndirectedGraphView const &);

std::unordered_set<DirectedEdge> get_transitive_reduction_delta(IDiGraphView const &);

template <typename Impl>
Impl get_subgraph(IUndirectedGraphView const &g, std::unordered_set<Node> const &nodes) {
  return materialize_undirected_graph_view<Impl>(g, nodes);
}

template <typename Impl>
Impl get_subgraph(IDiGraphView const &g, std::unordered_set<Node> const &nodes) {
  return materialize_digraph_view<Impl>(unsafe_view_subgraph(g, nodes));
}

template <typename Impl>
Impl get_subgraph(IMultiDiGraphView const &g, std::unordered_set<Node> const &nodes) {
  return materialize_multidigraph_view<Impl>(unsafe_view_subgraph(g, nodes));
}

template <typename Impl>
Impl join(IMultiDiGraphView const &lhs, IMultiDiGraphView const &rhs) {
  return materialize_multidigraph_view<Impl>(unsafe_view_as_joined(lhs, rhs));
}

template <typename Impl>
Impl join(IDiGraphView const &lhs, IDiGraphView const &rhs) {
  return materialize_digraph_view<Impl>(unsafe_view_as_joined(lhs, rhs));
}

template <typename Impl>
Impl join(IUndirectedGraphView const &lhs, IUndirectedGraphView const &rhs) {
  return materialize_undirected_graph_view<Impl>(unsafe_view_as_joined(lhs, rhs));
}

void export_as_dot(DotFile<Node> &, 
                   IDiGraphView const &, 
                   std::function<RecordFormatter(Node const &)> const &, 
                   tl::optional<std::function<std::string(DirectedEdge const &)> const &> = tl::nullopt);

}
}

#endif 

/**
 * @file algorithms.h
 * @brief General Use Algorithms for the Main Graph library API.
 *
 * Copyright 2024 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _FLEXFLOW_UTILS_GRAPH_ALGORITHMS_H
#define _FLEXFLOW_UTILS_GRAPH_ALGORITHMS_H

#include "digraph.h"
#include "labelled_graphs.h"
#include "multidigraph.h"
#include "node.h"
#include "open_graphs.h"
#include "undirected.h"
#include "utils/containers.h"
#include "utils/dot_file.h"
#include "utils/exception.h"
#include "utils/graph/multidiedge.h"
#include "utils/graph/open_graph_interfaces.h"
#include "utils/optional.h"
#include "views.h"
#include <unordered_map>
#include <vector>

namespace FlexFlow {

std::vector<Node> add_nodes(Graph &g, int num_nodes);
std::vector<Node> add_nodes(UndirectedGraph &g, int num_nodes);
std::vector<Node> add_nodes(DiGraph &g, int num_nodes);
std::vector<Node> add_nodes(MultiDiGraph &g, int num_nodes);
std::vector<Node> add_nodes(OpenMultiDiGraph &g, int num_nodes);

std::vector<NodePort> add_node_ports(MultiDiGraph &g, int num_node_ports);

std::unordered_set<Node> get_nodes(GraphView const &g);
std::unordered_set<NodePort> get_present_node_ports(MultiDiGraphView const &g);

std::unordered_set<Node> get_nodes(OpenMultiDiEdge const &edge);

/**
 *  @brief Does this do anything?
 **/
std::unordered_set<Node> query_nodes(GraphView const &g,
                                     std::unordered_set<Node> const &nodes);

void remove_node(MultiDiGraph &g, Node const &node);
void remove_node(DiGraph &g, Node const &node);
void remove_node(UndirectedGraph &g, Node const &node);

void remove_node_if_unused(MultiDiGraph &g, Node const &node);
void remove_node_if_unused(DiGraph &g, Node const &node);
void remove_node_if_unused(UndirectedGraph &g, Node const &node);

/**
 * @brief modifies the given graph in-place by merging the 2 nodes into a single
 *one
 **/
void contract_node_inplace(MultiDiGraph &g, Node const &from, Node const &into);
void contract_node_inplace(DiGraph &g, Node const &from, Node const &into);
void contract_node_inplace(UndirectedGraph &g,
                           Node const &from,
                           Node const &into);

/**
 * @brief identical to `contract_node_inplace`, but leaves the graph g unchanged
 *and returns a new graph instead.
 **/
MultiDiGraphView contract_node(MultiDiGraphView const &g,
                               Node const &from,
                               Node const &into);
DiGraphView
    contract_node(DiGraphView const &g, Node const &from, Node const &into);
UndirectedGraphView contract_node(UndirectedGraphView const &g,
                                  Node const &from,
                                  Node const &into);

/**
 * @brief modifies the given graph in-place by splitting the given node into 2
 *separate nodes.
 **/
void contract_out_node_inplace(MultiDiGraph &g, Node const &node);
void contract_out_node_inplace(DiGraph &g, Node const &node);
void contract_out_node_inplace(UndirectedGraph &g, Node const &node);

/**
 * @brief identical to `contract_out_node_inplace`, but leaves the graph g
 *unchanged and returns a new graph instead.
 **/
MultiDiGraphView contract_out_node(MultiDiGraphView const &g, Node const &node);
DiGraphView contract_out_node(DiGraphView const &g, Node const &node);
UndirectedGraphView contract_out_node(UndirectedGraphView const &g,
                                      Node const &node);

/**
 * @brief applies the contraction from node a into node b for all pairs of nodes
 *(a,b) present in the map.
 **/
MultiDiGraphView apply_contraction(MultiDiGraphView const &g,
                                   std::unordered_map<Node, Node> const &nodes);
DiGraphView apply_contraction(DiGraphView const &g,
                              std::unordered_map<Node, Node> const &nodes);
UndirectedGraphView
    apply_contraction(UndirectedGraphView const &g,
                      std::unordered_map<Node, Node> const &nodes);

std::size_t num_nodes(GraphView const &g);
bool empty(GraphView const &g);

void add_edges(MultiDiGraph &g, std::vector<MultiDiEdge> const &edges);
void add_edges(DiGraph &g, std::vector<DirectedEdge> const &edges);
void add_edges(UndirectedGraph &g, std::vector<UndirectedEdge> const &edges);

bool contains_node(GraphView const &gv, Node const &node);

bool contains_edge(MultiDiGraphView const &gv, MultiDiEdge const &edge);
bool contains_edge(DiGraphView const &gv, DirectedEdge const &edge);
bool contains_edge(UndirectedGraphView const &gv, UndirectedEdge const &edge);

void remove_edges(MultiDiGraph &g,
                  std::unordered_set<MultiDiEdge> const &edges);
void remove_edges(DiGraph &g, std::unordered_set<DirectedEdge> const &edges);
void remove_edges(UndirectedGraph &g, std::vector<UndirectedEdge> const &edges);

std::unordered_set<Node> get_endpoints(UndirectedEdge const &edge);

std::unordered_set<MultiDiEdge> get_edges(MultiDiGraphView const &g);
std::unordered_set<DirectedEdge> get_edges(DiGraphView const &g);
std::unordered_set<UndirectedEdge> get_edges(UndirectedGraphView const &g);
std::unordered_set<UpwardOpenMultiDiEdge>
    get_edges(UpwardOpenMultiDiGraphView const &g);
std::unordered_set<DownwardOpenMultiDiEdge>
    get_edges(DownwardOpenMultiDiGraphView const &g);
std::unordered_set<OpenMultiDiEdge> get_edges(OpenMultiDiGraphView const &g);

std::unordered_set<UndirectedEdge> get_node_edges(UndirectedGraphView const &g,
                                                  Node const &node);

std::unordered_set<MultiDiOutput> get_outputs(MultiDiGraphView const &g);
std::unordered_set<MultiDiInput> get_inputs(MultiDiGraphView const &g);

std::unordered_set<OutputMultiDiEdge>
    get_open_outputs(OpenMultiDiGraphView const &g);
std::unordered_set<InputMultiDiEdge>
    get_open_inputs(OpenMultiDiGraphView const &g);

std::unordered_set<MultiDiEdge> get_incoming_edges(MultiDiGraphView const &g,
                                                   Node const &node);
std::unordered_set<DirectedEdge> get_incoming_edges(DiGraphView const &g,
                                                    Node const &node);
std::unordered_set<UpwardOpenMultiDiEdge>
    get_incoming_edges(UpwardOpenMultiDiGraphView const &g, Node const &node);
std::unordered_set<MultiDiEdge>
    get_incoming_edges(DownwardOpenMultiDiGraphView const &g, Node const &node);
std::unordered_set<UpwardOpenMultiDiEdge>
    get_incoming_edges(OpenMultiDiGraphView const &g, Node const &node);

std::unordered_set<MultiDiEdge>
    get_incoming_edges(MultiDiGraphView const &g,
                       std::unordered_set<Node> nodes);
std::unordered_set<DirectedEdge>
    get_incoming_edges(DiGraphView const &g,
                       std::unordered_set<Node> const &nodes);

std::unordered_map<NodePort, std::unordered_set<MultiDiEdge>>
    get_incoming_edges_by_idx(MultiDiGraphView const &g, Node const &node);
std::unordered_map<NodePort, std::unordered_set<MultiDiEdge>>
    get_outgoing_edges_by_idx(MultiDiGraphView const &g, Node const &node);

std::unordered_set<MultiDiEdge> get_outgoing_edges(MultiDiGraphView const &g,
                                                   Node const &node);
std::unordered_set<DirectedEdge> get_outgoing_edges(DiGraphView const &g,
                                                    Node const &node);
std::unordered_set<MultiDiEdge>
    get_outgoing_edges(UpwardOpenMultiDiGraphView const &g, Node const &node);
std::unordered_set<DownwardOpenMultiDiEdge>
    get_outgoing_edges(DownwardOpenMultiDiGraphView const &g, Node const &node);
std::unordered_set<DownwardOpenMultiDiEdge>
    get_outgoing_edges(OpenMultiDiGraphView const &g, Node const &node);

std::unordered_set<MultiDiEdge>
    get_outgoing_edges(MultiDiGraphView const &g,
                       std::unordered_set<Node> const &nodes);
std::unordered_set<DirectedEdge>
    get_outgoing_edges(DiGraphView const &g,
                       std::unordered_set<Node> const &nodes);

std::unordered_set<DirectedEdge>
    get_outgoing_edges(DiGraphView const &g, std::unordered_set<Node> const &nodes);

std::unordered_set<UndirectedEdge> get_node_edges(UndirectedGraphView const &g,
                                                  Node const &node);
std::unordered_set<UndirectedEdge>
    get_node_edges(UndirectedGraphView const &g,
                   std::unordered_set<Node> const &nodes);

std::unordered_set<Node> get_predecessors(DiGraphView const &g, Node const &node);
std::unordered_map<Node, std::unordered_set<Node>>
    get_predecessors(DiGraphView const &g, std::unordered_set<Node> const &nodes);

Node get_src_node(MultiDiEdge const &edge);
Node get_dst_node(MultiDiEdge const &edge);
Node get_dst_node(InputMultiDiEdge const &edge);
Node get_src_node(OutputMultiDiEdge const &edge);

struct GetSrcNodeFunctor {
  template <typename T>
  Node operator()(T const &t) const {
    return get_src_node(t);
  }
};

struct GetDstNodeFunctor {
  template <typename T>
  Node operator()(T const &t) const {
    return get_dst_node(t);
  }
};

template <typename... Args>
Node get_src_node(std::variant<Args...> const &t) {
  return visit(GetSrcNodeFunctor{}, t);
}

template <typename... Args>
Node get_dst_node(std::variant<Args...> const &t) {
  return visit(GetDstNodeFunctor{}, t);
}

NodePort get_src_idx(MultiDiEdge const &edge);
NodePort get_dst_idx(MultiDiEdge const &edge);
NodePort get_dst_idx(InputMultiDiEdge const &edge);
NodePort get_src_idx(OutputMultiDiEdge const &edge);

struct GetSrcIdxFunctor {
  template <typename T>
  NodePort operator()(T const &t) const {
    return get_src_idx(t);
  }
};

struct GetDstIdxFunctor {
  template <typename T>
  NodePort operator()(T const &t) const {
    return get_dst_idx(t);
  }
};

template <typename... Args>
NodePort get_src_idx(std::variant<Args...> const &t) {
  return visit(GetSrcIdxFunctor{}, t);
}

template <typename... Args>
NodePort get_dst_idx(std::variant<Args...> const &t) {
  return visit(GetDstIdxFunctor{}, t);
}

std::unordered_set<Node> get_neighbors(UndirectedGraphView const &g,
                                       Node const &node);

/**
 * @brief returns all neighboring nodes to the given node. 
 * @details When fetching the neighbors, the graph is treated as undirected. So a,b are neighbors if edge (a,b) or (b,a) is present.
*/
std::unordered_set<Node> get_neighbors(DiGraphView const &g, Node const &node);
std::unordered_set<Node> get_neighbors(MultiDiGraphView const &g, Node const &node);

// return the set of nodes without incoming edges
std::unordered_set<Node> get_sources(DiGraphView const &g);

// return the set of nodes without outgoing edges
std::unordered_set<Node> get_sinks(DiGraphView const &g);

std::unordered_set<Node> get_closed_sources(OpenMultiDiGraphView const &g);
std::unordered_set<Node> get_closed_sinks(OpenMultiDiGraphView const &g);
std::unordered_set<Node> get_open_sources(OpenMultiDiGraphView const &g);
std::unordered_set<Node> get_open_sinks(OpenMultiDiGraphView const &g);

bool is_acyclic(MultiDiGraphView const &g, std::unordered_set<Node> const &nodes);

/**
 * @brief If the graph has no nodes, std::nullopt is returned.
*/
std::optional<bool> is_acyclic(DiGraphView const &g);
std::optional<bool> is_acyclic(MultiDiGraphView const &g);

/**
 * @brief Computes the dominators for all nodes in a directed graph.
 * @details A node "d" dominates a node "n" if every path from all sources to "n" must go through "d". Note that every node dominates itself.
*/
std::unordered_map<Node, std::unordered_set<Node>>
    get_dominators(DiGraphView const &g);

/**
 * @brief Computes the dominators for a specific node in a directed graph.
 * @details A node "d" dominates a node "n" if every path from all sources to "n" must go through "d". Note that every node dominates itself.
*/
std::unordered_set<Node> get_dominators(DiGraphView const &g, Node const &nodes);

/**
 * @brief Computes the intersection of dominators for a set of nodes in a directed graph.
 * @details A node "d" dominates a node "n" if every path from all sources to "n" must go through "d". Note that every node dominates itself.
*/
std::unordered_set<Node> get_dominators(DiGraphView const &g,
                                        std::unordered_set<Node> const &nodes);

/**
 * @brief Computes all post-dominators in a directed graph.
 * @details A node "d" post-dominates a node "n" if every path from "n" to all sinks must go through "d". Note that every node post-dominates itself.
*/
std::unordered_map<Node, std::unordered_set<Node>>
    get_post_dominators(DiGraphView const &g);

/**
 * @brief Computes the immediate dominator for all nodes in a directed graph
 * @details An immediate dominator is the unique node that strictly dominates "n" but does not strictly dominate any other node that strictly dominates "n". Every node, except the source node(s), has an immediate dominator.
 *          Thus, the sources will have a std::nullopt as the associated dominator.
*/
std::unordered_map<Node, std::optional<Node>>
    get_imm_dominators(DiGraphView const &g);

/**
 * @brief Computes the immediate post-dominator for all nodes in a directed graph
 * @details An immediate post-dominator is the unique node that strictly post-dominates "n" but does not post-dominate any other post-dominator of "n". Every node, except the sink node(s), has an immediate dominator.
 *          Thus, the sinks will have a std::nullopt as the associated dominator.
*/
std::unordered_map<Node, std::optional<Node>>
    get_imm_post_dominators(DiGraphView const &g);

/**
 * @brief Computes the immediate post-dominator for the given node in a directed graph
 * @details An immediate post-dominator is the unique node that strictly post-dominates "n" but does not post-dominate any other post-dominator of "n". Every node, except the sink node(s), has an immediate dominator.
 *          Thus, the sinks will have a std::nullopt as the associated dominator.
*/
std::optional<Node> get_imm_post_dominator(DiGraphView const &g, Node const &node);
std::optional<Node> get_imm_post_dominator(MultiDiGraphView const &g,
                                           Node const &node);
std::optional<Node> get_imm_post_dominator(DiGraphView const &g,
                                           std::unordered_set<Node> const &nodes);

std::vector<Node>
    get_dfs_ordering(DiGraphView const &g,
                     std::unordered_set<Node> const &starting_points);
std::vector<Node>
    get_unchecked_dfs_ordering(DiGraphView const &g,
                               std::unordered_set<Node> const &starting_points);
std::vector<Node>
    get_bfs_ordering(DiGraphView const &g,
                     std::unordered_set<Node> const &starting_points);
std::vector<Node> get_topological_ordering(DiGraphView const &g);
// std::vector<Node> get_topological_ordering(MultiDiGraphView const &);
// std::vector<Node> get_topological_ordering(OpenMultiDiGraphView const &);
std::vector<Node> get_unchecked_topological_ordering(DiGraphView const &g);

std::vector<DirectedEdge> get_edge_topological_ordering(DiGraphView const &g);
std::vector<MultiDiEdge>
    get_edge_topological_ordering(MultiDiGraphView const &g);

std::unordered_set<std::unordered_set<Node>>
    get_weakly_connected_components(MultiDiGraphView const &g);
std::unordered_set<std::unordered_set<Node>>
    get_weakly_connected_components(DiGraphView const &g);
std::unordered_set<std::unordered_set<Node>>
    get_connected_components(UndirectedGraphView const &g);

std::unordered_set<DirectedEdge>
    get_transitive_reduction_delta(DiGraphView const &g);

// Describes a bi-partition of a given set of nodes
using GraphSplit =
    std::pair<std::unordered_set<Node>, std::unordered_set<Node>>;

std::pair<OutputMultiDiEdge, InputMultiDiEdge> split_edge(MultiDiEdge const &e);
MultiDiEdge unsplit_edge(OutputMultiDiEdge const &out_edge, InputMultiDiEdge const &in_edge);
/**
 * @brief For a given graph split, returns the cut-set, which is the set of edges that have one endpoint in each subset of the GraphSplit
*/
std::unordered_set<MultiDiEdge> get_cut_set(MultiDiGraphView const &g,
                                            GraphSplit const &split);

/**
 * @brief For a given set of nodes, returns the set of edges that have one endpoint in the set of nodes and the other endpoint outside of it.
*/
std::unordered_set<MultiDiEdge> get_cut_set(MultiDiGraphView const &g,
                                            std::unordered_set<Node> const &nodes);

bidict<MultiDiEdge, std::pair<OutputMultiDiEdge, InputMultiDiEdge>>
    get_edge_splits(MultiDiGraphView const &g, GraphSplit const &split);

UndirectedGraphView get_subgraph(UndirectedGraphView const &g,
                                 std::unordered_set<Node> const &nodes);
DiGraphView get_subgraph(DiGraphView const &g, std::unordered_set<Node> const &nodes);
MultiDiGraphView get_subgraph(MultiDiGraphView const &g,
                              std::unordered_set<Node> const &nodes);

template <typename SubgraphView>
OpenMultiDiGraphView get_subgraph(OpenMultiDiGraphView const &g,
                                  std::unordered_set<Node> const &nodes) {
  return OpenMultiDiGraphView::create<SubgraphView>(g, nodes);
}

std::unordered_map<Node, int> calculate_topo_rank(DiGraphView const &g);
Node get_node_with_greatest_topo_rank(std::unordered_set<Node> const &nodes,
                                      DiGraphView const &g);

MultiDiGraphView join(MultiDiGraphView const &lhs, MultiDiGraphView const &rhs);
DiGraphView join(DiGraphView const &lhs, DiGraphView const &rhs);
UndirectedGraphView join(UndirectedGraphView const &lhs,
                         UndirectedGraphView const &rhs);
/**
 * @brief Returns a digraph with all the edges flipped
*/
DiGraphView flipped(DiGraphView const &g);

DiGraphView with_added_edges(DiGraphView const &g,
                             std::unordered_set<DirectedEdge> const &edges);

UndirectedGraphView as_undirected(DiGraphView const &g);
MultiDiGraphView as_multidigraph(DiGraphView const &g);
DiGraphView as_digraph(UndirectedGraphView const &g);
OpenMultiDiGraphView as_openmultidigraph(MultiDiGraphView const &g);

void export_as_dot(
    DotFile<Node> &,
    DiGraphView const &,
    std::function<RecordFormatter(Node const &)> const &,
    std::optional<std::function<std::string(DirectedEdge const &)>> =
        std::nullopt);

} // namespace FlexFlow

#endif

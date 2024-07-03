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
 *separate ones
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
    get_outgoing_edges(DiGraphView const &, std::unordered_set<Node> const &);

std::unordered_set<UndirectedEdge> get_node_edges(UndirectedGraphView const &,
                                                  Node const &);
std::unordered_set<UndirectedEdge>
    get_node_edges(UndirectedGraphView const &,
                   std::unordered_set<Node> const &);

std::unordered_set<Node> get_predecessors(DiGraphView const &, Node const &);
std::unordered_map<Node, std::unordered_set<Node>>
    get_predecessors(DiGraphView const &, std::unordered_set<Node> const &);

Node get_src_node(MultiDiEdge const &);
Node get_dst_node(MultiDiEdge const &);
Node get_dst_node(InputMultiDiEdge const &);
Node get_src_node(OutputMultiDiEdge const &);

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

NodePort get_src_idx(MultiDiEdge const &);
NodePort get_dst_idx(MultiDiEdge const &);
NodePort get_dst_idx(InputMultiDiEdge const &);
NodePort get_src_idx(OutputMultiDiEdge const &);

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

std::unordered_set<Node> get_neighbors(UndirectedGraphView const &,
                                       Node const &);
std::unordered_set<Node> get_neighbors(DiGraphView const &, Node const &);
std::unordered_set<Node> get_neighbors(MultiDiGraphView const &, Node const &);

// return the set of nodes without incoming edges
std::unordered_set<Node> get_sources(DiGraphView const &);

// return the set of nodes without outgoing edges
std::unordered_set<Node> get_sinks(DiGraphView const &);

std::unordered_set<Node> get_closed_sources(OpenMultiDiGraphView const &g);
std::unordered_set<Node> get_closed_sinks(OpenMultiDiGraphView const &g);
std::unordered_set<Node> get_open_sources(OpenMultiDiGraphView const &g);
std::unordered_set<Node> get_open_sinks(OpenMultiDiGraphView const &g);

bool is_acyclic(MultiDiGraphView const &, std::unordered_set<Node> const &);
std::optional<bool> is_acyclic(DiGraphView const &);
std::optional<bool> is_acyclic(MultiDiGraphView const &);

std::unordered_map<Node, std::unordered_set<Node>>
    get_dominators(DiGraphView const &);
std::unordered_set<Node> get_dominators(DiGraphView const &, Node const &);
std::unordered_set<Node> get_dominators(DiGraphView const &,
                                        std::unordered_set<Node> const &);

std::unordered_map<Node, std::unordered_set<Node>>
    get_post_dominators(DiGraphView const &);
std::unordered_map<Node, std::optional<Node>>
    get_imm_dominators(DiGraphView const &);
std::unordered_map<Node, std::optional<Node>>
    get_imm_post_dominators(DiGraphView const &);
std::optional<Node> get_imm_post_dominator(DiGraphView const &, Node const &);
std::optional<Node> get_imm_post_dominator(MultiDiGraphView const &,
                                           Node const &);
std::optional<Node> get_imm_post_dominator(DiGraphView const &,
                                           std::unordered_set<Node> const &);

std::vector<Node>
    get_dfs_ordering(DiGraphView const &,
                     std::unordered_set<Node> const &starting_points);
std::vector<Node>
    get_unchecked_dfs_ordering(DiGraphView const &,
                               std::unordered_set<Node> const &starting_points);
std::vector<Node>
    get_bfs_ordering(DiGraphView const &,
                     std::unordered_set<Node> const &starting_points);
std::vector<Node> get_topological_ordering(DiGraphView const &);
// std::vector<Node> get_topological_ordering(MultiDiGraphView const &);
// std::vector<Node> get_topological_ordering(OpenMultiDiGraphView const &);
std::vector<Node> get_unchecked_topological_ordering(DiGraphView const &);

std::vector<DirectedEdge> get_edge_topological_ordering(DiGraphView const &);
std::vector<MultiDiEdge>
    get_edge_topological_ordering(MultiDiGraphView const &);

std::unordered_set<std::unordered_set<Node>>
    get_weakly_connected_components(MultiDiGraphView const &);
std::unordered_set<std::unordered_set<Node>>
    get_weakly_connected_components(DiGraphView const &);
std::unordered_set<std::unordered_set<Node>>
    get_connected_components(UndirectedGraphView const &);

std::unordered_set<DirectedEdge>
    get_transitive_reduction_delta(DiGraphView const &);

using GraphSplit =
    std::pair<std::unordered_set<Node>, std::unordered_set<Node>>;

std::pair<OutputMultiDiEdge, InputMultiDiEdge> split_edge(MultiDiEdge const &e);
MultiDiEdge unsplit_edge(OutputMultiDiEdge const &, InputMultiDiEdge const &);

std::unordered_set<MultiDiEdge> get_cut_set(MultiDiGraphView const &,
                                            GraphSplit const &);

std::unordered_set<MultiDiEdge> get_cut_set(MultiDiGraphView const &,
                                            std::unordered_set<Node> const &);

bidict<MultiDiEdge, std::pair<OutputMultiDiEdge, InputMultiDiEdge>>
    get_edge_splits(MultiDiGraphView const &, GraphSplit const &);

UndirectedGraphView get_subgraph(UndirectedGraphView const &,
                                 std::unordered_set<Node> const &);
DiGraphView get_subgraph(DiGraphView const &, std::unordered_set<Node> const &);
MultiDiGraphView get_subgraph(MultiDiGraphView const &,
                              std::unordered_set<Node> const &);

template <typename SubgraphView>
OpenMultiDiGraphView get_subgraph(OpenMultiDiGraphView const &g,
                                  std::unordered_set<Node> const &nodes) {
  return OpenMultiDiGraphView::create<SubgraphView>(g, nodes);
}

std::unordered_map<Node, int> calculate_topo_rank(DiGraphView const &);
Node get_node_with_greatest_topo_rank(std::unordered_set<Node> const &,
                                      DiGraphView const &);

MultiDiGraphView join(MultiDiGraphView const &lhs, MultiDiGraphView const &rhs);
DiGraphView join(DiGraphView const &lhs, DiGraphView const &rhs);
UndirectedGraphView join(UndirectedGraphView const &lhs,
                         UndirectedGraphView const &rhs);

DiGraphView flipped(DiGraphView const &);

DiGraphView with_added_edges(DiGraphView const &,
                             std::unordered_set<DirectedEdge> const &);

UndirectedGraphView as_undirected(DiGraphView const &);
MultiDiGraphView as_multidigraph(DiGraphView const &);
DiGraphView as_digraph(UndirectedGraphView const &);
OpenMultiDiGraphView as_openmultidigraph(MultiDiGraphView const &);

void export_as_dot(
    DotFile<Node> &,
    DiGraphView const &,
    std::function<RecordFormatter(Node const &)> const &,
    std::optional<std::function<std::string(DirectedEdge const &)>> =
        std::nullopt);

} // namespace FlexFlow

#endif

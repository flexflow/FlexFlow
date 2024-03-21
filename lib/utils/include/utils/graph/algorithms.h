/**
 * @file algorithms.h
 * @brief Main Graph library API.
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

/**
 * @brief Add num_nodes Nodes to the graph.
 * 
 * @param g The graph to add Nodes.
 * @param num_nodes The number of Nodes to add.
 * @return std::vector<Node> The vector of added Nodes.
 */
std::vector<Node> add_nodes(Graph &g, int num_nodes);
std::vector<Node> add_nodes(UndirectedGraph &g, int num_nodes);
std::vector<Node> add_nodes(DiGraph &g, int num_nodes);
std::vector<Node> add_nodes(MultiDiGraph &g, int num_nodes);

/**
 * @brief Add num_node_ports NodePorts to the graph.
 * 
 * @param g The graph to add NodePorts.
 * @param num_node_ports The number of NodePorts to add.
 * @return std::vector<NodePort> The vector of added NodePorts.
 */
std::vector<NodePort> add_node_ports(MultiDiGraph &g, int num_node_ports);

/**
 * @brief Get all nodes from the GraphView.
 * 
 * @param g The graph view to query from.
 * @return std::unordered_set<Node> A set of all nodes.
 */
std::unordered_set<Node> get_nodes(GraphView const &g);

/**
 * @brief Get all NodePorts of the MultiDiGraphView.
 * 
 * @param g The graph view to query from.
 * @return std::unordered_set<NodePort> A set of all NodePorts.
 */
std::unordered_set<NodePort> get_present_node_ports(MultiDiGraphView const &g);

/**
 * @brief Get all nodes of the edge.
 * 
 * @param edge The edge to query from.
 * @return std::unordered_set<Node> A set of all nodes.
 */
std::unordered_set<Node> get_nodes(OpenMultiDiEdge const &edge);

/**
 * @brief Query the nodes from a GraphView.
 * 
 * @param g The graph view to query from.
 * @param nodes The nodes to search.
 * @return std::unordered_set<Node> The found nodes from graph view.
 */
std::unordered_set<Node> query_nodes(GraphView const &g,
                                     std::unordered_set<Node> const &nodes);

/**
 * @brief Remove Node n from the Graph g.
 * 
 * @details This function will remove the node and its connected edges.
 * 
 * @param g The graph which we want to remove a node.
 * @param n The node to remove.
 */
void remove_node(MultiDiGraph &g, Node const &n);
void remove_node(DiGraph &g, Node const &n);
void remove_node(UndirectedGraph &g, Node const &n);

/**
 * @brief Remove Node n from the Graph g if the node doesn't have any edge connected.
 * 
 * @param g The graph which we want to remove a node.
 * @param n The node to remove.
 */
void remove_node_if_unused(MultiDiGraph &g, Node const &n);
void remove_node_if_unused(DiGraph &g, Node const &n);
void remove_node_if_unused(UndirectedGraph &g, Node const &n);

void contract_node_inplace(MultiDiGraph &, Node const &from, Node const &into);
void contract_node_inplace(DiGraph &, Node const &from, Node const &into);
void contract_node_inplace(UndirectedGraph &,
                           Node const &from,
                           Node const &into);

void contract_out_node_inplace(MultiDiGraph &, Node const &);
void contract_out_node_inplace(DiGraph &, Node const &);
void contract_out_node_inplace(UndirectedGraph &, Node const &);

MultiDiGraphView contract_out_node(MultiDiGraphView const &, Node const &);
DiGraphView contract_out_node(DiGraphView const &, Node const &);
UndirectedGraphView contract_out_node(UndirectedGraphView const &,
                                      Node const &);

MultiDiGraphView
    contract_node(MultiDiGraphView const &, Node const &from, Node const &into);
DiGraphView
    contract_node(DiGraphView const &, Node const &from, Node const &into);
UndirectedGraphView contract_node(UndirectedGraphView const &,
                                  Node const &from,
                                  Node const &into);

MultiDiGraphView apply_contraction(MultiDiGraphView const &,
                                   std::unordered_map<Node, Node> const &);
DiGraphView apply_contraction(DiGraphView const &,
                              std::unordered_map<Node, Node> const &);
UndirectedGraphView apply_contraction(UndirectedGraphView const &,
                                      std::unordered_map<Node, Node> const &);

/**
 * @brief Get the number of nodes in the graph view.
 * 
 * @param g The GraphView.
 * @return std::size_t The number of nodes in the GraphView.
 */
std::size_t num_nodes(GraphView const &g);

/**
 * @brief Return true if the GraphView is empty.
 */
bool empty(GraphView const &g);

/**
 * @brief Add edges to the graph.
 * 
 * @param g The graph to add edges.
 * @param edges The edges to add to the graph.
 */
void add_edges(MultiDiGraph &g, std::vector<MultiDiEdge> const &edges);
void add_edges(DiGraph &g, std::vector<DirectedEdge> const &edges);
void add_edges(UndirectedGraph &g, std::vector<UndirectedEdge> const &edges);

/**
 * @brief Return true if the GraphView contains the Node.
 */
bool contains_node(GraphView const &g, Node const &n);

/**
 * @brief Return true if the GraphView g contains the Edge e.
 */
bool contains_edge(MultiDiGraphView const &g, MultiDiEdge const &e);
bool contains_edge(DiGraphView const &g, DirectedEdge const &e);
bool contains_edge(UndirectedGraphView const &g, UndirectedEdge const &e);

/**
 * @brief Remove the edges from the graph.
 * 
 * @param g The graph to remove edges from.
 * @param edges The edges to remove.
 */
void remove_edges(MultiDiGraph &g, std::unordered_set<MultiDiEdge> const &edges);
void remove_edges(DiGraph &g, std::unordered_set<DirectedEdge> const &edges);
void remove_edges(UndirectedGraph &g, std::vector<UndirectedEdge> const &edges);

/**
 * @brief Get the two Nodes connected to the UndirectedEdge e.
 * 
 * @param e The UndirectedEdge to query from.
 * @return std::unordered_set<Node> A set of two endpoint Nodes.
 */
std::unordered_set<Node> get_endpoints(UndirectedEdge const &e);

/**
 * @brief Get all edges of the GraphView g.
 * 
 * @param g The GraphView to query from.
 * @return std::unordered_set<MultiDiEdge> All the edges of the graph view.
 */
std::unordered_set<MultiDiEdge> get_edges(MultiDiGraphView const &g);
std::unordered_set<DirectedEdge> get_edges(DiGraphView const &g);
std::unordered_set<UndirectedEdge> get_edges(UndirectedGraphView const &g);
std::unordered_set<UpwardOpenMultiDiEdge>
    get_edges(UpwardOpenMultiDiGraphView const &g);
std::unordered_set<DownwardOpenMultiDiEdge>
    get_edges(DownwardOpenMultiDiGraphView const &g);
std::unordered_set<OpenMultiDiEdge> get_edges(OpenMultiDiGraphView const &g);

/**
 * @brief Get the edges of the node(s) n in GraphView g.
 * 
 * @param g The GraphView to query from.
 * @param n The Node(s) which we want to query connected edges.
 * @return std::unordered_set<UndirectedEdge> A set of edges connected to the node.
 */
std::unordered_set<UndirectedEdge> get_node_edges(UndirectedGraphView const &g,
                                                  Node const &n);
std::unordered_set<UndirectedEdge> get_node_edges(UndirectedGraphView const &g,
                                                  Node const &n);
std::unordered_set<UndirectedEdge>
    get_node_edges(UndirectedGraphView const &g,
                   std::unordered_set<Node> const &n);

/**
 * @brief Get the inputs / outputs of the MultiDiGraphView g.
 */
std::unordered_set<MultiDiInput> get_inputs(MultiDiGraphView const &g);
std::unordered_set<MultiDiOutput> get_outputs(MultiDiGraphView const &g);

/**
 * @brief Get the incoming edges of Node n in the GraphView g.
 * 
 * @return std::unordered_set<MultiDiEdge> 
 */
std::unordered_set<MultiDiEdge> get_incoming_edges(MultiDiGraphView const &g,
                                                   Node const &n);
std::unordered_set<DirectedEdge> get_incoming_edges(DiGraphView const &g,
                                                    Node const &n);
std::unordered_set<UpwardOpenMultiDiEdge>
    get_incoming_edges(UpwardOpenMultiDiGraphView const &g, Node const &n);
std::unordered_set<MultiDiEdge>
    get_incoming_edges(DownwardOpenMultiDiGraphView const &g, Node const &n);
std::unordered_set<UpwardOpenMultiDiEdge>
    get_incoming_edges(OpenMultiDiGraphView const &g, Node const &n);

/**
 * @brief Get the incoming edges of the dsts in the GraphView g.
 * 
 * @param g The GraphView to query from.
 * @param dsts The destination nodes of the incoming edges to query.
 * @return std::unordered_set<MultiDiEdge> A set of edges.
 */
std::unordered_set<MultiDiEdge> get_incoming_edges(MultiDiGraphView const &g,
                                                   std::unordered_set<Node> const &dsts);
std::unordered_set<DirectedEdge>
    get_incoming_edges(DiGraphView const &g, std::unordered_set<Node> const &dsts);

/**
 * @brief Get the NodePort to Edges mapping of the Node n from the GraphView g.
 * 
 * @param g The GraphView to query from.
 * @param n The node whose incoming edges are queried.
 * @return std::unordered_map<NodePort, std::unordered_set<MultiDiEdge>> The mapping from
 * edge destination index to the edges connected to the destination port.
 */
std::unordered_map<NodePort, std::unordered_set<MultiDiEdge>>
    get_incoming_edges_by_idx(MultiDiGraphView const &g, Node const &n);
std::unordered_map<NodePort, std::unordered_set<MultiDiEdge>>
    get_outgoing_edges_by_idx(MultiDiGraphView const &g, Node const &n);

/**
 * @brief Get the outgoing edges of the node(s) n (or srcs) from the GraphView g.
 */
std::unordered_set<MultiDiEdge> get_outgoing_edges(MultiDiGraphView const &g,
                                                   Node const &n);
std::unordered_set<DirectedEdge> get_outgoing_edges(DiGraphView const &g,
                                                    Node const &n);
std::unordered_set<MultiDiEdge>
    get_outgoing_edges(UpwardOpenMultiDiGraphView const &g, Node const &n);
std::unordered_set<DownwardOpenMultiDiEdge>
    get_outgoing_edges(DownwardOpenMultiDiGraphView const &g, Node const &n);
std::unordered_set<DownwardOpenMultiDiEdge>
    get_outgoing_edges(OpenMultiDiGraphView const &g, Node const &n);

std::unordered_set<MultiDiEdge>
    get_outgoing_edges(MultiDiGraphView const &g,
                       std::unordered_set<Node> const &srcs);
std::unordered_set<DirectedEdge>
    get_outgoing_edges(DiGraphView const &g, std::unordered_set<Node> const &srcs);

/**
 * @brief Get the predecessors of the Node(s) in the GraphView g.
 */
std::unordered_set<Node> get_predecessors(DiGraphView const &g, Node const &n);
std::unordered_map<Node, std::unordered_set<Node>>
    get_predecessors(DiGraphView const &g, std::unordered_set<Node> const &nodes);

/**
 * @brief Get the src/dst node of the Edge e.
 */
Node get_src_node(MultiDiEdge const &e);
Node get_dst_node(MultiDiEdge const &e);
Node get_dst_node(InputMultiDiEdge const &e);
Node get_src_node(OutputMultiDiEdge const &e);

// ? Do we need these functors here ?
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

/**
 * @brief Get the src/dst index of the Edge e.
 */
NodePort get_src_idx(MultiDiEdge const &e);
NodePort get_dst_idx(MultiDiEdge const &e);
NodePort get_dst_idx(InputMultiDiEdge const &e);
NodePort get_src_idx(OutputMultiDiEdge const &e);

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

/**
 * @brief Get the neighbor nodes of the Node n in the GraphView g.
 * 
 * @param g The GraphView to query from.
 * @param n The Node whose neighbors are returned.
 * @return std::unordered_set<Node> A set of nodes.
 */
std::unordered_set<Node> get_neighbors(UndirectedGraphView const &g,
                                       Node const &n);
std::unordered_set<Node> get_neighbors(DiGraphView const &g, Node const &n);
std::unordered_set<Node> get_neighbors(MultiDiGraphView const &g, Node const &n);

/**
 * @brief Get the source/sink nodes from GraphView g.
 */
std::unordered_set<Node> get_sources(DiGraphView const &g);
std::unordered_set<Node> get_sinks(DiGraphView const &g);
std::unordered_set<Node> get_closed_sources(OpenMultiDiGraphView const &g);
std::unordered_set<Node> get_closed_sinks(OpenMultiDiGraphView const &g);
std::unordered_set<Node> get_open_sources(OpenMultiDiGraphView const &g);
std::unordered_set<Node> get_open_sinks(OpenMultiDiGraphView const &g);

/**
 * @brief Return true if the GraphView is acyclic.
 */
bool is_acyclic(MultiDiGraphView const &, std::unordered_set<Node> const &);
std::optional<bool> is_acyclic(DiGraphView const &);
std::optional<bool> is_acyclic(MultiDiGraphView const &);

/**
 * @brief Get the dominators.
 */
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

/**
 * @brief Get the dfs/bfs ordering starting from starting_points.
 */
std::vector<Node>
    get_dfs_ordering(DiGraphView const &,
                     std::unordered_set<Node> const &starting_points);
std::vector<Node>
    get_unchecked_dfs_ordering(DiGraphView const &,
                               std::unordered_set<Node> const &starting_points);
std::vector<Node>
    get_bfs_ordering(DiGraphView const &,
                     std::unordered_set<Node> const &starting_points);

/**
 * @brief Get the topological ordering of the GraphView g.
 */
std::vector<Node> get_topological_ordering(DiGraphView const &g);
// std::vector<Node> get_topological_ordering(MultiDiGraphView const &);
// std::vector<Node> get_topological_ordering(OpenMultiDiGraphView const &);
std::vector<Node> get_unchecked_topological_ordering(DiGraphView const &g);

std::vector<DirectedEdge> get_edge_topological_ordering(DiGraphView const &g);
std::vector<MultiDiEdge>
    get_edge_topological_ordering(MultiDiGraphView const &g);

/**
 * @brief Get the weakly connected components of the GraphView g.
 */
std::unordered_set<std::unordered_set<Node>>
    get_weakly_connected_components(MultiDiGraphView const &g);
std::unordered_set<std::unordered_set<Node>>
    get_weakly_connected_components(DiGraphView const &g);
std::unordered_set<std::unordered_set<Node>>
    get_connected_components(UndirectedGraphView const &g);

// ? where is the definition of this function ?
std::unordered_set<DirectedEdge>
    get_transitive_reduction_delta(DiGraphView const &);

using GraphSplit =
    std::pair<std::unordered_set<Node>, std::unordered_set<Node>>;

std::pair<OutputMultiDiEdge, InputMultiDiEdge> split_edge(MultiDiEdge const &e);
MultiDiEdge unsplit_edge(OutputMultiDiEdge const &, InputMultiDiEdge const &);

/**
 * @brief Get the edges splitted by the GraphSplit (two sets of nodes).
 */
std::unordered_set<MultiDiEdge> get_cut_set(MultiDiGraphView const &,
                                            GraphSplit const &);
std::unordered_set<MultiDiEdge> get_cut_set(MultiDiGraphView const &,
                                            std::unordered_set<Node> const &);
bidict<MultiDiEdge, std::pair<OutputMultiDiEdge, InputMultiDiEdge>>
    get_edge_splits(MultiDiGraphView const &, GraphSplit const &);

/**
 * @brief Get the subgraph containing nodes from GraphView g.
 */
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

/**
 * @brief Get the topological order of Nodes in GraphView g.
 * 
 * @param g The GraphView to calculate topological order.
 * @return std::unordered_map<Node, int> A map from a node to its topological order.
 */
std::unordered_map<Node, int> calculate_topo_rank(DiGraphView const &g);

/**
 * @brief Get the node with greatest topological order in nodes.
 * 
 * @param nodes The Nodes to query the greatest topo order node from.
 * @param g The GraphView to analyze.
 * @return Node The node with greatest topo rank in nodes.
 */
Node get_node_with_greatest_topo_rank(std::unordered_set<Node> const &nodes,
                                      DiGraphView const &g);

/**
 * @brief Return the joined graph views.
 */
MultiDiGraphView join(MultiDiGraphView const &lhs, MultiDiGraphView const &rhs);
DiGraphView join(DiGraphView const &lhs, DiGraphView const &rhs);
UndirectedGraphView join(UndirectedGraphView const &lhs,
                         UndirectedGraphView const &rhs);

/**
 * @brief Return the flipped copy of GraphView g.
 */
DiGraphView flipped(DiGraphView const &g);

/**
 * @brief Return a GraphView that is the combination of the given GraphView and DirectedEdges.
 * ? where is the definition of this function ?
 */
DiGraphView with_added_edges(DiGraphView const &g,
                             std::unordered_set<DirectedEdge> const &edges);

/**
 * @brief Convert a graph view type to another graph view type.
 */
UndirectedGraphView as_undirected(DiGraphView const &);
MultiDiGraphView as_multidigraph(DiGraphView const &);
DiGraphView as_digraph(UndirectedGraphView const &);
OpenMultiDiGraphView as_openmultidigraph(MultiDiGraphView const &);

/**
 * @brief Export the GraphView to dot file.
 */
void export_as_dot(
    DotFile<Node> &,
    DiGraphView const &,
    std::function<RecordFormatter(Node const &)> const &,
    std::optional<std::function<std::string(DirectedEdge const &)>> =
        std::nullopt);

} // namespace FlexFlow

#endif

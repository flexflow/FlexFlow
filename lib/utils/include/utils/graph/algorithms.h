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

std::vector<Node> add_nodes(Graph &, int);
std::vector<Node> add_nodes(UndirectedGraph &, int);
std::vector<Node> add_nodes(DiGraph &, int);
std::vector<Node> add_nodes(MultiDiGraph &, int);
std::vector<Node> add_nodes(OpenMultiDiGraph &g, int num_nodes);

std::vector<NodePort> add_node_ports(MultiDiGraph &, int);

std::unordered_set<Node> get_nodes(GraphView const &);
std::unordered_set<NodePort> get_present_node_ports(MultiDiGraphView const &);

std::unordered_set<Node> get_nodes(OpenMultiDiEdge const &);

std::unordered_set<Node> query_nodes(GraphView const &,
                                     std::unordered_set<Node> const &);

void remove_node(MultiDiGraph &, Node const &);
void remove_node(DiGraph &, Node const &);
void remove_node(UndirectedGraph &, Node const &);

void remove_node_if_unused(MultiDiGraph &, Node const &);
void remove_node_if_unused(DiGraph &, Node const &);
void remove_node_if_unused(UndirectedGraph &, Node const &);

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

std::size_t num_nodes(GraphView const &);
bool empty(GraphView const &);

void add_edges(MultiDiGraph &, std::vector<MultiDiEdge> const &);
void add_edges(DiGraph &, std::vector<DirectedEdge> const &);
void add_edges(UndirectedGraph &, std::vector<UndirectedEdge> const &);

bool contains_node(GraphView const &, Node const &);

bool contains_edge(MultiDiGraphView const &, MultiDiEdge const &);
bool contains_edge(DiGraphView const &, DirectedEdge const &);
bool contains_edge(UndirectedGraphView const &, UndirectedEdge const &);

void remove_edges(MultiDiGraph &, std::unordered_set<MultiDiEdge> const &);
void remove_edges(DiGraph &, std::unordered_set<DirectedEdge> const &);
void remove_edges(UndirectedGraph &, std::vector<UndirectedEdge> const &);

std::unordered_set<Node> get_endpoints(UndirectedEdge const &);

std::unordered_set<MultiDiEdge> get_edges(MultiDiGraphView const &);
std::unordered_set<DirectedEdge> get_edges(DiGraphView const &);
std::unordered_set<UndirectedEdge> get_edges(UndirectedGraphView const &);
std::unordered_set<UpwardOpenMultiDiEdge>
    get_edges(UpwardOpenMultiDiGraphView const &);
std::unordered_set<DownwardOpenMultiDiEdge>
    get_edges(DownwardOpenMultiDiGraphView const &);
std::unordered_set<OpenMultiDiEdge> get_edges(OpenMultiDiGraphView const &);

std::unordered_set<UndirectedEdge> get_node_edges(UndirectedGraphView const &,
                                                  Node const &);

std::unordered_set<MultiDiOutput> get_outputs(MultiDiGraphView const &);
std::unordered_set<MultiDiInput> get_inputs(MultiDiGraphView const &);

std::unordered_set<OutputMultiDiEdge>
    get_open_outputs(OpenMultiDiGraphView const &);
std::unordered_set<InputMultiDiEdge>
    get_open_inputs(OpenMultiDiGraphView const &);

std::unordered_set<MultiDiEdge> get_incoming_edges(MultiDiGraphView const &,
                                                   Node const &);
std::unordered_set<DirectedEdge> get_incoming_edges(DiGraphView const &,
                                                    Node const &);
std::unordered_set<UpwardOpenMultiDiEdge>
    get_incoming_edges(UpwardOpenMultiDiGraphView const &, Node const &);
std::unordered_set<MultiDiEdge>
    get_incoming_edges(DownwardOpenMultiDiGraphView const &, Node const &);
std::unordered_set<UpwardOpenMultiDiEdge>
    get_incoming_edges(OpenMultiDiGraphView const &, Node const &);

std::unordered_set<MultiDiEdge> get_incoming_edges(MultiDiGraphView const &,
                                                   std::unordered_set<Node>);
std::unordered_set<DirectedEdge>
    get_incoming_edges(DiGraphView const &, std::unordered_set<Node> const &);

std::unordered_map<NodePort, std::unordered_set<MultiDiEdge>>
    get_incoming_edges_by_idx(MultiDiGraphView const &, Node const &);
std::unordered_map<NodePort, std::unordered_set<MultiDiEdge>>
    get_outgoing_edges_by_idx(MultiDiGraphView const &, Node const &);

std::unordered_set<MultiDiEdge> get_outgoing_edges(MultiDiGraphView const &,
                                                   Node const &);
std::unordered_set<DirectedEdge> get_outgoing_edges(DiGraphView const &,
                                                    Node const &);
std::unordered_set<MultiDiEdge>
    get_outgoing_edges(UpwardOpenMultiDiGraphView const &, Node const &);
std::unordered_set<DownwardOpenMultiDiEdge>
    get_outgoing_edges(DownwardOpenMultiDiGraphView const &, Node const &);
std::unordered_set<DownwardOpenMultiDiEdge>
    get_outgoing_edges(OpenMultiDiGraphView const &, Node const &);

std::unordered_set<MultiDiEdge>
    get_outgoing_edges(MultiDiGraphView const &,
                       std::unordered_set<Node> const &);
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

std::unordered_set<Node> get_successors(DiGraphView const &, Node const &);
std::unordered_map<Node, std::unordered_set<Node>>
    get_successors(DiGraphView const &, std::unordered_set<Node> const &);

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

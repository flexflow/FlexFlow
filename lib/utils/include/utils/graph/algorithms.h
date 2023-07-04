#ifndef _FLEXFLOW_UTILS_GRAPH_ALGORITHMS_H
#define _FLEXFLOW_UTILS_GRAPH_ALGORITHMS_H

#include "digraph.h"
#include "multidigraph.h"
#include "node.h"
#include "open_graphs.h"
#include "tl/optional.hpp"
#include "undirected.h"
#include "utils/containers.h"
#include "utils/dot_file.h"
#include "views.h"
#include <unordered_map>
#include <vector>

namespace FlexFlow {

std::vector<Node> add_nodes(Graph &, int);

std::vector<Node> add_nodes(IGraph &, int);

std::unordered_set<Node> get_nodes(GraphView const &);

std::unordered_set<Node> get_nodes(IMultiDiGraph const &);


std::unordered_set<Node> query_nodes(GraphView const &,
                                     std::unordered_set<Node> const &);

void remove_node(MultiDiGraph &, Node const &);
void remove_node(DiGraph &, Node const &);
void remove_node(UndirectedGraph &, Node const &);

void remove_node_if_unused(MultiDiGraph &, Node const &);
void remove_node_if_unused(DiGraph &, Node const &);
void remove_node_if_unused(UndirectedGraph &, Node const &);

void contract_node(MultiDiGraph &, Node const &);
void contract_node(DiGraph &, Node const &);
void contract_node(UndirectedGraph &, Node const &);

std::size_t num_nodes(GraphView const &);
bool empty(GraphView const &);

void add_edges(MultiDiGraph &, std::vector<MultiDiEdge> const &);
void add_edges(DiGraph &, std::vector<DirectedEdge> const &);
void add_edges(UndirectedGraph &, std::vector<UndirectedEdge> const &);

bool contains_edge(MultiDiGraph const &, MultiDiEdge const &);
bool contains_edge(DiGraph const &, DirectedEdge const &);
bool contains_edge(UndirectedGraph const &, UndirectedEdge const &);

void remove_edges(MultiDiGraph &, std::unordered_set<MultiDiEdge> const &);
void remove_edges(DiGraph &, std::unordered_set<DirectedEdge> const &);
void remove_edges(UndirectedGraph &, std::vector<UndirectedEdge> const &);

std::unordered_set<MultiDiEdge> get_edges(IMultiDiGraphView const &);
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


std::unordered_set<MultiDiEdge> get_incoming_edges(IMultiDiGraphView const &,
                                                   Node const &);
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

std::unordered_set<MultiDiEdge> get_incoming_edges(IMultiDiGraphView const &,
                                                   std::unordered_set<Node> const &);
std::unordered_set<MultiDiEdge> get_incoming_edges(MultiDiGraphView const &,
                                                   std::unordered_set<Node> const &);
std::unordered_set<DirectedEdge>
    get_incoming_edges(DiGraphView const &, std::unordered_set<Node> const &);

std::unordered_map<std::size_t, std::unordered_set<MultiDiEdge>>
    get_incoming_edges_by_idx(MultiDiGraphView const &, Node const &);
std::unordered_map<std::size_t, std::unordered_set<MultiDiEdge>>
    get_outgoing_edges_by_idx(MultiDiGraphView const &, Node const &);

std::unordered_set<MultiDiEdge> get_outgoing_edges(IMultiDiGraphView const &,
                                                   Node const &);
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
    get_outgoing_edges(IMultiDiGraphView const &,
                       std::unordered_set<Node> const &);
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

std::unordered_set<Node> get_predecessors(IMultiDiGraphView const &,
                                          Node const &);
std::unordered_set<Node> get_predecessors(MultiDiGraphView const &,
                                          Node const &);
std::unordered_set<Node> get_predecessors(DiGraphView const &, Node const &);
std::unordered_map<Node, std::unordered_set<Node>>
    get_predecessors(IMultiDiGraphView const &,
                     std::unordered_set<Node> const &);
std::unordered_map<Node, std::unordered_set<Node>>
    get_predecessors(MultiDiGraphView const &,
                     std::unordered_set<Node> const &);
std::unordered_map<Node, std::unordered_set<Node>>
    get_predecessors(DiGraphView const &, std::unordered_set<Node> const &);

std::unordered_set<Node> get_sucessors(MultiDiGraphView const &,
                                          Node const &);
std::unordered_set<Node> get_sucessors(DiGraphView const &, Node const &);
std::unordered_map<Node, std::unordered_set<Node>>
    get_sucessors(MultiDiGraphView const &,
                     std::unordered_set<Node> const &);
std::unordered_map<Node, std::unordered_set<Node>>
    get_sucessors(DiGraphView const &, std::unordered_set<Node> const &);

std::vector<Node> get_neighbors(DiGraphView const &, Node const &);
std::vector<Node> get_neighbors(MultiDiGraphView const &,
                                       Node const &);

std::unordered_set<Node> get_sources(DiGraphView const &);
std::unordered_set<Node> get_sources(MultiDiGraphView const &);

std::unordered_set<Node> get_sinks(DiGraphView const &);
std::unordered_set<Node> get_sinks(MultiDiGraphView const &);

bool is_acyclic(MultiDiGraphView const &, std::unordered_set<Node> const &);
tl::optional<bool> is_acyclic(DiGraphView const &);
tl::optional<bool> is_acyclic(MultiDiGraphView const &);

std::unordered_map<Node, std::unordered_set<Node>>
    get_dominators(MultiDiGraphView const &);
std::unordered_map<Node, std::unordered_set<Node>>
    get_dominators(DiGraphView const &);
std::unordered_set<Node> get_dominators(DiGraphView const &, Node const &);
std::unordered_set<Node> get_dominators(DiGraphView const &,
                                        std::unordered_set<Node> const &);

std::unordered_map<Node, std::unordered_set<Node>>
    get_post_dominators(MultiDiGraphView const &);
std::unordered_map<Node, std::unordered_set<Node>>
    get_post_dominators(DiGraphView const &);
std::unordered_map<Node, tl::optional<Node>>
    get_imm_dominators(MultiDiGraphView const &);
std::unordered_map<Node, tl::optional<Node>>
    get_imm_dominators(DiGraphView const &);
std::unordered_map<Node, tl::optional<Node>>
    get_imm_post_dominators(MultiDiGraphView const &);
std::unordered_map<Node, tl::optional<Node>>
    get_imm_post_dominators(DiGraphView const &);
tl::optional<Node> get_imm_post_dominator(DiGraphView const &, Node const &);
tl::optional<Node> get_imm_post_dominator(MultiDiGraphView const &,
                                          Node const &);
tl::optional<Node> get_imm_post_dominator(DiGraphView const &,
                                          std::unordered_set<Node> const &);

/* std::vector<Node> boundary_dfs_ordering(DiGraphView const &,
 * std::unordered_set<Node> const &starting_points); */
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
std::vector<Node> get_topological_ordering(MultiDiGraphView const &);
std::vector<Node> get_topological_ordering(OpenMultiDiGraphView const &);
std::vector<Node> get_unchecked_topological_ordering(DiGraphView const &);

std::vector<DirectedEdge> get_edge_topological_ordering(DiGraphView const &);
std::vector<MultiDiEdge>
    get_edge_topological_ordering(MultiDiGraphView const &);

std::vector<std::unordered_set<Node>>
    get_weakly_connected_components(MultiDiGraphView const &);
std::vector<std::unordered_set<Node>>
    get_weakly_connected_components(DiGraphView const &);
std::vector<std::unordered_set<Node>>
    get_connected_components(UndirectedGraphView const &);

std::unordered_set<DirectedEdge>
    get_transitive_reduction_delta(DiGraphView const &);

using GraphSplit =
    std::pair<std::unordered_set<Node>, std::unordered_set<Node>>;

std::pair<OutputMultiDiEdge, InputMultiDiEdge> split_edge(MultiDiEdge const &e);
MultiDiEdge unsplit_edge(OutputMultiDiEdge const &, InputMultiDiEdge const &);

UndirectedGraphView get_subgraph(UndirectedGraphView const &g,
                                 std::unordered_set<Node> const &nodes);
DiGraphView get_subgraph(DiGraphView const &g,
                         std::unordered_set<Node> const &nodes);
MultiDiGraphView get_subgraph(MultiDiGraphView const &g,
                              std::unordered_set<Node> const &nodes);
MultiDiGraphView join(MultiDiGraphView const &lhs, MultiDiGraphView const &rhs);
DiGraphView join(DiGraphView const &lhs, DiGraphView const &rhs);
UndirectedGraphView join(UndirectedGraphView const &lhs,
                         UndirectedGraphView const &rhs);

void export_as_dot(
    DotFile<Node> &,
    DiGraphView const &,
    std::function<RecordFormatter(Node const &)> const &,
    tl::optional<std::function<std::string(DirectedEdge const &)> const &> =
        tl::nullopt);

} // namespace FlexFlow

#endif

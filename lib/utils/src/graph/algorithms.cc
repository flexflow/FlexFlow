#include "utils/graph/algorithms.h"
#include "utils/containers.h"
#include "utils/exception.h"
#include "utils/graph/diedge.h"
#include "utils/graph/digraph.h"
#include "utils/graph/multidiedge.h"
#include "utils/graph/multidigraph.h"
#include "utils/graph/multidigraph_interfaces.h"
#include "utils/graph/traversal.h"
#include "utils/graph/undirected.h"
#include "utils/graph/views.h"
#include "utils/variant.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>
#include <unordered_set>

namespace FlexFlow {

template <typename G>
static std::vector<Node> add_nodes_impl(G &g, int num_nodes) {
  std::vector<Node> nodes;
  for (int i = 0; i < num_nodes; i++) {
    nodes.push_back(g.add_node());
  }
  return nodes;
}

std::vector<Node> add_nodes(Graph &g, int num_nodes) {
  return add_nodes_impl<Graph>(g, num_nodes);
}

std::vector<Node> add_nodes(UndirectedGraph &g, int num_nodes) {
  return add_nodes_impl<UndirectedGraph>(g, num_nodes);
}

std::vector<Node> add_nodes(DiGraph &g, int num_nodes) {
  return add_nodes_impl<DiGraph>(g, num_nodes);
}

std::vector<Node> add_nodes(MultiDiGraph &g, int num_nodes) {
  return add_nodes_impl<MultiDiGraph>(g, num_nodes);
}

std::vector<Node> add_nodes(OpenMultiDiGraph &g, int num_nodes) {
  return add_nodes_impl<OpenMultiDiGraph>(g, num_nodes);
}

std::vector<NodePort> add_node_ports(MultiDiGraph &g, int num_node_ports) {
  std::vector<NodePort> node_ports;
  for (int i = 0; i < num_node_ports; i++) {
    node_ports.push_back(g.add_node_port());
  }
  return node_ports;
}

std::unordered_set<Node> get_nodes(GraphView const &g) {
  return g.query_nodes(NodeQuery::all());
}

std::unordered_set<Node> get_nodes(InputMultiDiEdge const &edge) {
  return {edge.dst};
}

std::unordered_set<Node> get_nodes(OutputMultiDiEdge const &edge) {
  return {edge.src};
}

std::unordered_set<Node> get_nodes(MultiDiEdge const &edge) {
  return {edge.src, edge.dst};
}

struct GetNodesFunctor {
  template <typename T>
  std::unordered_set<Node> operator()(T const &t) {
    return get_nodes(t);
  }
};

std::unordered_set<Node> get_nodes(OpenMultiDiEdge const &edge) {
  return visit(GetNodesFunctor{}, edge);
}

std::unordered_set<Node> query_nodes(GraphView const &g,
                                     std::unordered_set<Node> const &nodes) {
  return g.query_nodes({nodes});
}

std::unordered_set<NodePort> get_present_node_ports(MultiDiGraphView const &g) {
  return flatmap(get_edges(g), [](MultiDiEdge const &e) {
    return std::unordered_set<NodePort>{e.src_idx, e.dst_idx};
  });
}

void remove_node(MultiDiGraph &g, Node const &n) {
  for (MultiDiEdge const &e : get_incoming_edges(g, n)) {
    g.remove_edge(e);
  }
  for (MultiDiEdge const &e : get_outgoing_edges(g, n)) {
    g.remove_edge(e);
  }
  g.remove_node_unsafe(n);
}

void remove_node(DiGraph &g, Node const &n) {
  for (DirectedEdge const &e : get_incoming_edges(g, n)) {
    g.remove_edge(e);
  }
  for (DirectedEdge const &e : get_outgoing_edges(g, n)) {
    g.remove_edge(e);
  }
  g.remove_node_unsafe(n);
}

void remove_node(UndirectedGraph &g, Node const &n) {
  for (UndirectedEdge const &e : get_node_edges(g, n)) {
    g.remove_edge(e);
  }
  g.remove_node_unsafe(n);
}

void remove_node_if_unused(MultiDiGraph &g, Node const &n) {
  if (!get_incoming_edges(g, n).empty()) {
    return;
  }
  if (!get_outgoing_edges(g, n).empty()) {
    return;
  }

  g.remove_node_unsafe(n);
}

void remove_node_if_unused(DiGraph &g, Node const &n) {
  if (!get_incoming_edges(g, n).empty()) {
    return;
  }
  if (!get_outgoing_edges(g, n).empty()) {
    return;
  }

  g.remove_node_unsafe(n);
}

void remove_node_if_unused(UndirectedGraph &g, Node const &n) {
  if (!get_node_edges(g, n).empty()) {
    return;
  }

  g.remove_node_unsafe(n);
}

std::size_t num_nodes(GraphView const &g) {
  return get_nodes(g).size();
}

bool empty(GraphView const &g) {
  return num_nodes(g) == 0;
}

DiGraphView
    contract_node(DiGraphView const &g, Node const &from, Node const &into) {
  return DiGraphView::create<ContractNodeView>(g, from, into);
}

DiGraphView apply_contraction(DiGraphView const &g,
                              std::unordered_map<Node, Node> const &nodes) {
  DiGraphView contractedView = g;
  for (auto const &kv : nodes) {
    Node from = kv.first;
    Node into = kv.second;
    if (from != into) {
      contractedView = contract_node(contractedView, from, into);
    }
  }
  return contractedView;
}

void add_edges(MultiDiGraph &g, std::vector<MultiDiEdge> const &edges) {
  for (MultiDiEdge const &e : edges) {
    g.add_edge(e);
  }
}

void add_edges(DiGraph &g, std::vector<DirectedEdge> const &edges) {
  for (DirectedEdge const &e : edges) {
    g.add_edge(e);
  }
}

void add_edges(UndirectedGraph &g, std::vector<UndirectedEdge> const &edges) {
  for (UndirectedEdge const &e : edges) {
    g.add_edge(e);
  }
}

bool contains_edge(MultiDiGraphView const &g, MultiDiEdge const &e) {
  return contains(g.query_edges({e.src, e.dst, e.src_idx, e.dst_idx}), e);
}

bool contains_edge(DiGraphView const &g, DirectedEdge const &e) {
  return contains(g.query_edges({e.src, e.dst}), e);
}

bool contains_edge(UndirectedGraphView const &g, UndirectedEdge const &e) {
  UndirectedEdgeQuery q = {{e.bigger, e.smaller}};
  return contains(g.query_edges(q), e);
}

void remove_edges(MultiDiGraph &g,
                  std::unordered_set<MultiDiEdge> const &edges) {
  for (MultiDiEdge const &e : edges) {
    assert(contains_edge(g, e));
    g.remove_edge(e);
  }
}

void remove_edges(DiGraph &g, std::unordered_set<DirectedEdge> const &edges) {
  for (DirectedEdge const &e : edges) {
    assert(contains_edge(g, e));
    g.remove_edge(e);
  }
}

void remove_edges(UndirectedGraph &g,
                  std::unordered_set<UndirectedEdge> const &edges) {
  for (UndirectedEdge const &e : edges) {
    assert(contains_edge(g, e));
    g.remove_edge(e);
  }
}

std::unordered_set<Node> get_endpoints(UndirectedEdge const &e) {
  return {e.smaller, e.bigger};
}

std::unordered_set<MultiDiEdge> get_edges(MultiDiGraphView const &g) {
  return g.query_edges(MultiDiEdgeQuery::all());
}

std::unordered_set<DirectedEdge> get_edges(DiGraphView const &g) {
  return g.query_edges(DirectedEdgeQuery::all());
}

std::unordered_set<UndirectedEdge> get_edges(UndirectedGraphView const &g) {
  return g.query_edges(UndirectedEdgeQuery::all());
}

std::unordered_set<OpenMultiDiEdge> get_edges(OpenMultiDiGraphView const &g) {
  return g.query_edges(OpenMultiDiEdgeQuery::all());
}

std::unordered_set<UndirectedEdge> get_node_edges(UndirectedGraphView const &g,
                                                  Node const &n) {
  return g.query_edges({n});
}

std::unordered_set<MultiDiOutput> get_outputs(MultiDiGraphView const &g) {
  return transform(get_edges(g), [&](MultiDiEdge const &e) -> MultiDiOutput {
    return static_cast<MultiDiOutput>(e);
  });
}

std::unordered_set<MultiDiInput> get_inputs(MultiDiGraphView const &g) {
  return transform(get_edges(g), [&](MultiDiEdge const &e) -> MultiDiInput {
    return static_cast<MultiDiInput>(e);
  });
}

std::unordered_set<MultiDiEdge> get_incoming_edges(MultiDiGraphView const &g,
                                                   Node const &n) {
  return get_incoming_edges(g, std::unordered_set<Node>{n});
}

std::unordered_set<DirectedEdge> get_incoming_edges(DiGraphView const &g,
                                                    Node const &n) {
  return get_incoming_edges(g, std::unordered_set<Node>{n});
}

std::unordered_set<MultiDiEdge>
    get_incoming_edges(MultiDiGraphView const &g,
                       std::unordered_set<Node> dsts) {
  return g.query_edges(MultiDiEdgeQuery::all().with_dst_nodes(dsts));
}

std::unordered_set<DirectedEdge>
    get_incoming_edges(DiGraphView const &g,
                       std::unordered_set<Node> const &dsts) {
  auto multidigraph_view = as_multidigraph(g);
  return to_directed_edges(get_incoming_edges(multidigraph_view, dsts));
}

std::unordered_set<MultiDiEdge>
    get_outgoing_edges(MultiDiGraphView const &g,
                       std::unordered_set<Node> const &srcs) {
  return g.query_edges(MultiDiEdgeQuery::all().with_src_nodes(srcs));
}

std::unordered_set<MultiDiEdge> get_outgoing_edges(MultiDiGraphView const &g,
                                                   Node const &n) {
  return get_outgoing_edges(g, std::unordered_set<Node>{n});
}

std::unordered_set<DirectedEdge>
    get_outgoing_edges(DiGraphView const &g,
                       std::unordered_set<Node> const &dsts) {
  auto multidigraph_view = as_multidigraph(g);
  return to_directed_edges(get_outgoing_edges(multidigraph_view, dsts));
}

std::unordered_set<DirectedEdge> get_outgoing_edges(DiGraphView const &g,
                                                    Node const &n) {
  return get_outgoing_edges(g, std::unordered_set<Node>{n});
}
std::unordered_map<NodePort, std::unordered_set<MultiDiEdge>>
    get_incoming_edges_by_idx(MultiDiGraphView const &g, Node const &n) {
  std::unordered_set<MultiDiEdge> edges = get_incoming_edges(g, n);
  std::unordered_map<NodePort, std::unordered_set<MultiDiEdge>> result;
  for (MultiDiEdge const &e : edges) {
    result[e.dst_idx].insert(e);
  }
  return result;
}

std::unordered_map<NodePort, std::unordered_set<MultiDiEdge>>
    get_outgoing_edges_by_idx(MultiDiGraphView const &g, Node const &n) {
  std::unordered_set<MultiDiEdge> edges = get_outgoing_edges(g, n);
  std::unordered_map<NodePort, std::unordered_set<MultiDiEdge>> result;
  for (MultiDiEdge const &e : edges) {
    result[e.src_idx].insert(e);
  }
  return result;
}

std::unordered_set<DownwardOpenMultiDiEdge>
    get_outgoing_edges(OpenMultiDiGraphView const &g, Node const &n) {
  return value_all(
      narrow<DownwardOpenMultiDiEdge>(g.query_edges(OpenMultiDiEdgeQuery(
          InputMultiDiEdgeQuery::none(),
          MultiDiEdgeQuery::all().with_src_nodes({n}),
          OutputMultiDiEdgeQuery::all().with_src_nodes({n})))));
}

std::unordered_set<UpwardOpenMultiDiEdge>
    get_incoming_edges(OpenMultiDiGraphView const &g, Node const &n) {
  return value_all(narrow<UpwardOpenMultiDiEdge>(g.query_edges(
      OpenMultiDiEdgeQuery(InputMultiDiEdgeQuery::all().with_dst_nodes({n}),
                           MultiDiEdgeQuery::all().with_dst_nodes({n}),
                           OutputMultiDiEdgeQuery::none()))));
}

std::unordered_set<OutputMultiDiEdge>
    get_open_outputs(OpenMultiDiGraphView const &g) {
  return narrow<OutputMultiDiEdge>(
      g.query_edges(OutputMultiDiEdgeQuery::all()));
}
std::unordered_set<InputMultiDiEdge>
    get_open_inputs(OpenMultiDiGraphView const &g) {
  return narrow<InputMultiDiEdge>(g.query_edges(InputMultiDiEdgeQuery::all()));
}

std::unordered_map<Node, std::unordered_set<Node>>
    get_predecessors(DiGraphView const &g,
                     std::unordered_set<Node> const &nodes) {
  std::unordered_map<Node, std::unordered_set<Node>> predecessors;
  for (Node const &n : nodes) {
    predecessors[n];
  }
  for (DirectedEdge const &e : get_incoming_edges(g, nodes)) {
    predecessors.at(e.dst).insert(e.src);
  }
  return predecessors;
}

std::unordered_set<Node> get_predecessors(DiGraphView const &g, Node const &n) {
  return get_predecessors(g, std::unordered_set<Node>{n}).at(n);
}

std::unordered_map<Node, std::unordered_set<Node>>
    get_successors(DiGraphView const &g,
                   std::unordered_set<Node> const &nodes) {
  return get_predecessors(flipped(g), nodes);
}

std::unordered_set<Node> get_successors(DiGraphView const &g, Node const &n) {
  return get_successors(g, std::unordered_set<Node>{n}).at(n);
}

std::vector<Node> get_unchecked_dfs_ordering(
    DiGraphView const &g, std::unordered_set<Node> const &starting_points) {
  UncheckedDFSView dfs_view = unchecked_dfs(g, starting_points);
  return {dfs_view.begin(), dfs_view.end()};
}

std::vector<Node>
    get_dfs_ordering(DiGraphView const &g,
                     std::unordered_set<Node> const &starting_points) {
  CheckedDFSView dfs_view = dfs(g, starting_points);
  return {dfs_view.begin(), dfs_view.end()};
}

std::vector<Node>
    get_bfs_ordering(DiGraphView const &g,
                     std::unordered_set<Node> const &starting_points) {
  BFSView bfs_view = bfs(g, starting_points);
  return {bfs_view.begin(), bfs_view.end()};
}

std::unordered_set<Node> get_sinks(DiGraphView const &g) {
  return filter(get_nodes(g), [&](Node const &n) {
    return get_outgoing_edges(g, n).size() == 0;
  });
}

DiGraphView flipped(DiGraphView const &g) {
  return DiGraphView::create<FlippedView>(g);
}

std::unordered_set<Node> get_sources(DiGraphView const &g) {
  return filter(get_nodes(g), [&](Node const &n) {
    return get_incoming_edges(g, n).size() == 0;
  });
}

std::optional<bool> is_acyclic(DiGraphView const &g) {
  if (num_nodes(g) == 0) {
    return std::nullopt;
  }
  std::unordered_set<Node> sources = get_sources(g);
  if (sources.size() == 0) {
    return false;
  }
  auto dfs_view = unchecked_dfs(g, sources);
  std::unordered_set<Node> seen;
  for (unchecked_dfs_iterator it = dfs_view.begin(); it != dfs_view.end();
       it++) {
    if (contains(seen, *it)) {
      return false;
    } else {
      seen.insert(*it);
    }
  }
  if (seen != get_nodes(g)) {
    return false;
  }
  return true;
}

std::optional<bool> is_acyclic(MultiDiGraph const &g) {
  return is_acyclic(g);
}

std::vector<Node> get_unchecked_topological_ordering(DiGraphView const &g) {
  auto dfs_view = unchecked_dfs(g, get_sources(g));
  std::vector<Node> order;
  std::unordered_set<Node> seen;
  std::unordered_map<Node, std::unordered_set<Node>> predecessors =
      get_predecessors(g, get_nodes(g));

  auto all_predecessors_seen = [&](Node const &n) -> bool {
    bool result = true;
    for (Node const &pred : predecessors.at(n)) {
      result &= contains(seen, pred);
    }
    return result;
  };

  unchecked_dfs_iterator it = dfs_view.cbegin();
  while (it != dfs_view.cend()) {
    if (all_predecessors_seen(*it)) {
      order.push_back(*it);
      seen.insert(*it);
      it++;
    } else {
      it.skip();
    }
  }

  return order;
}

std::vector<Node> get_topological_ordering(DiGraphView const &g) {
  assert(is_acyclic(g));
  return get_unchecked_topological_ordering(g);
}

std::vector<DirectedEdge> get_edge_topological_ordering(DiGraphView const &g) {
  std::vector<DirectedEdge> result;
  for (Node const &n : get_topological_ordering(g)) {
    for (DirectedEdge const &e : get_outgoing_edges(g, n)) {
      result.push_back(e);
    }
  }

  assert(result.size() == get_edges(g).size());

  return result;
}

Node get_src_node(MultiDiEdge const &e) {
  return e.src;
}

Node get_dst_node(MultiDiEdge const &e) {
  return e.dst;
}

Node get_dst_node(InputMultiDiEdge const &e) {
  return e.dst;
}

Node get_src_node(OutputMultiDiEdge const &e) {
  return e.src;
}

NodePort get_src_idx(MultiDiEdge const &e) {
  return e.src_idx;
}

NodePort get_dst_idx(MultiDiEdge const &e) {
  return e.dst_idx;
}

NodePort get_dst_idx(InputMultiDiEdge const &e) {
  return e.dst_idx;
}

NodePort get_src_idx(OutputMultiDiEdge const &e) {
  return e.src_idx;
}

std::unordered_set<Node> get_neighbors(DiGraphView const &g, Node const &n) {
  UndirectedGraphView undirected = as_undirected(g);
  return get_neighbors(undirected, n);
}

std::unordered_set<Node> get_neighbors(MultiDiGraphView const &g,
                                       Node const &n) {
  UndirectedGraphView undirected = as_undirected(g);
  return get_neighbors(undirected, n);
}

std::unordered_set<Node> get_neighbors(UndirectedGraphView const &g,
                                       Node const &n) {
  return flatmap(get_node_edges(g, n), [&](UndirectedEdge const &edge) {
    return set_difference(get_endpoints(edge), {n});
  });
}

std::vector<MultiDiEdge>
    get_edge_topological_ordering(MultiDiGraphView const &g) {
  std::vector<MultiDiEdge> result;
  for (Node const &n : get_topological_ordering(g)) {
    for (MultiDiEdge const &e : get_outgoing_edges(g, n)) {
      result.push_back(e);
    }
  }

  assert(result.size() == get_edges(g).size());

  return result;
}

std::unordered_map<Node, std::unordered_set<Node>>
    get_dominators(DiGraphView const &g) {
  std::vector<Node> topo = get_topological_ordering(g);
  std::unordered_map<Node, std::unordered_set<Node>> result;

  for (Node const &n : topo) {
    result[n] =
        intersection(transform(get_predecessors(g, n), [&](Node const &n) {
          return result.at(n);
        })).value_or(std::unordered_set<Node>{});
    ;
    result[n].insert(n);
  }

  return result;
}

std::unordered_set<Node> get_dominators(DiGraphView const &g, Node const &n) {
  return get_dominators(g).at(n);
}

std::unordered_set<Node> get_dominators(DiGraphView const &g,
                                        std::unordered_set<Node> const &n) {
  if (n.empty()) {
    throw mk_runtime_error("Cannot find dominators of no nodes");
  }
  std::optional<std::unordered_set<Node>> result =
      intersection(values(restrict_keys(get_dominators(g), n)));
  assert(result.has_value());

  return result.value();
}

std::unordered_map<Node, std::unordered_set<Node>>
    get_post_dominators(DiGraphView const &g) {
  return get_dominators(flipped(g));
}

std::unordered_map<Node, std::optional<Node>>
    get_imm_dominators(DiGraphView const &g) {

  std::unordered_map<Node, std::optional<Node>> result;
  for (auto const &kv : get_dominators(g)) {
    Node node = kv.first;
    std::unordered_set<Node> node_dominators = kv.second;

    assert(node_dominators.size() >= 1);

    // a node cannot immediately dominate itself
    if (node_dominators.size() == 1) {
      assert(get_only(node_dominators) == node);
      result[node] = std::nullopt;
    } else {
      node_dominators.erase(node);
      result[node] = get_node_with_greatest_topo_rank(node_dominators, g);
    }
  }
  return result;
}

std::unordered_map<Node, std::optional<Node>>
    get_imm_post_dominators(DiGraphView const &g) {
  return get_imm_dominators(flipped(g));
}

std::optional<Node> imm_post_dominator(DiGraphView const &g, Node const &n) {
  return get_imm_post_dominators(g).at(n);
}

std::unordered_map<Node, int> calculate_topo_rank(DiGraphView const &g) {
  std::vector<Node> topo_ordering = get_topological_ordering(g);
  std::unordered_map<Node, int> topo_rank;
  for (int i = 0; i < topo_ordering.size(); i++) {
    topo_rank[topo_ordering[i]] = i;
  }
  return topo_rank;
}

Node get_node_with_greatest_topo_rank(std::unordered_set<Node> const &nodes,
                                      DiGraphView const &g) {
  std::unordered_map<Node, int> topo_rank = calculate_topo_rank(g);
  return *std::max_element(nodes.cbegin(),
                           nodes.cend(),
                           [&topo_rank](Node const &lhs, Node const &rhs) {
                             return topo_rank.at(lhs) < topo_rank.at(rhs);
                           });
}

std::optional<Node>
    get_imm_post_dominator(DiGraphView const &g,
                           std::unordered_set<Node> const &nodes) {

  if (nodes.empty()) {
    throw mk_runtime_error("Cannot get imm_post_dominator of no nodes");
  }
  std::unordered_set<Node> commonDoms = assert_unwrap(
      intersection(values(restrict_keys(get_post_dominators(g), nodes))));

  if (!commonDoms.empty()) {
    return get_node_with_greatest_topo_rank(commonDoms, g);
  } else {
    return std::nullopt;
  }
}

std::pair<OutputMultiDiEdge, InputMultiDiEdge>
    split_edge(MultiDiEdge const &e) {
  return {OutputMultiDiEdge{e.src, e.src_idx, e.get_uid()},
          InputMultiDiEdge{e.dst, e.dst_idx, e.get_uid()}};
}

MultiDiEdge unsplit_edge(OutputMultiDiEdge const &output_edge,
                         InputMultiDiEdge const &input_edge) {
  assert(output_edge.uid.first == input_edge.dst.value());
  assert(output_edge.uid.second == input_edge.dst_idx.value());
  assert(input_edge.uid.first == output_edge.src.value());
  assert(input_edge.uid.second == output_edge.src_idx.value());
  return {
      input_edge.dst, input_edge.dst_idx, output_edge.src, output_edge.src_idx};
}

std::unordered_set<MultiDiEdge> get_cut_set(MultiDiGraphView const &g,
                                            GraphSplit const &s) {
  return set_union(
      g.query_edges(
          MultiDiEdgeQuery::all().with_src_nodes(s.first).with_dst_nodes(
              s.second)),
      g.query_edges(
          MultiDiEdgeQuery::all().with_src_nodes(s.second).with_dst_nodes(
              s.first)));
}

std::unordered_set<MultiDiEdge>
    get_cut_set(MultiDiGraphView const &g,
                std::unordered_set<Node> const &nodes) {
  return get_cut_set(g, GraphSplit{nodes, set_difference(get_nodes(g), nodes)});
}

bidict<MultiDiEdge, std::pair<OutputMultiDiEdge, InputMultiDiEdge>>
    get_edge_splits(MultiDiGraphView const &graph, GraphSplit const &split) {
  bidict<MultiDiEdge, std::pair<OutputMultiDiEdge, InputMultiDiEdge>> result;
  std::unordered_set<MultiDiEdge> cut_set = get_cut_set(graph, split);
  for (MultiDiEdge const &edge : cut_set) {
    result.equate(edge, split_edge(edge));
  }
  return result;
  return generate_bidict(get_cut_set(graph, split),
                         [](MultiDiEdge const &e) { return split_edge(e); });
}

UndirectedGraphView get_subgraph(UndirectedGraphView const &g,
                                 std::unordered_set<Node> const &nodes) {
  return UndirectedGraphView::create<UndirectedSubgraphView>(g, nodes);
}

DiGraphView get_subgraph(DiGraphView const &g,
                         std::unordered_set<Node> const &nodes) {
  return DiGraphView::create<DiSubgraphView>(g, nodes);
}

MultiDiGraphView get_subgraph(MultiDiGraphView const &g,
                              std::unordered_set<Node> const &nodes) {
  return MultiDiGraphView::create<MultiDiSubgraphView>(g, nodes);
}

MultiDiGraphView join(MultiDiGraphView const &lhs,
                      MultiDiGraphView const &rhs) {
  return MultiDiGraphView::create<JoinedMultiDigraphView>(lhs, rhs);
}

DiGraphView join(DiGraphView const &lhs, DiGraphView const &rhs) {
  return DiGraphView::create<JoinedDigraphView>(lhs, rhs);
}

UndirectedGraphView join(UndirectedGraphView const &lhs,
                         UndirectedGraphView const &rhs) {
  return UndirectedGraphView::create<JoinedUndirectedGraphView>(lhs, rhs);
}

UndirectedGraphView as_undirected(DiGraphView const &g) {
  return UndirectedGraphView::create<ViewDiGraphAsUndirectedGraph>(g);
}

MultiDiGraphView as_multidigraph(DiGraphView const &g) {
  return MultiDiGraphView::create<ViewDiGraphAsMultiDiGraph>(g);
}

DiGraphView as_digraph(UndirectedGraphView const &g) {
  return DiGraphView::create<ViewUndirectedGraphAsDiGraph>(g);
}

OpenMultiDiGraphView as_openmultidigraph(MultiDiGraphView const &g) {
  return OpenMultiDiGraphView::create<ViewMultiDiGraphAsOpenMultiDiGraph>(g);
}

std::unordered_set<std::unordered_set<Node>>
    get_weakly_connected_components(DiGraphView const &g) {
  return get_connected_components(as_undirected(g));
}

std::unordered_set<std::unordered_set<Node>>
    get_weakly_connected_components(MultiDiGraphView const &g) {
  return get_connected_components(as_undirected(g));
}

std::unordered_set<std::unordered_set<Node>>
    get_connected_components(UndirectedGraphView const &g) {
  std::unordered_set<std::unordered_set<Node>> components;
  std::unordered_set<Node> visited;

  for (Node const &node : get_nodes(g)) {
    std::unordered_set<Node> component =
        without_order(get_bfs_ordering(as_digraph(g), {node}));
    components.insert(component);
    visited = set_union(visited, component);
  }
  return components;
}

std::unordered_set<Node> get_closed_sources(OpenMultiDiGraphView const &g) {
  return filter(get_nodes(g), [&](Node const &n) {
    return get_incoming_edges(g, n).size() == 0;
  });
}

std::unordered_set<Node> get_closed_sinks(OpenMultiDiGraphView const &g) {
  return filter(get_nodes(g), [&](Node const &n) {
    return get_outgoing_edges(g, n).size() == 0;
  });
}

std::unordered_set<Node> get_open_sources(OpenMultiDiGraphView const &g) {
  return filter(get_nodes(g), [&](Node const &n) {
    return !get_incoming_edges(g, n).empty();
  });
}

std::unordered_set<Node> get_open_sinks(OpenMultiDiGraphView const &g) {
  return filter(get_nodes(g), [&](Node const &n) {
    return !get_outgoing_edges(g, n).empty();
  });
}

} // namespace FlexFlow

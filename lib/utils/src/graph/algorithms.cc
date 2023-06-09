#include "utils/graph/algorithms.h"
#include "utils/containers.h"
#include "utils/graph/conversions.h"
#include "utils/graph/traversal.h"
#include "utils/graph/views.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>

namespace FlexFlow {

std::vector<Node> add_nodes(IGraph &g, int num_nodes) {
  std::vector<Node> nodes;
  std::generate_n(
      std::back_inserter(nodes), num_nodes, [&g]() { return g.add_node(); });
  return nodes;
}

std::unordered_set<Node> get_nodes(GraphView const &g) {
  return g.query_nodes({});
}

std::unordered_set<Node> query_nodes(IGraphView const &g,
                                     std::unordered_set<Node> const &nodes) {
  return g.query_nodes({nodes});
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

std::unordered_set<MultiDiEdge> get_edges(MultiDiGraphView const &g) {
  return g.query_edges({});
}

std::unordered_set<DirectedEdge> get_edges(DiGraphView const &g) {
  return g.query_edges({});
}

std::unordered_set<UndirectedEdge> get_edges(UndirectedGraphView const &g) {
  return g.query_edges({tl::nullopt});
}

std::unordered_set<UndirectedEdge> get_node_edges(UndirectedGraphView const &g,
                                                  Node const &n) {
  UndirectedEdgeQuery query(std::unordered_set<Node>{n});
  return g.query_edges(query);
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
  auto multidigraph_view = unsafe_view_as_multidigraph(g);
  return to_directed_edges(get_incoming_edges(multidigraph_view, dsts));
}

std::unordered_set<MultiDiEdge>
    get_outgoing_edges(MultiDiGraphView const &g,
                       std::unordered_set<Node> const &srcs) {
  return g.query_edges(MultiDiEdgeQuery::all().with_src_nodes(srcs));
}

std::unordered_set<DirectedEdge>
    get_outgoing_edges(DiGraphView const &g,
                       std::unordered_set<Node> const &dsts) {
  auto multidigraph_view = unsafe_view_as_multidigraph(g);
  return to_directed_edges(get_outgoing_edges(multidigraph_view, dsts));
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
  return get_predecessors(g, {n});
}

std::unordered_map<Node, std::unordered_set<Node>>
    get_predecessors(MultiDiGraphView const &g,
                     std::unordered_set<Node> const &nodes) {
  return get_predecessors(unsafe_view_as_digraph(g), nodes);
}

std::unordered_set<Node> get_predecessors(MultiDiGraphView const &g,
                                          Node const &n) {
  return get_predecessors(g, {n});
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

/* std::vector<Node> boundary_dfs_ordering(DiGraphView const &g,
 * std::unordered_set<Node> const &starting_points) { */
/*   auto boundary_dfs_view = boundary_dfs(g, starting_points); */
/*   return {boundary_dfs_view.begin(), boundary_dfs_view.end()}; */
/* } */

std::unordered_set<Node> get_sources(DiGraphView const &g) {
  std::unordered_set<Node> sources;
  for (Node const &n : get_nodes(g)) {
    auto incoming = get_incoming_edges(g, n);
    if (incoming.size() == 0) {
      sources.insert(n);
    }
  }
  return sources;
}

tl::optional<bool> is_acyclic(DiGraphView const &g) {
  if (num_nodes(g) == 0) {
    return tl::nullopt;
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
  assert(seen == get_nodes(g));
  return true;
}

tl::optional<bool> is_acyclic(MultiDiGraph const &g) {
  auto digraph_view = unsafe_view_as_digraph(g);
  return is_acyclic(digraph_view);
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

std::vector<Node> get_topological_ordering(MultiDiGraphView const &g) {
  return get_topological_ordering(unsafe_view_as_digraph(g));
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
    get_dominators(MultiDiGraphView const &g) {
  return get_dominators(unsafe_view_as_digraph(g));
}

std::unordered_map<Node, std::unordered_set<Node>>
    get_dominators(DiGraphView const &g) {
  std::vector<Node> topo = get_topological_ordering(g);
  std::unordered_map<Node, std::unordered_set<Node>> result;

  for (Node const &n : topo) {
    for (Node const &pred : get_predecessors(g, n)) {
      if (contains_key(result, n)) {
        result[n] = result.at(pred);
      } else {
        result.at(n) = intersection(result.at(n), result.at(pred));
      }
    }
    result[n].insert(n);
  }

  return result;
}

std::unordered_map<Node, std::unordered_set<Node>>
    get_post_dominators(MultiDiGraphView const &g) {
  return get_post_dominators(unsafe_view_as_digraph(g));
}

std::unordered_map<Node, std::unordered_set<Node>>
    get_post_dominators(DiGraphView const &g) {
  return get_dominators(unsafe_view_as_flipped(g));
}

std::unordered_map<Node, tl::optional<Node>>
    get_imm_dominators(DiGraphView const &g) {
  std::unordered_map<Node, int> topo_rank = [&g]() {
    std::vector<Node> topo_ordering = get_topological_ordering(g);
    std::unordered_map<Node, int> topo_rank;
    for (int i = 0; i < topo_ordering.size(); i++) {
      topo_rank[topo_ordering[i]] = i;
    }
    return topo_rank;
  }();

  auto with_greatest_topo_rank =
      [&topo_rank](std::unordered_set<Node> const &nodes) -> Node {
    return *std::max_element(nodes.cbegin(),
                             nodes.cend(),
                             [&topo_rank](Node const &lhs, Node const &rhs) {
                               return topo_rank.at(lhs) < topo_rank.at(rhs);
                             });
  };

  std::unordered_map<Node, tl::optional<Node>> result;
  for (auto const &kv : get_dominators(g)) {
    Node node = kv.first;
    std::unordered_set<Node> node_dominators = kv.second;

    assert(node_dominators.size() >= 1);

    // a node cannot immediately dominate itself
    if (node_dominators.size() == 1) {
      assert(get_only(node_dominators) == node);
      result[node] = tl::nullopt;
    } else {
      node_dominators.erase(node);
      result[node] = with_greatest_topo_rank(node_dominators);
    }
  }
  return result;
}

std::unordered_map<Node, tl::optional<Node>>
    get_imm_dominators(MultiDiGraphView const &g) {
  return get_imm_dominators(unsafe_view_as_digraph(g));
}

std::unordered_map<Node, tl::optional<Node>>
    get_imm_post_dominators(DiGraphView const &g) {
  return get_imm_dominators(unsafe_view_as_flipped(g));
}

std::unordered_map<Node, tl::optional<Node>>
    get_imm_post_dominators(MultiDiGraphView const &g) {
  return get_imm_post_dominators(unsafe_view_as_digraph(g));
}

tl::optional<Node> imm_post_dominator(DiGraphView const &g, Node const &n) {
  return get_imm_post_dominators(g).at(n);
}

tl::optional<Node> imm_post_dominator(MultiDiGraphView const &g,
                                      Node const &n) {
  return get_imm_post_dominators(g).at(n);
}

std::pair<OutputMultiDiEdge, InputMultiDiEdge>
    split_edge(MultiDiEdge const &e) {
  return {OutputMultiDiEdge{{e.dst.value(), e.dstIdx.value()}, e.src, e.srcIdx},
          InputMultiDiEdge{{e.src.value(), e.srcIdx.value()}, e.dst, e.dstIdx}};
}

MultiDiEdge unsplit_edge(OutputMultiDiEdge const &output_edge,
                         InputMultiDiEdge const &input_edge) {
  assert(output_edge.uid.first == input_edge.dst.value());
  assert(output_edge.uid.second == input_edge.dstIdx.value());
  assert(input_edge.uid.first == output_edge.src.value());
  assert(input_edge.uid.second == output_edge.srcIdx.value());
  return {
      output_edge.src, input_edge.dst, output_edge.srcIdx, input_edge.dstIdx};
}

UndirectedGraphView get_subgraph(UndirectedGraphView const &g,
                                 std::unordered_set<Node> const &nodes) {
  return view_subgraph(g, nodes);
}

DiGraphView get_subgraph(DiGraphView const &g,
                         std::unordered_set<Node> const &nodes) {
  return view_subgraph(g, nodes);
}

MultiDiGraphView get_subgraph(MultiDiGraphView const &g,
                              std::unordered_set<Node> const &nodes) {
  return view_subgraph(g, nodes);
}

MultiDiGraphView join(MultiDiGraphView const &lhs,
                      MultiDiGraphView const &rhs) {
  return view_as_joined(lhs, rhs);
}

DiGraphView join(DiGraphView const &lhs, DiGraphView const &rhs) {
  return view_as_joined(lhs, rhs);
}

UndirectedGraphView join(UndirectedGraphView const &lhs,
                         UndirectedGraphView const &rhs) {
  return view_as_joined(lhs, rhs);
}

} // namespace FlexFlow

#include "utils/graph/algorithms.h"
#include "utils/graph/conversions.h"
#include <queue>
#include <algorithm>
#include <iostream>
#include "utils/graph/traversal.h"
#include "utils/containers.h"
#include <cassert>
#include "utils/graph/views.h"

namespace FlexFlow {

std::vector<Node> add_nodes(IGraph &g, int num_nodes) {
  std::vector<Node> nodes;
  std::generate_n(std::back_inserter(nodes), num_nodes, [&g]() { return g.add_node(); });
  return nodes;
}

std::unordered_set<Node> get_nodes(IGraphView const &g) {
  return g.query_nodes({});
}

std::unordered_set<Node> query_nodes(IGraphView const &g, std::unordered_set<Node> const &nodes) {
  return g.query_nodes({nodes});
}

void remove_node(IMultiDiGraph &g, Node const &n) {
  for (MultiDiEdge const &e : get_incoming_edges(g, n)) {
    g.remove_edge(e);
  }
  for (MultiDiEdge const &e : get_outgoing_edges(g, n)) {
    g.remove_edge(e);
  }
  g.remove_node_unsafe(n);
   
}

void remove_node(IDiGraph &g, Node const &n) {
  for (DirectedEdge const &e : get_incoming_edges(g, n)) {
    g.remove_edge(e);
  }
  for (DirectedEdge const &e : get_outgoing_edges(g, n)) {
    g.remove_edge(e);
  }
  g.remove_node_unsafe(n);
}

void remove_node(IUndirectedGraph &g, Node const &n) {
  for (UndirectedEdge const &e : get_node_edges(g, n)) {
    g.remove_edge(e);
  }
  g.remove_node_unsafe(n);
}

void remove_node_if_unused(IMultiDiGraph &g, Node const &n) {
  if (!get_incoming_edges(g, n).empty()) {
    return;
  }
  if (!get_outgoing_edges(g, n).empty()) {
    return;
  }

  g.remove_node_unsafe(n);
}

void remove_node_if_unused(IDiGraph &g, Node const &n) {
  if (!get_incoming_edges(g, n).empty()) {
    return;
  }
  if (!get_outgoing_edges(g, n).empty()) {
    return;
  }

  g.remove_node_unsafe(n);
}

void remove_node_if_unused(IUndirectedGraph &g, Node const &n) {
  if (!get_node_edges(g, n).empty()) {
    return;
  }

  g.remove_node_unsafe(n);
}

std::size_t num_nodes(IGraphView const &g) {
  return get_nodes(g).size();
}

bool empty(IGraphView const &g) {
  return num_nodes(g) == 0;
}

void add_edges(IMultiDiGraph &g, std::vector<MultiDiEdge> const &edges) {
  for (MultiDiEdge const &e : edges) {
    g.add_edge(e);
  }
}

void add_edges(IDiGraph &g, std::vector<DirectedEdge> const &edges) {
  for (DirectedEdge const &e : edges) {
    g.add_edge(e);
  }
}

void add_edges(IUndirectedGraph &g, std::vector<UndirectedEdge> const &edges) {
  for (UndirectedEdge const &e : edges) {
    g.add_edge(e);
  }
}

void remove_edges(IMultiDiGraph &g, std::unordered_set<MultiDiEdge> const &edges) {
  for (MultiDiEdge const &e : edges) {
    assert (contains_edge(g, e));
    g.remove_edge(e);
  }
}

void remove_edges(IDiGraph &g, std::unordered_set<DirectedEdge> const &edges) {
  for (DirectedEdge const &e : edges) {
    assert (contains_edge(g, e));
    g.remove_edge(e);
  }
}

void remove_edges(IUndirectedGraph &g, std::unordered_set<UndirectedEdge> const &edges) {
  for (UndirectedEdge const &e : edges) {
    assert (contains_edge(g, e));
    g.remove_edge(e);
  }
}

std::unordered_set<MultiDiEdge> get_edges(IMultiDiGraphView const &g) {
  return g.query_edges({});
}

std::unordered_set<DirectedEdge> get_edges(IDiGraphView const &g) {
  return g.query_edges({});
}

std::unordered_set<UndirectedEdge> get_edges(IUndirectedGraphView const &g) {
  return g.query_edges({tl::nullopt});
}

std::unordered_set<UndirectedEdge> get_node_edges(IUndirectedGraphView const &g, Node const &n) {
  UndirectedEdgeQuery query(std::unordered_set<Node>{n});
  return g.query_edges(query);
}

std::unordered_set<MultiDiEdge> get_incoming_edges(IMultiDiGraphView const &g, Node const &n) {
  return get_incoming_edges(g, std::unordered_set<Node>{n});
}

std::unordered_set<DirectedEdge> get_incoming_edges(IDiGraphView const &g, Node const &n) {
  return get_incoming_edges(g, std::unordered_set<Node>{n});
}

std::unordered_set<MultiDiEdge> get_incoming_edges(IMultiDiGraphView const &g, std::unordered_set<Node> dsts) {
  return g.query_edges(MultiDiEdgeQuery::all().with_dst_nodes(dsts));
}

std::unordered_set<DirectedEdge> get_incoming_edges(IDiGraphView const &g, std::unordered_set<Node> const &dsts) {
  auto multidigraph_view = unsafe_view_as_multidigraph(g);
  return to_directed_edges(get_incoming_edges(*multidigraph_view, dsts));
}

std::unordered_set<MultiDiEdge> get_outgoing_edges(IMultiDiGraphView const &g, std::unordered_set<Node> const &srcs) {
  return g.query_edges(MultiDiEdgeQuery::all().with_src_nodes(srcs));
}

std::unordered_set<DirectedEdge> get_outgoing_edges(IDiGraphView const &g, std::unordered_set<Node> const &dsts) {
  auto multidigraph_view = unsafe_view_as_multidigraph(g);
  return to_directed_edges(get_outgoing_edges(*multidigraph_view, dsts));
}

std::unordered_map<Node, std::unordered_set<Node>> get_predecessors(IDiGraphView const &g, std::unordered_set<Node> const &nodes) {
  std::unordered_map<Node, std::unordered_set<Node>> predecessors;
  for (Node const &n : nodes) {
    predecessors[n];
  }
  for (DirectedEdge const &e : get_incoming_edges(g, nodes)) {
    predecessors.at(e.dst).insert(e.src);
  }
  return predecessors;
}

std::unordered_set<Node> get_predecessors(IDiGraphView const &g, Node const &n) {
  return get_predecessors(g, {n});
}

std::unordered_map<Node, std::unordered_set<Node>> get_predecessors(IMultiDiGraphView const &g, std::unordered_set<Node> const &nodes) {
  return get_predecessors(*unsafe_view_as_digraph(g), nodes);
}

std::unordered_set<Node> get_predecessors(IMultiDiGraphView const &g, Node const &n) {
  return get_predecessors(g, {n});
}


std::vector<Node> get_unchecked_dfs_ordering(IDiGraphView const &g, std::unordered_set<Node> const &starting_points) {
  UncheckedDFSView dfs_view = unchecked_dfs(g, starting_points);
  return {dfs_view.begin(), dfs_view.end()};
}

std::vector<Node> get_dfs_ordering(IDiGraphView const &g, std::unordered_set<Node> const &starting_points) {
  CheckedDFSView dfs_view = dfs(g, starting_points);
  return {dfs_view.begin(), dfs_view.end()};
}

std::vector<Node> get_bfs_ordering(IDiGraphView const &g, std::unordered_set<Node> const &starting_points) {
  BFSView bfs_view = bfs(g, starting_points);
  return {bfs_view.begin(), bfs_view.end()};
}

/* std::vector<Node> boundary_dfs_ordering(IDiGraphView const &g, std::unordered_set<Node> const &starting_points) { */
/*   auto boundary_dfs_view = boundary_dfs(g, starting_points); */
/*   return {boundary_dfs_view.begin(), boundary_dfs_view.end()}; */
/* } */

std::unordered_set<Node> get_sources(IDiGraphView const &g) {
  std::unordered_set<Node> sources;
  for (Node const &n : get_nodes(g)) {
    auto incoming = get_incoming_edges(g, n);
    if (incoming.size() == 0) {
      sources.insert(n);
    }
  }
  return sources;
}

tl::optional<bool> is_acyclic(IDiGraphView const &g) {
  if (num_nodes(g) == 0) {
    return tl::nullopt;
  }
  std::unordered_set<Node> sources = get_sources(g);
  if (sources.size() == 0) {
    return false;
  }
  auto dfs_view = unchecked_dfs(g, sources);
  std::unordered_set<Node> seen;
  for (unchecked_dfs_iterator it = dfs_view.begin(); it != dfs_view.end(); it++) {
    if (contains(seen, *it)) {
      return false;
    } else {
      seen.insert(*it);
    }
  }
  assert (seen == get_nodes(g));
  return true;
}

tl::optional<bool> is_acyclic(IMultiDiGraph const &g) {
  auto digraph_view = unsafe_view_as_digraph(g);
  return is_acyclic(*digraph_view);
}

std::vector<Node> get_unchecked_topological_ordering(IDiGraphView const &g) {
  auto dfs_view = unchecked_dfs(g, get_sources(g));
  std::vector<Node> order;
  std::unordered_set<Node> seen;
  std::unordered_map<Node, std::unordered_set<Node>> predecessors = get_predecessors(g, get_nodes(g));

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

std::vector<Node> get_topological_ordering(IDiGraphView const &g) {
  assert (is_acyclic(g));
  return get_unchecked_topological_ordering(g);
}

std::vector<Node> get_topological_ordering(IMultiDiGraphView const &g) {
  return get_topological_ordering(*unsafe_view_as_digraph(g));
}

std::vector<DirectedEdge> get_edge_topological_ordering(IDiGraphView const &g) {
  std::vector<DirectedEdge> result;
  for (Node const &n : get_topological_ordering(g)) {
    for (DirectedEdge const &e : get_outgoing_edges(g, n)) {
      result.push_back(e);
    }
  }

  assert (result.size() == get_edges(g).size());

  return result;
}

std::vector<MultiDiEdge> get_edge_topological_ordering(IMultiDiGraphView const &g) {
  std::vector<MultiDiEdge> result;
  for (Node const &n : get_topological_ordering(g)) {
    for (MultiDiEdge const &e : get_outgoing_edges(g, n)) {
      result.push_back(e);
    }
  }

  assert (result.size() == get_edges(g).size());

  return result;
}

std::unordered_map<Node, std::unordered_set<Node>> get_dominators(IMultiDiGraphView const &g) {
  return get_dominators(*unsafe_view_as_digraph(g));
}

std::unordered_map<Node, std::unordered_set<Node>> get_dominators(IDiGraphView const &g) {
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

std::unordered_set<Node> get_dominators(IDiGraphView const &, Node const &) {
  // TODO
  std::unordered_set<Node> result;
  return result;
}

std::unordered_set<Node> get_dominators(IDiGraphView const &, std::unordered_set<Node> const &) {
  // TODO
  std::unordered_set<Node> result;
  return result;
}

std::unordered_map<Node, std::unordered_set<Node>> get_post_dominators(IMultiDiGraphView const &g) {
  return get_post_dominators(*unsafe_view_as_digraph(g));
}

std::unordered_map<Node, std::unordered_set<Node>> get_post_dominators(IDiGraphView const &g) {
  return get_dominators(*unsafe_view_as_flipped(g));
}

std::unordered_map<Node, tl::optional<Node>> get_imm_dominators(IDiGraphView const &g) {
  std::unordered_map<Node, int> topo_rank = [&g]() {
    std::vector<Node> topo_ordering = get_topological_ordering(g);
    std::unordered_map<Node, int> topo_rank;
    for (int i = 0; i < topo_ordering.size(); i++) {
      topo_rank[topo_ordering[i]] = i;
    }
    return topo_rank;
  }();

  auto with_greatest_topo_rank = [&topo_rank](std::unordered_set<Node> const &nodes) -> Node {
    return *std::max_element(nodes.cbegin(), nodes.cend(), [&topo_rank](Node const &lhs, Node const &rhs) {
      return topo_rank.at(lhs) < topo_rank.at(rhs);
    });
  };

  std::unordered_map<Node, tl::optional<Node>> result;
  for (auto const &kv : get_dominators(g)) {
    Node node = kv.first;
    std::unordered_set<Node> node_dominators = kv.second;

    assert (node_dominators.size() >= 1);
    
    // a node cannot immediately dominate itself
    if (node_dominators.size() == 1) {
      assert (get_only(node_dominators) == node);
      result[node] = tl::nullopt;
    } else {
      node_dominators.erase(node);
      result[node] = with_greatest_topo_rank(node_dominators);
    }
  }
  return result;
}


std::unordered_map<Node, tl::optional<Node>> get_imm_dominators(IMultiDiGraphView const &g) {
  return get_imm_dominators(*unsafe_view_as_digraph(g));
}

std::unordered_map<Node, tl::optional<Node>> get_imm_post_dominators(IDiGraphView const &g) {
  return get_imm_dominators(*unsafe_view_as_flipped(g));
}

std::unordered_map<Node, tl::optional<Node>> get_imm_post_dominators(IMultiDiGraphView const &g) {
  return get_imm_post_dominators(*unsafe_view_as_digraph(g));
}

tl::optional<Node> imm_post_dominator(IDiGraphView const &g, Node const &n) {
  return get_imm_post_dominators(g).at(n);
}

tl::optional<Node> imm_post_dominator(IMultiDiGraphView const &g, Node const &n) {
  return get_imm_post_dominators(g).at(n);
}

std::pair<OutputMultiDiEdge, InputMultiDiEdge> split_edge(MultiDiEdge const &e) {
  return { OutputMultiDiEdge{{e.dst.idx, e.dstIdx}, e.src, e.srcIdx}, InputMultiDiEdge{{e.src.idx, e.srcIdx}, e.dst, e.dstIdx} };
}

MultiDiEdge unsplit_edge(OutputMultiDiEdge const &output_edge, InputMultiDiEdge const &input_edge) {
  assert (output_edge.uid.first == input_edge.dst.idx);
  assert (output_edge.uid.second == input_edge.dstIdx);
  assert (input_edge.uid.first == output_edge.src.idx);
  assert (input_edge.uid.second == output_edge.srcIdx);
  return { output_edge.src, input_edge.dst, output_edge.srcIdx, input_edge.dstIdx };
}

bool contains_edge(IMultiDiGraph const &, MultiDiEdge const &) {
  // TODO
  return false;
}

bool contains_edge(IDiGraph const &, DirectedEdge const &) {
  // TODO
  return false;
}

bool contains_edge(IUndirectedGraph const &, UndirectedEdge const &) {
  // TODO
  return false;
}

}

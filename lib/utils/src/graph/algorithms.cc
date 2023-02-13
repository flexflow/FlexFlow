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
namespace utils {

std::vector<Node> add_nodes(IGraph &g, int num_nodes) {
  std::vector<Node> nodes;
  std::generate_n(std::back_inserter(nodes), num_nodes, [&g]() { return g.add_node(); });
  return nodes;
}

std::unordered_set<Node> get_nodes(IGraphView const &g) {
  return g.query_nodes({});
}

std::size_t num_nodes(IGraphView const &g) {
  return get_nodes(g).size();
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

std::unordered_set<MultiDiEdge> get_edges(IMultiDiGraphView const &g) {
  return g.query_edges({});
}

std::unordered_set<DirectedEdge> get_edges(IDiGraphView const &g) {
  return g.query_edges({});
}

std::unordered_set<UndirectedEdge> get_edges(IUndirectedGraphView const &g) {
  return g.query_edges({});
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
  return to_directed_edges(get_incoming_edges(multidigraph_view, dsts));
}

std::unordered_set<MultiDiEdge> get_outgoing_edges(IMultiDiGraphView const &g, std::unordered_set<Node> const &srcs) {
  return g.query_edges(MultiDiEdgeQuery::all().with_src_nodes(srcs));
}

std::unordered_set<DirectedEdge> get_outgoing_edges(IDiGraphView const &g, std::unordered_set<Node> const &dsts) {
  auto multidigraph_view = unsafe_view_as_multidigraph(g);
  return to_directed_edges(get_outgoing_edges(multidigraph_view, dsts));
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
  return get_predecessors(unsafe_view_as_digraph(g), nodes);
}

std::unordered_set<Node> get_predecessors(IMultiDiGraphView const &g, Node const &n) {
  return get_predecessors(g, {n});
}


std::vector<Node> unchecked_dfs_ordering(IDiGraphView const &g, std::unordered_set<Node> const &starting_points) {
  UncheckedDFSView dfs_view = unchecked_dfs(g, starting_points);
  return {dfs_view.begin(), dfs_view.end()};
}

std::vector<Node> dfs_ordering(IDiGraphView const &g, std::unordered_set<Node> const &starting_points) {
  CheckedDFSView dfs_view = dfs(g, starting_points);
  return {dfs_view.begin(), dfs_view.end()};
}

std::vector<Node> bfs_ordering(IDiGraphView const &g, std::unordered_set<Node> const &starting_points) {
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
  return is_acyclic(digraph_view);
}

std::vector<Node> unchecked_topological_ordering(IDiGraphView const &g) {
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

std::vector<Node> topological_ordering(IDiGraphView const &g) {
  assert (is_acyclic(g));
  return unchecked_topological_ordering(g);
}

std::vector<Node> topological_ordering(IMultiDiGraphView const &g) {
  return topological_ordering(unsafe_view_as_digraph(g));
}

std::unordered_map<Node, std::unordered_set<Node>> dominators(IDiGraphView const &g) {
  std::vector<Node> topo = topological_ordering(g);
  std::unordered_map<Node, std::unordered_set<Node>> result;

  for (Node const &n : topo) {
    for (Node const &pred : get_predecessors(g, n)) {
      if (contains(result, n)) {
        result[n] = result.at(pred);
      } else {
        result.at(n) = intersection(result.at(n), result.at(pred));
      }
    }
    result[n].insert(n);
  }

  return result;
}

std::unordered_map<Node, std::unordered_set<Node>> post_dominators(IDiGraphView const &g) {
  return dominators(unsafe_view_as_flipped(g));
}


}
}

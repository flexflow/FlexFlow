#include "utils/graph/algorithms.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/get_only.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/set_difference.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/graph/digraph/algorithms/get_incoming_edges.h"
#include "utils/graph/digraph/algorithms/get_node_with_greatest_topo_rank.h"
#include "utils/graph/digraph/algorithms/get_outgoing_edges.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/digraph/directed_edge_query.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node_query.h"
#include "utils/graph/traversal.h"
#include "utils/graph/undirected/undirected_edge_query.h"
#include "utils/graph/views/views.h"
#include "utils/hash-utils.h"
#include "utils/variant.h"
#include <algorithm>
#include <iostream>
#include <libassert/assert.hpp>
#include <queue>

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

struct GetNodesFunctor {
  template <typename T>
  std::set<Node> operator()(T const &t) {
    return get_nodes(t);
  }
};

std::set<Node> query_nodes(GraphView const &g, std::set<Node> const &nodes) {
  NodeQuery query = NodeQuery{
      query_set<Node>::match_values_in(set_of(nodes)),
  };

  return g.query_nodes(query);
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

void add_edges(DiGraph &g, std::initializer_list<DirectedEdge> edges) {
  add_edges(g, std::vector<DirectedEdge>{edges});
}

void add_edges(UndirectedGraph &g,
               std::initializer_list<UndirectedEdge> edges) {
  add_edges(g, std::vector<UndirectedEdge>{edges});
}

void add_edges(DiGraph &g, std::set<DirectedEdge> const &edges) {
  for (DirectedEdge const &e : edges) {
    g.add_edge(e);
  }
}

void add_edges(UndirectedGraph &g, std::set<UndirectedEdge> const &edges) {
  for (UndirectedEdge const &e : edges) {
    g.add_edge(e);
  }
}

bool contains_edge(DiGraphView const &g, DirectedEdge const &e) {
  DirectedEdgeQuery query = DirectedEdgeQuery{
      query_set<Node>::match_single_value(e.src),
      query_set<Node>::match_single_value(e.dst),
  };

  return contains(g.query_edges(query), e);
}

bool contains_edge(UndirectedGraphView const &g, UndirectedEdge const &e) {
  UndirectedEdgeQuery q = UndirectedEdgeQuery{
      query_set<Node>::match_values_in({e.endpoints.max(), e.endpoints.min()})};
  return contains(g.query_edges(q), e);
}

void remove_edges(DiGraph &g, std::set<DirectedEdge> const &edges) {
  for (DirectedEdge const &e : edges) {
    ASSERT(contains_edge(g, e),
           "remove_edges expected edge to exist in DiGraph");
    g.remove_edge(e);
  }
}

void remove_edges(UndirectedGraph &g, std::set<UndirectedEdge> const &edges) {
  for (UndirectedEdge const &e : edges) {
    ASSERT(contains_edge(g, e),
           "remove_edges expected edge to exist in UndirectedGraph");
    g.remove_edge(e);
  }
}

std::set<UndirectedEdge> get_node_edges(UndirectedGraphView const &g,
                                        Node const &n) {
  UndirectedEdgeQuery query = UndirectedEdgeQuery{
      query_set<Node>::match_single_value(n),
  };

  return g.query_edges(query);
}

std::vector<Node>
    get_unchecked_dfs_ordering(DiGraphView const &g,
                               std::set<Node> const &starting_points) {
  UncheckedDFSView dfs_view = unchecked_dfs(g, starting_points);
  return {dfs_view.begin(), dfs_view.end()};
}

std::vector<Node> get_dfs_ordering(DiGraphView const &g,
                                   std::set<Node> const &starting_points) {
  CheckedDFSView dfs_view = dfs(g, starting_points);
  return {dfs_view.begin(), dfs_view.end()};
}

std::vector<Node> get_bfs_ordering(DiGraphView const &g,
                                   std::set<Node> const &starting_points) {
  BFSView bfs_view = bfs(g, starting_points);
  return {bfs_view.begin(), bfs_view.end()};
}

std::set<Node> get_neighbors(DiGraphView const &g, Node const &n) {
  UndirectedGraphView undirected = as_undirected(g);
  return get_neighbors(undirected, n);
}

std::set<Node> get_neighbors(UndirectedGraphView const &g, Node const &n) {
  return flatmap(get_node_edges(g, n), [&](UndirectedEdge const &edge) {
    return set_difference(get_endpoints(edge), {n});
  });
}

UndirectedGraphView get_subgraph(UndirectedGraphView const &g,
                                 std::set<Node> const &nodes) {
  return UndirectedGraphView::create<UndirectedSubgraphView>(g, nodes);
}

DiGraphView get_subgraph(DiGraphView const &g, std::set<Node> const &nodes) {
  return DiGraphView::create<DiSubgraphView>(g, nodes);
}

UndirectedGraphView as_undirected(DiGraphView const &g) {
  return UndirectedGraphView::create<ViewDiGraphAsUndirectedGraph>(g);
}

DiGraphView as_digraph(UndirectedGraphView const &g) {
  return DiGraphView::create<ViewUndirectedGraphAsDiGraph>(g);
}

} // namespace FlexFlow

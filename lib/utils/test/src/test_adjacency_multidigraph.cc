#include "doctest.h"
#include "utils/graph/adjacency_multidigraph.h"

using namespace FlexFlow::utils::graph::multidigraph;

TEST_CASE("AdjacencyMultiDiGraph:basic_test") {
  AdjacencyMultiDiGraph g;
  Node n1 = g.add_node();
  Node n2 = g.add_node();
  Node n3 = g.add_node();
  Edge e1 {n1, n2, 0, 0};
  Edge e2 {n1, n3, 0, 1};
  Edge e3 {n3, n1, 1, 1};
  Edge e4 {n3, n2, 1, 1};
  Edge e5 {n2, n2, 2, 2};
  g.add_edge(e1);
  g.add_edge(e2);
  g.add_edge(e3);
  g.add_edge(e4);
  g.add_edge(e5);

  CHECK(g.query_nodes({}) == std::unordered_set<Node>{n1, n2, n3});

  CHECK(g.query_nodes(NodeQuery{{n1, n3}}) == std::unordered_set<Node>{n1, n3});

  CHECK(g.query_edges({}) == std::unordered_set<Edge>{e1, e2, e3, e4, e5});

  CHECK(g.query_edges(EdgeQuery::all().with_src_node(n1)) == std::unordered_set<Edge>{e1, e2});
  CHECK(g.query_edges(EdgeQuery::all().with_dst_node(n1)) == std::unordered_set<Edge>{e3});
  CHECK(g.query_edges(EdgeQuery::all().with_src_idx(1)) == std::unordered_set<Edge>{e3, e4});
  CHECK(g.query_edges(EdgeQuery::all().with_dst_idx(1)) == std::unordered_set<Edge>{e2, e3, e4});
  CHECK(g.query_edges(EdgeQuery::all().with_src_nodes({n1, n2})) == std::unordered_set<Edge>{e1, e2, e5});
  CHECK(g.query_edges(EdgeQuery::all().with_dst_nodes({n1, n3})) == std::unordered_set<Edge>{e2, e3});
  CHECK(g.query_edges(EdgeQuery::all().with_src_idxs({1, 2})) == std::unordered_set<Edge>{e3, e4, e5});
  CHECK(g.query_edges(EdgeQuery::all().with_dst_idxs({0, 2})) == std::unordered_set<Edge>{e1, e5});
  CHECK(g.query_edges(EdgeQuery::all().with_src_node(n1).with_dst_node(n3).with_src_idx(0).with_dst_idx(1)) == std::unordered_set<Edge>{e2});
  CHECK(g.query_edges(EdgeQuery::all().with_dst_idx(3)) == std::unordered_set<Edge>{});
}

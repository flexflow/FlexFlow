#include "doctest.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/adjacency_multidigraph.h"
#include "utils/graph/adjacency_digraph.h"
#include "utils/graph/construction.h"

using namespace FlexFlow::utils::graph;

TEST_CASE("MultiDiGraph") {
  using namespace FlexFlow::utils::graph::multidigraph;

  AdjacencyMultiDiGraph g;
  Node n1 = g.add_node();
  Node n2 = g.add_node();
  Node n3 = g.add_node();
  Node n4 = g.add_node();
  Edge e1 {n1, n4, 0, 0};
  Edge e2 {n1, n2, 0, 1};
  Edge e3 {n1, n3, 0, 0};
  Edge e4 {n2, n3, 0, 0};
  g.add_edge(e1);
  g.add_edge(e2);
  g.add_edge(e3);
  g.add_edge(e4);

  CHECK(get_nodes(g) == std::unordered_set<Node>{n1, n2, n3, n4});
  CHECK(get_edges(g) == std::unordered_set<Edge>{e1, e2, e3, e4});
  CHECK(get_incoming_edges(g, {n2, n4}) == std::unordered_set<Edge>{e1, e2});
  CHECK(get_outgoing_edges(g, {n2, n4}) == std::unordered_set<Edge>{e4});
  CHECK(get_predecessors(g, {n1, n2, n3}) == std::unordered_map<Node, std::unordered_set<Node>>{
    {n1, {}},
    {n2, {n1}},
    {n3, {n1, n2}},
  });
}

TEST_CASE("DiGraph") {
  using namespace FlexFlow::utils::graph::digraph;

  AdjacencyDiGraph g;
  Node n1 = g.add_node();
  Node n2 = g.add_node();
  Node n3 = g.add_node();
  Node n4 = g.add_node();
  Edge e1 {n1, n4};
  Edge e2 {n1, n2};
  Edge e3 {n1, n3};
  Edge e4 {n2, n3};
  g.add_edge(e1);
  g.add_edge(e2);
  g.add_edge(e3);
  g.add_edge(e4);

  CHECK(get_nodes(g) == std::unordered_set<Node>{n1, n2, n3, n4});
  CHECK(get_edges(g) == std::unordered_set<Edge>{e1, e2, e3, e4});
  CHECK(get_incoming_edges(g, {n2, n4}) == std::unordered_set<Edge>{e1, e2});
  CHECK(get_outgoing_edges(g, {n2, n4}) == std::unordered_set<Edge>{e4});
  CHECK(get_predecessors(g, {n1, n2, n3}) == std::unordered_map<Node, std::unordered_set<Node>>{
    {n1, {}},
    {n2, {n1}},
    {n3, {n1, n2}},
  });
}

TEST_CASE("dfs") {
  using namespace FlexFlow::utils::graph::digraph;

  AdjacencyDiGraph g;
  std::vector<Node> const n = add_nodes(g, 4);
  g.add_edge({n[0], n[1]});
  g.add_edge({n[1], n[2]});
  g.add_edge({n[2], n[3]});

  /* CHECK(n[0] == n[1]); */
  /* CHECK(std::vector<Node>{n[0]} == std::vector<Node>{n[0], n[1], n[2], n[3]}); */
  CHECK(dfs_ordering(g, {n[0]}) == std::vector<Node>{n[0], n[1], n[2], n[3]});
}

#include "doctest.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/adjacency_multidigraph.h"
#include "utils/graph/construction.h"

using namespace FlexFlow::utils::graph;
using namespace FlexFlow::utils::graph::multidigraph;

TEST_CASE("MultiDiGraph") {
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
  CHECK(get_predecessors(g, {n1, n2, n3}) == std::unordered_map<Node, std::unordered_set<Node>>{
    {n1, {}},
    {n2, {n1}},
    {n3, {n1, n2}},
  });
}

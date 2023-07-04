#include "doctest.h"
#include "utils/graph/adjacency_digraph.h"
#include "utils/graph/adjacency_multidigraph.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/construction.h"
#include "utils/containers.h"
using namespace FlexFlow;

TEST_CASE("MultiDiGraph") {
  AdjacencyMultiDiGraph g;
  Node n0 = g.add_node();
  Node n1 = g.add_node();
  Node n2 = g.add_node();
  Node n3 = g.add_node();
  NodePort p0(0);
  NodePort p1(1);
  NodePort p2(2);
  NodePort p3(3); 
  MultiDiEdge e0{n0, n3, p0, p3};
  MultiDiEdge e1{n1, n2, p0, p2};
  MultiDiEdge e2{n1, n3, p1, p3};
  MultiDiEdge e3{n2, n3, p2, p3};
  g.add_edge(e0);
  g.add_edge(e1);
  g.add_edge(e2);
  g.add_edge(e3);

  CHECK(get_nodes(g) == std::unordered_set<Node>{n0, n1, n2, n3});
  CHECK(get_edges(g) == std::unordered_set<MultiDiEdge>{e0, e1, e2, e3});
  CHECK(get_incoming_edges(g, {n1, n3}) ==
        std::unordered_set<MultiDiEdge>{e0, e2, e3});
  CHECK(get_incoming_edges(g, {n1}) == std::unordered_set<MultiDiEdge>{});
  CHECK(get_outgoing_edges(g, {n2, n3}) == std::unordered_set<MultiDiEdge>{e3});
  auto res = get_predecessors(g, {n1, n2, n3});
  auto expected_result = std::unordered_map<Node, std::unordered_set<Node>>{
            {n1, {}},
            {n2, {n1}},
            {n3, {n0,n1, n2}},
        };
  for(auto kv : res) {
    CHECK(expected_result[kv.first] == kv.second);
  }
}

// TEST_CASE("DiGraph") {
//   AdjacencyDiGraph g;
//   Node n0 = g.add_node();
//   Node n1 = g.add_node();
//   Node n2 = g.add_node();
//   Node n3 = g.add_node();
//   DirectedEdge e0{n0, n3};
//   DirectedEdge e1{n0, n1};
//   DirectedEdge e2{n0, n2};
//   DirectedEdge e3{n1, n2};
//   g.add_edge(e0);
//   g.add_edge(e1);
//   g.add_edge(e2);
//   g.add_edge(e3);

//   CHECK(get_nodes(g) == std::unordered_set<Node>{n0, n1, n2, n3});
//   CHECK(get_edges(g) == std::unordered_set<DirectedEdge>{e0, e1, e2, e3});
//   CHECK(get_incoming_edges(g, {n2, n3}) ==
//         std::unordered_set<DirectedEdge>{e0, e2, e3});
//   CHECK(get_outgoing_edges(g, {n2, n3}) ==
//         std::unordered_set<DirectedEdge>{});
//   CHECK(get_predecessors(g, {n1, n2, n3}) ==
//         std::unordered_map<Node, std::unordered_set<Node>>{
//             {n1, {n0}},
//             {n2, {n0, n1}},
//             {n3, {n0}},
//         });
// }

// TEST_CASE("traversal") {
//   AdjacencyDiGraph g;
//   std::vector<Node> const n = add_nodes(g, 4);
//   g.add_edge({n[0], n[1]});
//   g.add_edge({n[1], n[2]});
//   g.add_edge({n[2], n[3]});

//   /* CHECK(get_incoming_edges(g, n[0]) == std::unordered_set<DirectedEdge>{});
//    */
//   CHECK(get_sources(g) == std::unordered_set<Node>{n[0]});
//   CHECK(unchecked_dfs_ordering(g, {n[0]}) ==
//         std::vector<Node>{n[0], n[1], n[2], n[3]});
//   CHECK(bfs_ordering(g, {n[0]}) == std::vector<Node>{n[0], n[1], n[2], n[3]});
//   CHECK(is_acyclic(g) == true);

//   SUBCASE("with root") {
//     g.add_edge({n[3], n[2]});

//     CHECK(dfs_ordering(g, {n[0]}) == std::vector<Node>{n[0], n[1], n[2], n[3]});
//     CHECK(is_acyclic(g) == false);
//   }

//   SUBCASE("without root") {
//     g.add_edge({n[3], n[0]});

//     CHECK(dfs_ordering(g, {n[0]}) == std::vector<Node>{n[0], n[1], n[2], n[3]});
//     CHECK(is_acyclic(g) == false);
//   }

//   SUBCASE("nonlinear") {
//     g.add_edge({n[1], n[3]});
//     CHECK(is_acyclic(g) == true);
//   }
// }

// TEST_CASE("bfs") {
//   AdjacencyDiGraph g;
//   std::vector<Node> const n = add_nodes(g, 7);
//   add_edges(g,
//             {
//                 {n[0], n[1]},
//                 {n[0], n[2]},
//                 {n[1], n[6]},
//                 {n[2], n[3]},
//                 {n[3], n[4]},
//                 {n[4], n[5]},
//                 {n[5], n[6]},
//                 {n[6], n[0]},
//             });

//   std::vector<Node> ordering = bfs_ordering(g, {n[0]});
//   auto CHECK_BEFORE = [&](int l, int r) {
//     CHECK(index_of(ordering, n[l]).has_value());
//     CHECK(index_of(ordering, n[r]).has_value());
//     CHECK(index_of(ordering, n[l]).value() < index_of(ordering, n[r]).value());
//   };

//   CHECK(ordering.size() == n.size());
//   CHECK_BEFORE(0, 1);
//   CHECK_BEFORE(0, 2);

//   CHECK_BEFORE(1, 3);
//   CHECK_BEFORE(1, 6);
//   CHECK_BEFORE(2, 3);
//   CHECK_BEFORE(2, 6);

//   CHECK_BEFORE(3, 4);
//   CHECK_BEFORE(6, 4);

//   CHECK_BEFORE(4, 5);
// }

// TEST_CASE("topological_ordering") {
//   AdjacencyDiGraph g;
//   std::vector<Node> const n = add_nodes(g, 6);
//   add_edges(g,
//             {{n[0], n[1]},
//              {n[0], n[2]},
//              {n[1], n[5]},
//              {n[2], n[3]},
//              {n[3], n[4]},
//              {n[4], n[5]}});

//   std::vector<Node> ordering = topological_ordering(g);
//   auto CHECK_BEFORE = [&](int l, int r) {
//     CHECK(index_of(ordering, n[l]).has_value());
//     CHECK(index_of(ordering, n[r]).has_value());
//     CHECK(index_of(ordering, n[l]) < index_of(ordering, n[r]));
//   };

//   CHECK(ordering.size() == n.size());
//   CHECK_BEFORE(0, 1);
//   CHECK_BEFORE(0, 2);
//   CHECK_BEFORE(1, 5);
//   CHECK_BEFORE(2, 3);
//   CHECK_BEFORE(3, 4);
//   CHECK_BEFORE(4, 5);
// }

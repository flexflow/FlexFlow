#include "doctest.h"
#include "utils/containers.h"
#include "utils/graph/adjacency_digraph.h"
#include "utils/graph/adjacency_multidigraph.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/construction.h"
#include <cinttypes>
#include <iterator>
#include <type_traits>
#include <unordered_set>

using namespace FlexFlow;

TEST_CASE("MultiDiGraph") {
  MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();
  std::vector<Node> nodes = g.add_nodes(4);
  std::vector<NodePort> ports = g.add_node_ports(4);

  Node n0 = nodes[0];
  Node n1 = nodes[1];
  Node n2 = nodes[2];
  Node n3 = nodes[3];
  NodePort p0 = ports[0];
  NodePort p1 = ports[1];
  NodePort p2 = ports[2];
  NodePort p3 = ports[3];

  MultiDiEdge e0{n0, n3, p0, p3};
  MultiDiEdge e1{n1, n2, p0, p2};
  MultiDiEdge e2{n1, n3, p1, p3};
  MultiDiEdge e3{n2, n3, p2, p3};

  std::vector<MultiDiEdge> edges = {e0, e1, e2, e3};

  g.add_edges(edges);

  CHECK(get_incoming_edges(g, {n1, n3}) ==
        std::unordered_set<MultiDiEdge>{e0, e2, e3});
  CHECK(get_incoming_edges(g, {n1}) == std::unordered_set<MultiDiEdge>{});
  CHECK(get_outgoing_edges(g, {n2, n3}) == std::unordered_set<MultiDiEdge>{e3});
  std::unordered_map<Node, std::unordered_set<Node>> res =
      get_predecessors(g, {n1, n2, n3});
  std::unordered_map<Node, std::unordered_set<Node>> expected_result =
      std::unordered_map<Node, std::unordered_set<Node>>{
          {n1, {}},
          {n2, {n1}},
          {n3, {n0, n1, n2}},
      };
  CHECK(res == expected_result);
}

TEST_CASE("DiGraph") {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();

  std::vector<Node> nodes = add_nodes(g, 4);
  Node n0 = nodes[0];
  Node n1 = nodes[1];
  Node n2 = nodes[2];
  Node n3 = nodes[3];

  DirectedEdge e0{n0, n3};
  DirectedEdge e1{n0, n1};
  DirectedEdge e2{n0, n2};
  DirectedEdge e3{n1, n2};

  std::vector<DirectedEdge> edges = {e0, e1, e2, e3};
  add_edges(g, edges);

  CHECK(get_incoming_edges(g, {n2, n3}) ==
        std::unordered_set<DirectedEdge>{e0, e2, e3});
  CHECK(get_outgoing_edges(g, {n2, n3}) == std::unordered_set<DirectedEdge>{});
  auto expected_result = std::unordered_map<Node, std::unordered_set<Node>>{
      {n1, {n0}},
      {n2, {n0, n1}},
      {n3, {n0}},
  };
  auto res = get_predecessors(g, {n1, n2, n3});
  for (auto kv : res) {
    CHECK(expected_result[kv.first] == kv.second);
  }

  SUBCASE("get_imm_dominators") {
    std::unordered_map<Node, optional<Node>> result = get_imm_dominators(g);

    CHECK(result.size() == 4);
    CHECK(result[n0] == nullopt);

    CHECK(*result[n1] == n0);
    CHECK(*result[n2] == n0);
    CHECK(*result[n3] == n0);
  }

  SUBCASE("get_dominators") {
    std::unordered_map<Node, std::unordered_set<Node>> result =
        get_dominators(g);

    CHECK(result.size() == 4);
    CHECK(result[n0] == std::unordered_set<Node>{n0});
    CHECK(result[n1] == std::unordered_set<Node>{n1});
    CHECK(result[n2] == std::unordered_set<Node>{n0, n2});
    CHECK(result[n3] == std::unordered_set<Node>{n3});
  }

  SUBCASE("get_neighbors") {
    std::unordered_set<Node> result = get_neighbors(g, n0);
    auto expected = std::unordered_set<Node>{n3, n1, n2};
    CHECK(result == expected);
  }

  SUBCASE("get_sinks") {
    std::unordered_set<Node> result = get_sinks(g);
    auto expected = std::unordered_set<Node>{n2, n3};
    CHECK(result == expected);
  }

  SUBCASE("get_bfs") {
    std::unordered_set<Node> start_points = std::unordered_set<Node>{n0};
    auto result = get_bfs_ordering(g, start_points);
    auto expected = std::vector<Node>{n0, n2, n1, n3};
    CHECK(result == expected);
  }

  SUBCASE("get_predecessors") {
    std::unordered_set<Node> nodes{n1, n2};
    std::unordered_map<Node, std::unordered_set<Node>> result =
        get_predecessors(g, nodes);
    CHECK(result.size() == 2);

    auto n1_predecessors = result[n1];
    auto n1_expected = std::unordered_set<Node>{n0};
    CHECK(n1_predecessors == n1_expected);

    auto n2_predecessors = result[n2];
    auto n2_expected = std::unordered_set<Node>{n0, n1};
    CHECK(n2_predecessors == n2_expected);
  }
}

TEST_CASE("traversal") {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> const n = add_nodes(g, 4);
  g.add_edge({n[0], n[1]});
  g.add_edge({n[1], n[2]});
  g.add_edge({n[2], n[3]});

  CHECK(get_sources(g) == std::unordered_set<Node>{n[0]});
  CHECK(get_unchecked_dfs_ordering(g, {n[0]}) ==
        std::vector<Node>{n[0], n[1], n[2], n[3]});
  CHECK(get_bfs_ordering(g, {n[0]}) ==
        std::vector<Node>{n[0], n[1], n[2], n[3]});
  CHECK(is_acyclic(g) == true);

  SUBCASE("with root") {
    g.add_edge({n[3], n[2]});

    CHECK(get_dfs_ordering(g, {n[0]}) ==
          std::vector<Node>{n[0], n[1], n[2], n[3]});
    CHECK(is_acyclic(g) == false);
  }

  SUBCASE("without root") {
    g.add_edge({n[3], n[0]});

    CHECK(get_dfs_ordering(g, {n[0]}) ==
          std::vector<Node>{n[0], n[1], n[2], n[3]});
    CHECK(is_acyclic(g) == false);
  }

  //   SUBCASE("nonlinear") {
  //     g.add_edge({n[1], n[3]});
  //     CHECK(is_acyclic(g) == true);//TODO, maybe a bug about the
  //     unchecked_dfs
  //   }
}

TEST_CASE("bfs") {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> const n = add_nodes(g, 7);

  std::vector<DirectedEdge> edges = {
      {n[0], n[1]},
      {n[0], n[2]},
      {n[1], n[6]},
      {n[2], n[3]},
      {n[3], n[4]},
      {n[4], n[5]},
      {n[5], n[6]},
      {n[6], n[0]},
  };

  add_edges(g, edges);

  std::vector<Node> ordering = get_bfs_ordering(g, {n[0]});
  auto CHECK_BEFORE = [&](int l, int r) {
    CHECK(index_of(ordering, n[l]).has_value());
    CHECK(index_of(ordering, n[r]).has_value());
    CHECK(index_of(ordering, n[l]).value() < index_of(ordering, n[r]).value());
  };

  CHECK(ordering.size() == n.size());
  CHECK_BEFORE(0, 1);
  CHECK_BEFORE(0, 2);

  CHECK_BEFORE(1, 3);
  CHECK_BEFORE(1, 6);
  CHECK_BEFORE(2, 3);
  CHECK_BEFORE(2, 6);

  CHECK_BEFORE(3, 4);
  CHECK_BEFORE(6, 4);

  CHECK_BEFORE(4, 5);
}

TEST_CASE("topological_ordering") {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> n = add_nodes(g, 6);
  std::vector<DirectedEdge> edges = {{n[0], n[1]},
                                     {n[0], n[2]},
                                     {n[1], n[5]},
                                     {n[2], n[3]},
                                     {n[3], n[4]},
                                     {n[4], n[5]}};
  add_edges(g, edges);
  std::vector<Node> ordering = get_topological_ordering(g);
  auto CHECK_BEFORE = [&](int l, int r) {
    CHECK(index_of(ordering, n[l]).has_value());
    CHECK(index_of(ordering, n[r]).has_value());
    CHECK(index_of(ordering, n[l]) < index_of(ordering, n[r]));
  };

  CHECK(ordering.size() == n.size());
  CHECK_BEFORE(0, 1);
  CHECK_BEFORE(0, 2);
  CHECK_BEFORE(1, 5);
  CHECK_BEFORE(2, 3);
  CHECK_BEFORE(3, 4);
  CHECK_BEFORE(4, 5);
}
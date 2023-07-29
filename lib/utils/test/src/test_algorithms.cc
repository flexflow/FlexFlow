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
  std::vector<Node> n = g.add_nodes(4);
  std::vector<NodePort> p = g.add_node_ports(4);

  std::vector<MultiDiEdge> e = {
      {n[0], n[3], p[0], p[3]},
      {n[1], n[2], p[0], p[2]},
      {n[1], n[3], p[1], p[3]},
      {n[2], n[3], p[2], p[3]},
  };

  g.add_edges(e);

  CHECK(get_incoming_edges(g, {n[1], n[3]}) ==
        std::unordered_set<MultiDiEdge>{e[0], e[2], e[3]});
  CHECK(get_incoming_edges(g, {n[1]}) == std::unordered_set<MultiDiEdge>{});
  CHECK(get_outgoing_edges(g, {n[2], n[3]}) ==
        std::unordered_set<MultiDiEdge>{e[3]});
  std::unordered_map<Node, std::unordered_set<Node>> res =
      get_predecessors(g, {n[1], n[2], n[3]});
  std::unordered_map<Node, std::unordered_set<Node>> expected_result =
      std::unordered_map<Node, std::unordered_set<Node>>{
          {n[1], {}},
          {n[2], {n[1]}},
          {n[3], {n[0], n[1], n[2]}},
      };
  CHECK(res == expected_result);
}

TEST_CASE("DiGraph") {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();

  std::vector<Node> n = add_nodes(g, 4);
  std::vector<DirectedEdge> e = {
      {n[0], n[3]},
      {n[0], n[1]},
      {n[0], n[2]},
      {n[1], n[2]},
  };
  add_edges(g, e);

  CHECK(get_incoming_edges(g, {n[2], n[3]}) ==
        std::unordered_set<DirectedEdge>{e[0], e[2], e[3]});
  CHECK(get_outgoing_edges(g, {n[2], n[3]}) ==
        std::unordered_set<DirectedEdge>{});
  auto expected_result = std::unordered_map<Node, std::unordered_set<Node>>{
      {n[1], {n[0]}},
      {n[2], {n[0], n[1]}},
      {n[3], {n[0]}},
  };
  auto res = get_predecessors(g, {n[1], n[2], n[3]});
  CHECK(res == expected_result);

  SUBCASE("get_imm_dominators") {
    std::unordered_map<Node, Node> result = map_values(get_imm_dominators(g), [&](optional<Node> const & node) { return *node; });

    std::unordered_map<Node, Node> expected_result = {
        {n[2], n[0]},
        {n[1], n[0]},
        {n[3], n[0]},
        {n[0], n[0]},
    };
    CHECK(result == expected_result);
  }

  SUBCASE("get_dominators") {
    std::unordered_map<Node, std::unordered_set<Node>> result =
        get_dominators(g);

    CHECK(result.size() == 4);
    CHECK(result[n[0]] == std::unordered_set<Node>{n[0]});
    CHECK(result[n[1]] == std::unordered_set<Node>{n[1]});
    CHECK(result[n[2]] == std::unordered_set<Node>{n[0], n[2]});
    CHECK(result[n[3]] == std::unordered_set<Node>{n[3]});
  }

  SUBCASE("get_sinks") {
    std::unordered_set<Node> result = get_sinks(g);
    auto expected = std::unordered_set<Node>{n[2], n[3]};
    CHECK(result == expected);
  }

  SUBCASE("get_bfs") {
    std::unordered_set<Node> start_points = std::unordered_set<Node>{n[0]};
    auto result = get_bfs_ordering(g, start_points);
    auto expected = std::vector<Node>{n[0], n[2], n[1], n[3]};
    CHECK(result == expected);
  }

  SUBCASE("get_predecessors") {
    std::unordered_set<Node> nodes{n[1], n[2]};
    std::unordered_map<Node, std::unordered_set<Node>> result =
        get_predecessors(g, nodes);
    CHECK(result.size() == 2);

    auto n1_predecessors = result[n[1]];
    auto n1_expected = std::unordered_set<Node>{n[0]};
    CHECK(n1_predecessors == n1_expected);

    auto n2_predecessors = result[n[2]];
    auto n2_expected = std::unordered_set<Node>{n[0], n[1]};
    CHECK(n2_predecessors == n2_expected);
  }
}

TEST_CASE("traversal") {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> const n = add_nodes(g, 4);
  std::vector<DirectedEdge> edges = {{n[0], n[1]}, {n[1], n[2]}, {n[2], n[3]}};
  add_edges(g, edges);

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
  // SUBCASE("nonlinear") {
  //     g.add_edge({n[1], n[3]});
  //   CHECK(is_acyclic(g) == true);//TODO, maybe a bug about the
  //   unchecked_dfs
  // }
}

TEST_CASE("bfs") {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> const n = add_nodes(g, 7);

  std::vector<DirectedEdge> e = {
      {n[0], n[1]},
      {n[0], n[2]},
      {n[1], n[6]},
      {n[2], n[3]},
      {n[3], n[4]},
      {n[4], n[5]},
      {n[5], n[6]},
      {n[6], n[0]},
  };

  add_edges(g, e);

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

TEST_CASE("get_weakly_connected_components") {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> n = add_nodes(g, 4);

  std::vector<DirectedEdge> edges = {{n[0], n[1]}, {n[1], n[2]}, {n[2], n[3]}};

  add_edges(g, edges);
  std::vector<std::unordered_set<Node>> components =
      get_weakly_connected_components(g);
  std::vector<std::unordered_set<Node>> expected_components = {
      {n[0]},
      {n[1]},
      {n[2]},
      {n[3]},
  };

  CHECK(components == expected_components);
}
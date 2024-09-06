#include "test/utils/doctest.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/construction.h"
#include "utils/graph/hashmap_undirected_graph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/undirected.h"
#include <cinttypes>
#include <iterator>
#include <type_traits>
#include <unordered_set>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("MultiDiGraph") {
    MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();
    std::vector<Node> n = add_nodes(g, 4);
    std::vector<NodePort> p = add_node_ports(g, 4);

    MultiDiEdge e0{n[3], p[3], n[0], p[0]};
    MultiDiEdge e1{n[2], p[2], n[1], p[0]};
    MultiDiEdge e2{n[3], p[3], n[1], p[1]};
    MultiDiEdge e3{n[3], p[3], n[2], p[2]};

    std::vector<MultiDiEdge> e = {e0, e1, e2, e3};

    add_edges(g, e);

    CHECK(get_incoming_edges(g, {n[1], n[3]}) ==
          std::unordered_set<MultiDiEdge>{e[0], e[2], e[3]});
    CHECK(get_incoming_edges(g, {n[1]}) == std::unordered_set<MultiDiEdge>{});
    CHECK(get_outgoing_edges(g, {n[2], n[3]}) ==
          std::unordered_set<MultiDiEdge>{e[3]});
    std::unordered_map<Node, std::unordered_set<Node>> expected_result =
        std::unordered_map<Node, std::unordered_set<Node>>{
            {n[1], {}},
            {n[2], {n[1]}},
            {n[3], {n[0], n[1], n[2]}},
        };
    CHECK(get_predecessors(g, {n[1], n[2], n[3]}) == expected_result);
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
    CHECK(get_predecessors(g, {n[1], n[2], n[3]}) == expected_result);

    SUBCASE("get_imm_dominators") {
      std::unordered_map<Node, std::optional<Node>> result =
          get_imm_dominators(g);

      std::unordered_map<Node, std::optional<Node>> expected_result = {
          {n[2], n[0]},
          {n[1], n[0]},
          {n[3], n[0]},
          {n[0], std::nullopt},
      };
      CHECK(result == expected_result);
    }

    SUBCASE("get_dominators") {
      std::unordered_map<Node, std::unordered_set<Node>> expected = {
          {n[0], {n[0]}},
          {n[1], {n[0], n[1]}},
          {n[2], {n[0], n[2]}},
          {n[3], {n[0], n[3]}},
      };
      CHECK(get_dominators(g) == expected);
    }

    SUBCASE("get_sinks") {
      auto expected = std::unordered_set<Node>{n[2], n[3]};
      CHECK(get_sinks(g) == expected);
    }

    SUBCASE("get_bfs") {
      std::unordered_set<Node> start_points = std::unordered_set<Node>{n[0]};
      auto expected = std::vector<Node>{n[0], n[2], n[1], n[3]};
      CHECK(get_bfs_ordering(g, start_points) == expected);
    }

    SUBCASE("get_predecessors") {
      std::unordered_map<Node, std::unordered_set<Node>> expected_result = {
          {n[1], {n[0]}},
          {n[2], {n[0], n[1]}},
      };
      CHECK(get_predecessors(g, {n[1], n[2]}) == expected_result);
    }
  }

  TEST_CASE("traversal") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> const n = add_nodes(g, 5);
    std::vector<DirectedEdge> edges = {
        {n[0], n[1]}, {n[1], n[2]}, {n[2], n[3]}};
    add_edges(g, edges);

    CHECK(get_sources(g) == std::unordered_set<Node>{n[0], n[4]});
    CHECK(get_unchecked_dfs_ordering(g, {n[0]}) ==
          std::vector<Node>{n[0], n[1], n[2], n[3]});
    CHECK(get_bfs_ordering(g, {n[0]}) ==
          std::vector<Node>{n[0], n[1], n[2], n[3]});
    CHECK(is_acyclic(g) == true);
    CHECK(get_bfs_ordering(g, {n[4]}) == std::vector<Node>{n[4]});
    CHECK(get_dfs_ordering(g, {n[4]}) == std::vector<Node>{n[4]});

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
    SUBCASE("nonlinear") {
      g.add_edge({n[1], n[3]});
      CHECK(is_acyclic(g) == true);
    }

    SUBCASE("not connected") {
      g.remove_edge({n[2], n[3]});
      CHECK(get_dfs_ordering(g, {n[0]}) == std::vector<Node>{n[0], n[1], n[2]});
    }
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
      CHECK(index_of(ordering, n[l]).value() <
            index_of(ordering, n[r]).value());
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

  TEST_CASE("get_topological_ordering") {
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

  TEST_CASE("get_connected_components") {
    UndirectedGraph g = UndirectedGraph::create<HashmapUndirectedGraph>();
    std::vector<Node> n = add_nodes(g, 4);
    std::vector<UndirectedEdge> edges = {{n[0], n[1]}, {n[2], n[1]}};

    add_edges(g, edges);
    std::unordered_set<std::unordered_set<Node>> expected_components = {
        {n[0], n[1], n[2]},
        {n[3]},
    };

    CHECK(get_connected_components(g) == expected_components);
  }

  TEST_CASE("get_weakly_connected_components") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 4);

    std::vector<DirectedEdge> edges = {{n[0], n[1]}, {n[2], n[1]}};

    add_edges(g, edges);
    std::unordered_set<std::unordered_set<Node>> expected_components = {
        {n[0], n[1], n[2]},
        {n[3]},
    };

    CHECK(get_outgoing_edges(as_digraph(as_undirected(g)), n[0]).size() == 1);

    CHECK(get_weakly_connected_components(g) == expected_components);
  }
}

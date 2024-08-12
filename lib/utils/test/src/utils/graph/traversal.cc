#include "utils/graph/traversal.h"
#include "test/utils/doctest.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/algorithms.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("traversal") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> const n = add_nodes(g, 5);
    std::vector<DirectedEdge> edges = {
        DirectedEdge{n[0], n[1]}, DirectedEdge{n[1], n[2]}, DirectedEdge{n[2], n[3]}};
    add_edges(g, edges);

    CHECK(get_unchecked_dfs_ordering(g, {n[0]}) ==
          std::vector<Node>{n[0], n[1], n[2], n[3]});
    CHECK(get_bfs_ordering(g, {n[0]}) ==
          std::vector<Node>{n[0], n[1], n[2], n[3]});
    CHECK(get_bfs_ordering(g, {n[4]}) == std::vector<Node>{n[4]});
    CHECK(get_dfs_ordering(g, {n[4]}) == std::vector<Node>{n[4]});

    SUBCASE("with root") {
      g.add_edge(DirectedEdge{n[3], n[2]});

      CHECK(get_dfs_ordering(g, {n[0]}) ==
            std::vector<Node>{n[0], n[1], n[2], n[3]});
    }

    SUBCASE("without root") {
      g.add_edge(DirectedEdge{n[3], n[0]});

      CHECK(get_dfs_ordering(g, {n[0]}) ==
            std::vector<Node>{n[0], n[1], n[2], n[3]});
    }
    SUBCASE("nonlinear") {
      g.add_edge(DirectedEdge{n[1], n[3]});
    }

    SUBCASE("not connected") {
      g.remove_edge(DirectedEdge{n[2], n[3]});
      CHECK(get_dfs_ordering(g, {n[0]}) == std::vector<Node>{n[0], n[1], n[2]});
    }
  }
}
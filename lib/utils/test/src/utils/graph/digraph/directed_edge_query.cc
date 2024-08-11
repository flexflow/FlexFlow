#include "utils/graph/digraph/directed_edge_query.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("directed_edge_query") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 5);

    add_edges(g,
              {DirectedEdge{n[0], n[1]},
               DirectedEdge{n[0], n[2]},
               DirectedEdge{n[1], n[2]},
               DirectedEdge{n[2], n[4]},
               DirectedEdge{n[1], n[3]}});

    SUBCASE("directed_edge_query_all") {

      DirectedEdgeQuery result = directed_edge_query_all();

      CHECK(matches_edge(result, DirectedEdge{n[0], n[1]}));
      CHECK(matches_edge(result, DirectedEdge{n[0], n[2]}));
      CHECK(matches_edge(result, DirectedEdge{n[1], n[2]}));
      CHECK(matches_edge(result, DirectedEdge{n[2], n[4]}));
      CHECK(matches_edge(result, DirectedEdge{n[1], n[3]}));
    }

    SUBCASE("matches_edge") {
      DirectedEdgeQuery q{{n[0]}, {n[1]}};

      CHECK(matches_edge(q, DirectedEdge{n[0], n[1]}));
      CHECK_FALSE(matches_edge(q, DirectedEdge{n[1], n[2]}));
    }

    SUBCASE("query_intersection") {
      DirectedEdgeQuery q1{{n[0], n[1]}, {n[1], n[2], n[4]}};
      DirectedEdgeQuery q2{{n[1], n[2]}, {n[2], n[3]}};

      DirectedEdgeQuery result = query_intersection(q1, q2);

      std::unordered_set<Node> expected_srcs = {n[1]};
      std::unordered_set<Node> expected_dsts = {n[2]};

      CHECK(result.srcs == expected_srcs);
      CHECK(result.dsts == expected_dsts);
    }
  }
}

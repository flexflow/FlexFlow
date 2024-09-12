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
              {DirectedEdge{n.at(0), n.at(1)},
               DirectedEdge{n.at(0), n.at(2)},
               DirectedEdge{n.at(1), n.at(2)},
               DirectedEdge{n.at(2), n.at(4)},
               DirectedEdge{n.at(1), n.at(3)}});

    SUBCASE("directed_edge_query_all") {

      DirectedEdgeQuery result = directed_edge_query_all();

      CHECK(matches_edge(result, DirectedEdge{n.at(0), n.at(1)}));
      CHECK(matches_edge(result, DirectedEdge{n.at(0), n.at(2)}));
      CHECK(matches_edge(result, DirectedEdge{n.at(1), n.at(2)}));
      CHECK(matches_edge(result, DirectedEdge{n.at(2), n.at(4)}));
      CHECK(matches_edge(result, DirectedEdge{n.at(1), n.at(3)}));
    }

    SUBCASE("matches_edge") {
      DirectedEdgeQuery q =
          DirectedEdgeQuery{query_set{n.at(0)}, query_set{n.at(1)}};

      CHECK(matches_edge(q, DirectedEdge{n.at(0), n.at(1)}));
      CHECK_FALSE(matches_edge(q, DirectedEdge{n.at(1), n.at(2)}));
    }

    SUBCASE("query_intersection") {
      SUBCASE("standard intersection") {
        DirectedEdgeQuery q1 = DirectedEdgeQuery{
            query_set{n.at(0), n.at(1)}, query_set{n.at(1), n.at(2), n.at(4)}};
        DirectedEdgeQuery q2 = DirectedEdgeQuery{query_set{n.at(1), n.at(2)},
                                                 query_set{n.at(2), n.at(3)}};

        DirectedEdgeQuery result = query_intersection(q1, q2);
        DirectedEdgeQuery correct = DirectedEdgeQuery{
            query_set{n.at(1)},
            query_set{n.at(2)},
        };

        CHECK(result == correct);
      }
      SUBCASE("intersection with std::nullopt") {
        DirectedEdgeQuery q1 =
            DirectedEdgeQuery{query_set{n.at(1), n.at(2)}, matchall<Node>()};
        DirectedEdgeQuery q2 =
            DirectedEdgeQuery{matchall<Node>(), query_set{n.at(3), n.at(4)}};

        DirectedEdgeQuery result = query_intersection(q1, q2);

        CHECK(matches_edge(result, DirectedEdge{n.at(1), n.at(3)}));
        CHECK(matches_edge(result, DirectedEdge{n.at(2), n.at(4)}));
        // CHECK_FALSE(matches_edge(result, DirectedEdge{n.at(2), n.at(3)}));
        // CHECK_FALSE(matches_edge(result, DirectedEdge{n.at(1), n.at(4)}));
      }
    }
  }
}

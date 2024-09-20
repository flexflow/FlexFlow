#include "utils/graph/digraph/algorithms/apply_contraction.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("apply_contraction") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 6);

    add_edges(g,
              {
                  DirectedEdge{n.at(0), n.at(1)},
                  DirectedEdge{n.at(0), n.at(2)},
                  DirectedEdge{n.at(0), n.at(4)},
                  DirectedEdge{n.at(1), n.at(3)},
                  DirectedEdge{n.at(2), n.at(4)},
                  DirectedEdge{n.at(3), n.at(0)},
                  DirectedEdge{n.at(3), n.at(4)},
                  DirectedEdge{n.at(3), n.at(5)},
                  DirectedEdge{n.at(4), n.at(5)},
              });

    DiGraphView result = apply_contraction(g,
                                           {
                                               {n.at(0), n.at(3)},
                                               {n.at(3), n.at(2)},
                                               {n.at(2), n.at(2)},
                                               {n.at(1), n.at(4)},
                                           });

    SUBCASE("nodes") {
      std::unordered_set<Node> result_nodes = get_nodes(result);
      std::unordered_set<Node> correct_nodes = {n.at(2), n.at(4), n.at(5)};
      CHECK(result_nodes == correct_nodes);
    }

    SUBCASE("edges") {
      std::unordered_set<DirectedEdge> result_edges = get_edges(result);
      std::unordered_set<DirectedEdge> correct_edges = {
          DirectedEdge{n.at(2), n.at(4)},
          DirectedEdge{n.at(2), n.at(2)},
          DirectedEdge{n.at(4), n.at(2)},
          DirectedEdge{n.at(2), n.at(5)},
          DirectedEdge{n.at(4), n.at(5)},
      };
      CHECK(result_edges == correct_edges);
    }
  }
}

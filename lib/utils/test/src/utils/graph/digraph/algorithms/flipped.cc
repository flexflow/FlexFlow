#include "utils/graph/digraph/algorithms/flipped.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("flipped_directed_edge") {
    DirectedEdge input = DirectedEdge{Node{0}, Node{1}};
    DirectedEdge result = flipped_directed_edge(input);
    DirectedEdge correct = DirectedEdge{Node{1}, Node{0}};
    CHECK(result == correct);
  }

  TEST_CASE("flipped") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 6);

    add_edges(g,
              {
                  DirectedEdge{n.at(0), n.at(1)},
                  DirectedEdge{n.at(1), n.at(2)},
                  DirectedEdge{n.at(1), n.at(3)},
                  DirectedEdge{n.at(1), n.at(5)},
                  DirectedEdge{n.at(2), n.at(4)},
                  DirectedEdge{n.at(3), n.at(1)},
                  DirectedEdge{n.at(3), n.at(4)},
              });

    DiGraphView result = flipped(g);

    SUBCASE("nodes") {
      std::unordered_set<Node> correct_nodes = unordered_set_of(n);
      std::unordered_set<Node> result_nodes = get_nodes(result);
      CHECK(result_nodes == correct_nodes);
    }

    SUBCASE("edges") {
      std::unordered_set<DirectedEdge> correct_edges = {
          DirectedEdge{n.at(1), n.at(0)},
          DirectedEdge{n.at(2), n.at(1)},
          DirectedEdge{n.at(3), n.at(1)},
          DirectedEdge{n.at(5), n.at(1)},
          DirectedEdge{n.at(4), n.at(2)},
          DirectedEdge{n.at(1), n.at(3)},
          DirectedEdge{n.at(4), n.at(3)},
      };
      std::unordered_set<DirectedEdge> result_edges = get_edges(result);
      CHECK(result_edges == correct_edges);
    }
  }
}

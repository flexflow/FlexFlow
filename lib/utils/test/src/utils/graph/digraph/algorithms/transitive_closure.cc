#include <doctest/doctest.h>
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/transitive_closure.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transitive_closure(DiGraphView)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    SUBCASE("maximum number of new edges") {
      std::vector<Node> n = add_nodes(g, 5);

      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(2), n.at(3)},
                    DirectedEdge{n.at(3), n.at(4)},
                });

      DiGraphView result = transitive_closure(g);

      SUBCASE("nodes") {
        std::unordered_set<Node> result_nodes = get_nodes(result);
        std::unordered_set<Node> correct_nodes = unordered_set_of(n);
        CHECK(result_nodes == correct_nodes);
      }

      SUBCASE("edges") {
        std::unordered_set<DirectedEdge> result_edges = get_edges(result);
        std::unordered_set<DirectedEdge> correct_edges = {
            DirectedEdge{n.at(0), n.at(1)},
            DirectedEdge{n.at(0), n.at(2)},
            DirectedEdge{n.at(0), n.at(3)},
            DirectedEdge{n.at(0), n.at(4)},
            DirectedEdge{n.at(1), n.at(2)},
            DirectedEdge{n.at(1), n.at(3)},
            DirectedEdge{n.at(1), n.at(4)},
            DirectedEdge{n.at(2), n.at(3)},
            DirectedEdge{n.at(2), n.at(4)},
            DirectedEdge{n.at(3), n.at(4)},
        };
        CHECK(result_edges == correct_edges);
      }
    }
  }
}

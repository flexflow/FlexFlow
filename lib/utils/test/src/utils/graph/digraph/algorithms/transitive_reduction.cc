#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transitive_reduction") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    SUBCASE("base case") {
      std::vector<Node> n = add_nodes(g, 3);

      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(0), n.at(2)},
                });

      DiGraphView result = transitive_reduction(g);

      SUBCASE("nodes") {
        std::unordered_set<Node> result_nodes = get_nodes(result);
        std::unordered_set<Node> correct_nodes = unordered_set_of(n);
        CHECK(result_nodes == correct_nodes);
      }

      SUBCASE("edges") {
        std::unordered_set<DirectedEdge> result_edges = get_edges(result);
        std::unordered_set<DirectedEdge> correct_edges = {
            DirectedEdge{n.at(0), n.at(1)},
            DirectedEdge{n.at(1), n.at(2)},
        };
        CHECK(result_edges == correct_edges);
      }
    }

    SUBCASE("nontrivial graph") {
      // from
      // https://en.wikipedia.org/w/index.php?title=Transitive_reduction&oldid=1226082357#In_directed_acyclic_graphs

      std::vector<Node> n = add_nodes(g, 5);

      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(0), n.at(3)},
                    DirectedEdge{n.at(0), n.at(4)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(2), n.at(3)},
                    DirectedEdge{n.at(2), n.at(4)},
                    DirectedEdge{n.at(3), n.at(4)},
                });

      DiGraphView result = transitive_reduction(g);

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
            DirectedEdge{n.at(1), n.at(3)},
            DirectedEdge{n.at(2), n.at(3)},
            DirectedEdge{n.at(3), n.at(4)},
        };
        CHECK(result_edges == correct_edges);
      }
    }
  }
}

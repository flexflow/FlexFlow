#include "utils/graph/digraph/algorithms/get_dominators.h"
#include "test/utils/doctest.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_dominators") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 4);
    std::vector<DirectedEdge> e = {
        DirectedEdge{n.at(0), n.at(3)},
        DirectedEdge{n.at(0), n.at(1)},
        DirectedEdge{n.at(0), n.at(2)},
        DirectedEdge{n.at(1), n.at(2)},
    };
    add_edges(g, e);

    SUBCASE("single node") {
      Node node = n.at(2);
      std::unordered_set<Node> correct = {n.at(0), n.at(2)};
      std::unordered_set<Node> result = get_dominators(g, node);
      CHECK(correct == result);
    }

    SUBCASE("multiple nodes") {
      std::unordered_set<Node> nodes = {n.at(1), n.at(3)};
      std::unordered_set<Node> result = get_dominators(g, nodes);
      std::unordered_set<Node> correct = {n.at(0)};
      CHECK(correct == result);
    }

    SUBCASE("graph with cycles") {
      // example from
      // https://en.wikipedia.org/w/index.php?title=Dominator_(graph_theory)&oldid=1189814332

      DiGraph g = DiGraph::create<AdjacencyDiGraph>();

      std::vector<Node> n = add_nodes(g, 6);

      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(1), n.at(5)},
                    DirectedEdge{n.at(2), n.at(4)},
                    DirectedEdge{n.at(3), n.at(4)},
                    DirectedEdge{n.at(4), n.at(1)},
                });

      SUBCASE("node 1") {
        std::unordered_set<Node> result = get_dominators(g, n.at(1));
        std::unordered_set<Node> correct = {n.at(0), n.at(1)};
        CHECK(result == correct);
      }

      SUBCASE("node 3") {
        std::unordered_set<Node> result = get_dominators(g, n.at(3));
        std::unordered_set<Node> correct = {n.at(0), n.at(1), n.at(3)};
        CHECK(result == correct);
      }
    }
  }
}

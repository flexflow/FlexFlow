#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_acyclic") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    SUBCASE("empty graph") {
      CHECK(is_acyclic(g));
    }

    SUBCASE("single node") {
      add_nodes(g, 1);
      CHECK(is_acyclic(g));
    }

    SUBCASE("simple acyclic graph") {
      std::vector<Node> n = add_nodes(g, 3);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(1), n.at(2)},
                });
      CHECK(is_acyclic(g));
    }

    SUBCASE("simple cyclic graph") {
      std::vector<Node> n = add_nodes(g, 3);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(2), n.at(0)},
                });
      CHECK_FALSE(is_acyclic(g));
    }

    SUBCASE("2 parallel chains") {
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(2), n.at(4)},
                    DirectedEdge{n.at(3), n.at(5)},
                    DirectedEdge{n.at(4), n.at(5)},
                });
      CHECK(is_acyclic(g));
    }
    SUBCASE("traversal with root") {
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {DirectedEdge{n.at(0), n.at(1)},
                 DirectedEdge{n.at(1), n.at(2)},
                 DirectedEdge{n.at(2), n.at(3)},
                 DirectedEdge{n.at(3), n.at(2)}});
      CHECK_FALSE(is_acyclic(g));
    }

    SUBCASE("traversal without root") {
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {DirectedEdge{n.at(0), n.at(1)},
                 DirectedEdge{n.at(1), n.at(2)},
                 DirectedEdge{n.at(2), n.at(3)},
                 DirectedEdge{n.at(3), n.at(0)}});
      CHECK_FALSE(is_acyclic(g));
    }

    SUBCASE("traversal nonlinear") {
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {DirectedEdge{n.at(0), n.at(1)},
                 DirectedEdge{n.at(1), n.at(2)},
                 DirectedEdge{n.at(2), n.at(3)},
                 DirectedEdge{n.at(1), n.at(3)}});
      CHECK(is_acyclic(g));
    }

    SUBCASE("complex cyclic graph") {
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(2), n.at(4)},
                    DirectedEdge{n.at(3), n.at(5)},
                    DirectedEdge{n.at(4), n.at(5)},
                    DirectedEdge{n.at(5), n.at(1)},
                });
      CHECK_FALSE(is_acyclic(g));
    }

    SUBCASE("complex cyclic graph #2") {
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
      CHECK_FALSE(is_acyclic(g));
    }
  }
}

#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_acyclic - empty graph") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    bool correct = true;
    bool result = is_acyclic(g);

    CHECK(result == correct);
  }

  TEST_CASE("is_acyclic - single node") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    add_nodes(g, 1);

    bool correct = true;
    bool result = is_acyclic(g);

    CHECK(result == correct);
  }

  TEST_CASE("is_acyclic - simple acyclic graph") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 3);

    add_edges(g,
              {
                  DirectedEdge{n.at(0), n.at(1)},
                  DirectedEdge{n.at(1), n.at(2)},
              });

    bool correct = true;
    bool result = is_acyclic(g);

    CHECK(result == correct);
  }

  TEST_CASE("is_acyclic - simple cyclic graph") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 3);

    add_edges(g,
              {
                  DirectedEdge{n.at(0), n.at(1)},
                  DirectedEdge{n.at(1), n.at(2)},
                  DirectedEdge{n.at(2), n.at(0)},
              });

    bool correct = false;
    bool result = is_acyclic(g);

    CHECK(result == correct);
  }

  TEST_CASE("is_acyclic - 2 parallel chains") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

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

    bool correct = true;
    bool result = is_acyclic(g);

    CHECK(result == correct);
  }

  TEST_CASE("is_acyclic - complex cyclic graph") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

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

    bool correct = false;
    bool result = is_acyclic(g);

    CHECK(result == correct);
  }

  TEST_CASE("is_acyclic - complex acyclic graph ") {
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

    bool correct = false;
    bool result = is_acyclic(g);

    CHECK(result == correct);
  }
}

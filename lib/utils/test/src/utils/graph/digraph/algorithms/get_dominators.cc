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
        DirectedEdge{n[0], n[3]},
        DirectedEdge{n[0], n[1]},
        DirectedEdge{n[0], n[2]},
        DirectedEdge{n[1], n[2]},
    };
    add_edges(g, e);

    SUBCASE("single node") {
      std::unordered_set<Node> expected_dominators = {n[0], n[2]};
      CHECK(get_dominators(g, n[2]) == expected_dominators);
    }

    SUBCASE("multiple nodes") {
      std::unordered_set<Node> nodes = {n[1], n[3]};
      std::unordered_set<Node> expected_dominators = {n[0]};
      CHECK(get_dominators(g, nodes) == expected_dominators);
    }
  }
}

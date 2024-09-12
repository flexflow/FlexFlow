#include "utils/graph/digraph/algorithms/get_weakly_connected_components.h"
#include "test/utils/doctest.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_weakly_connected_components") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 4);

    add_edges(g, {DirectedEdge{n[0], n[1]}, DirectedEdge{n[2], n[1]}});

    std::unordered_set<std::unordered_set<Node>> correct = {
        {n[0], n[1], n[2]},
        {n[3]},
    };
    std::unordered_set<std::unordered_set<Node>> result =
        get_weakly_connected_components(g);

    CHECK(result == correct);
  }
}

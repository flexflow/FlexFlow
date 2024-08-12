#include "utils/graph/digraph/algorithms/get_weakly_connected_components.h"
#include "test/utils/doctest.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/algorithms.h"

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE(FF_TEST_SUITE) {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 4);

    std::vector<DirectedEdge> edges = {DirectedEdge{n[0], n[1]}, DirectedEdge{n[2], n[1]}};

    add_edges(g, edges);
    std::unordered_set<std::unordered_set<Node>> expected_components = {
        {n[0], n[1], n[2]},
        {n[3]},
    };

    CHECK(get_weakly_connected_components(g) == expected_components);
  }
}

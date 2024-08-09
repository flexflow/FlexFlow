#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_longest_path_lengths_from_root.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_longest_path_lengths_from_root - linear graph") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 5);
    std::vector<DirectedEdge> edges = {
        DirectedEdge{n[0], n[1]},
        DirectedEdge{n[1], n[2]},
        DirectedEdge{n[2], n[3]},
        DirectedEdge{n[3], n[4]},
    };

    add_edges(g, edges);

    std::unordered_map<Node, int> expected_lengths = {
        {n[0], 1},
        {n[1], 2},
        {n[2], 3},
        {n[3], 4},
        {n[4], 5},
    };

    CHECK(get_longest_path_lengths_from_root(g) == expected_lengths);
  }

  TEST_CASE("get_longest_path_lengths_from_root - more complex graph") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 7);
    std::vector<DirectedEdge> edges = {DirectedEdge{n[0], n[1]},
                                       DirectedEdge{n[0], n[3]},
                                       DirectedEdge{n[0], n[4]},
                                       DirectedEdge{n[0], n[6]},
                                       DirectedEdge{n[1], n[2]},
                                       DirectedEdge{n[2], n[3]},
                                       DirectedEdge{n[3], n[5]},
                                       DirectedEdge{n[4], n[5]},
                                       DirectedEdge{n[5], n[6]}};

    add_edges(g, edges);

    std::unordered_map<Node, int> expected_lengths = {
        {n[0], 1},
        {n[1], 2},
        {n[2], 3},
        {n[3], 4},
        {n[4], 2},
        {n[5], 5},
        {n[6], 6},
    };

    CHECK(get_longest_path_lengths_from_root(g) == expected_lengths);
  }
}

#include "utils/graph/digraph/algorithms.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("DiGraph - algorithms.cc") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 4);
    std::vector<DirectedEdge> e = {
        DirectedEdge{n[0], n[3]},
        DirectedEdge{n[0], n[1]},
        DirectedEdge{n[0], n[2]},
        DirectedEdge{n[1], n[2]},
    };
    add_edges(g, e);

    SUBCASE("get_edges") {
      std::unordered_set<DirectedEdge> expected_edges = unordered_set_of(e);
      std::unordered_set<DirectedEdge> actual_edges = get_edges(g);
      CHECK(actual_edges == expected_edges);
    }

    SUBCASE("get_sinks") {
      std::unordered_set<Node> expected_sinks = {n[2], n[3]};
      std::unordered_set<Node> actual_sinks = get_sinks(g);
      CHECK(actual_sinks == expected_sinks);
    }

    SUBCASE("get_sources") {
      std::unordered_set<Node> expected_sources = {n[0]};
      std::unordered_set<Node> actual_sources = get_sources(g);
      CHECK(actual_sources == expected_sources);
    }
  }
}

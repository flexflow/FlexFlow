#include <doctest/doctest.h>
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_post_dominators.h"
#include "utils/graph/instances/adjacency_digraph.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_post_dominators") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    SUBCASE("trivial sequential graph") {
      std::vector<Node> n = add_nodes(g, 2);

      g.add_edge(DirectedEdge{n.at(0), n.at(1)});

      std::unordered_map<Node, std::unordered_set<Node>> result = get_post_dominators(g);
      std::unordered_map<Node, std::unordered_set<Node>> correct = {
        {n.at(0), {n.at(0), n.at(1)}},
        {n.at(1), {n.at(1)}},
      };
    }

    SUBCASE("actual non-straight-line graph") {
      std::vector<Node> n = add_nodes(g, 9);

      add_edges(g, {
        DirectedEdge{n.at(0), n.at(1)},
        DirectedEdge{n.at(0), n.at(2)},
        DirectedEdge{n.at(1), n.at(3)},
        DirectedEdge{n.at(1), n.at(4)},
        DirectedEdge{n.at(2), n.at(4)},
        DirectedEdge{n.at(2), n.at(5)},
        DirectedEdge{n.at(3), n.at(6)},
        DirectedEdge{n.at(4), n.at(6)},
        DirectedEdge{n.at(4), n.at(7)},
        DirectedEdge{n.at(5), n.at(7)},
        DirectedEdge{n.at(6), n.at(8)},
        DirectedEdge{n.at(7), n.at(8)},
      });

      std::unordered_map<Node, std::unordered_set<Node>> result = get_post_dominators(g);
      std::unordered_map<Node, std::unordered_set<Node>> correct = {
        {n.at(0), {n.at(0), n.at(8)}},
        {n.at(1), {n.at(1), n.at(6), n.at(8)}},
        {n.at(2), {n.at(2), n.at(7), n.at(8)}},
        {n.at(3), {n.at(3), n.at(6), n.at(8)}},
        {n.at(4), {n.at(4), n.at(8)}},
        {n.at(5), {n.at(5), n.at(7), n.at(8)}},
        {n.at(6), {n.at(6), n.at(8)}},
        {n.at(7), {n.at(7), n.at(8)}},
        {n.at(8), {n.at(8)}},
      };

      CHECK(result == correct);
    }

    // TODO add a cyclic graph
  }
}

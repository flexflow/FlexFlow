#include "utils/graph/digraph/algorithms/get_outgoing_edges.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_outgoing_edges") {
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

    std::unordered_map<Node, std::unordered_set<DirectedEdge>> correct = {
        {n.at(0), {DirectedEdge{n.at(0), n.at(1)}}},
        {n.at(1),
         {DirectedEdge{n.at(1), n.at(2)},
          DirectedEdge{n.at(1), n.at(3)},
          DirectedEdge{n.at(1), n.at(5)}}},
        {n.at(2), {DirectedEdge{n.at(2), n.at(4)}}},
        {n.at(3), {DirectedEdge{n.at(3), n.at(4)}}},
        {n.at(4), {DirectedEdge{n.at(4), n.at(1)}}},
        {n.at(5), {}},
    };

    std::unordered_map<Node, std::unordered_set<DirectedEdge>> result =
        get_outgoing_edges(g, get_nodes(g));

    CHECK(result == correct);
  }
}

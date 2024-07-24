#include <doctest/doctest.h>
#include "utils/containers/get_only.h"
#include "utils/graph/digraph/algorithms/inverse_line_graph/get_inverse_line_graph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/algorithms.h"
#include "utils/containers/transform.h"
#include "utils/graph/digraph/algorithms/get_successors.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_inverse_line_graph") {
    // Tests that inverse line graph of the diamond graph
    //   b-d
    //  /   \
    // a     e
    //  \   /
    //   -c-
    //
    // is 
    //     2
    //    / \
    // 0-1   3-4
    //    \ /
    //     -
    // which means that the following is the mappings between line edges and 
    // inverse nodes is 
    // (0, 1) -> a
    // (1, 2) -> b
    // (1, 3) -> c
    // (2, 3) -> d
    // (3, 4) -> e

    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    
    std::vector<Node> n = add_nodes(g, 5);
    std::vector<DirectedEdge> es = {
      DirectedEdge{n.at(0), n.at(1)},
      DirectedEdge{n.at(0), n.at(2)},
      DirectedEdge{n.at(1), n.at(3)},
      DirectedEdge{n.at(3), n.at(4)},
      DirectedEdge{n.at(2), n.at(4)},
    };
    add_edges(g, es);

    InverseLineGraphResult result = get_inverse_line_graph(g);

    std::unordered_set<Node> result_nodes = get_nodes(result.graph);
    REQUIRE(result_nodes.size() == 5);

    std::vector<Node> inv = get_topological_ordering(result.graph);

    SUBCASE("edges") {
      std::unordered_set<DirectedEdge> result_edges = get_edges(result.graph);
      std::unordered_set<DirectedEdge> correct_edges = {
        DirectedEdge{inv.at(0), inv.at(1)},
        DirectedEdge{inv.at(1), inv.at(2)},
        DirectedEdge{inv.at(1), inv.at(3)},
        DirectedEdge{inv.at(2), inv.at(3)},
        DirectedEdge{inv.at(3), inv.at(4)},
      };
      CHECK(result_edges == correct_edges);
    }

    SUBCASE("inverse_edge_to_line_node_bidict") {
      bidict<DirectedEdge, Node> result_bidict = result.inverse_edge_to_line_node_bidict;
      bidict<DirectedEdge, Node> correct_bidict = {
        {DirectedEdge{inv.at(0), inv.at(1)}, n.at(0)},
        {DirectedEdge{inv.at(1), inv.at(2)}, n.at(1)},
        {DirectedEdge{inv.at(1), inv.at(3)}, n.at(2)},
        {DirectedEdge{inv.at(2), inv.at(3)}, n.at(3)},
        {DirectedEdge{inv.at(3), inv.at(4)}, n.at(4)},
      };
    }
  }
}

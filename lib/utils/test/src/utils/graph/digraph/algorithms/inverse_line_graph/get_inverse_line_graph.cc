#include <doctest/doctest.h>
#include "utils/containers/get_only.h"
#include "utils/graph/digraph/algorithms/inverse_line_graph/get_inverse_line_graph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/multidigraph/algorithms/get_directed_edge.h"
#include "utils/graph/multidigraph/algorithms/get_edge_counts.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/algorithms.h"
#include "utils/containers/transform.h"
#include "utils/containers/map_values.h"
#include "utils/graph/digraph/algorithms/get_successors.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_inverse_line_graph") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      
    SUBCASE("diamond-ish") {
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
        std::unordered_map<DirectedEdge, int> result_edges = get_edge_counts(result.graph);
        std::unordered_map<DirectedEdge, int> correct_edges = {
          {DirectedEdge{inv.at(0), inv.at(1)}, 1},
          {DirectedEdge{inv.at(1), inv.at(2)}, 1},
          {DirectedEdge{inv.at(1), inv.at(3)}, 1},
          {DirectedEdge{inv.at(2), inv.at(3)}, 1},
          {DirectedEdge{inv.at(3), inv.at(4)}, 1},
        };
        CHECK(result_edges == correct_edges);
      }

      SUBCASE("inverse_edge_to_line_node_bidict") {
        std::unordered_map<Node, DirectedEdge> result_bidict = map_values(result.inverse_edge_to_line_node_bidict.reversed().as_unordered_map(),
                                                                          [&](MultiDiEdge const &e) { return get_directed_edge(result.graph, e); });
        std::unordered_map<Node, DirectedEdge> correct_bidict = {
          {n.at(0), DirectedEdge{inv.at(0), inv.at(1)}},
          {n.at(1), DirectedEdge{inv.at(1), inv.at(2)}},
          {n.at(2), DirectedEdge{inv.at(1), inv.at(3)}},
          {n.at(3), DirectedEdge{inv.at(2), inv.at(3)}},
          {n.at(4), DirectedEdge{inv.at(3), inv.at(4)}},
        };
        CHECK(result_bidict == correct_bidict);
      }
    }

    SUBCASE("duplicate edges") {
      // Tests that inverse line graph of the two-node graph
      //
      // a b  (no edges)
      //
      // is 
      //   
      //  /\
      // 0  1
      //  \/
      //  
      // which means that the following is the mappings between line edges and 
      // inverse nodes is 
      // (0, 1) -> a
      // (0, 1) -> b
      std::vector<Node> n = add_nodes(g, 2);

      InverseLineGraphResult result = get_inverse_line_graph(g);

      std::unordered_set<Node> result_nodes = get_nodes(result.graph);
      REQUIRE(result_nodes.size() == 2);

      std::vector<Node> inv = get_topological_ordering(result.graph);

      SUBCASE("edges") {
        std::unordered_map<DirectedEdge, int> result_edges = get_edge_counts(result.graph);
        std::unordered_map<DirectedEdge, int> correct_edges = {
          {DirectedEdge{inv.at(0), inv.at(1)}, 2},
        };
        CHECK(result_edges == correct_edges);
      }

      SUBCASE("inverse_edge_to_line_node_bidict") {
        std::unordered_map<Node, DirectedEdge> result_bidict = map_values(result.inverse_edge_to_line_node_bidict.reversed().as_unordered_map(),
                                                                          [&](MultiDiEdge const &e) { return get_directed_edge(result.graph, e); });
        std::unordered_map<Node, DirectedEdge> correct_bidict = {
          {n.at(0), DirectedEdge{inv.at(0), inv.at(1)}},
          {n.at(1), DirectedEdge{inv.at(0), inv.at(1)}},
        };
        CHECK(result_bidict == correct_bidict);
      }
    }
  }
}

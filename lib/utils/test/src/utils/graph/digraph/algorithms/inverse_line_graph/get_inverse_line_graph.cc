#include <doctest/doctest.h>
#include "utils/containers/get_only.h"
#include "utils/graph/digraph/algorithms/inverse_line_graph/get_inverse_line_graph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/algorithms.h"
#include "utils/containers/transform.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_inverse_line_graph") {
    // Tests that inverse line graph of
    //     2
    //    / \
    // 0-1   4-5
    //    \ /
    //     3
    // is the diamong graph, a.k.a,
    //   b-d
    //  /   \
    // a     f
    //  \   /
    //   c-e
    // which means that the following is the mappings between line edges and 
    // inverse nodes is 
    // (0, 1) -> a
    // (1, 2) -> b
    // (1, 3) -> c
    // (2, 4) -> d
    // (3, 4) -> e
    // (4, 5) -> f

    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    
    std::vector<Node> n = add_nodes(g, 6);

    DirectedEdge e_a = DirectedEdge{n.at(0), n.at(1)};
    DirectedEdge e_b = DirectedEdge{n.at(1), n.at(2)};
    DirectedEdge e_c = DirectedEdge{n.at(1), n.at(3)};
    DirectedEdge e_d = DirectedEdge{n.at(2), n.at(4)};
    DirectedEdge e_e = DirectedEdge{n.at(3), n.at(4)};
    DirectedEdge e_f = DirectedEdge{n.at(4), n.at(5)};

    std::vector<DirectedEdge> es = {
      e_a,
      e_b,
      e_c,
      e_d,
      e_e,
      e_f,
    };
    add_edges(g, es);

    InverseLineGraphResult result = get_inverse_line_graph(g);
    auto get_inv_node = [&](DirectedEdge const &e) -> Node {
      return result.line_edge_to_inverse_node_bidict.at_l(e);
    };

    SUBCASE("nodes") {
      std::unordered_set<Node> result_nodes = get_nodes(result.graph);
      std::unordered_set<Node> correct_nodes = unordered_set_of(transform(es, get_inv_node));
      // one inverse node for every line edge
      CHECK(result_nodes == correct_nodes);
    }

    SUBCASE("edges") {
      std::unordered_set<DirectedEdge> result_edges = get_edges(result.graph);
      // diamond pattern edges
      std::unordered_set<DirectedEdge> correct_edges = {
        DirectedEdge{get_inv_node(e_a), get_inv_node(e_b)},
        DirectedEdge{get_inv_node(e_a), get_inv_node(e_c)},
        DirectedEdge{get_inv_node(e_b), get_inv_node(e_d)},
        DirectedEdge{get_inv_node(e_c), get_inv_node(e_e)},
        DirectedEdge{get_inv_node(e_d), get_inv_node(e_f)},
        DirectedEdge{get_inv_node(e_e), get_inv_node(e_f)},
      };
      CHECK(result_edges == correct_edges);
    }
  }
}

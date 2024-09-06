#include "utils/graph/digraph/algorithms/complete_bipartite_composite/get_cbc_decomposition.h"
#include "utils/containers/reversed.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_cbc_decomposition") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    // used to check that the cbc decomposition result is the same regardless
    // of the order in which the graph edges are processed, as this is a property
    // that should hold, and violations of this property have been a source of bugs
    // in the past
    auto check_cbc_decomposition_is_edge_order_invariant = [](DiGraphView const &g) {
      std::unordered_set<DirectedEdge> edges = get_edges(g);

      std::vector<DirectedEdge> edge_order1 = vector_of(edges);
      std::vector<DirectedEdge> edge_order2 = reversed(edge_order1);

      std::optional<CompleteBipartiteCompositeDecomposition> result1 = 
        get_cbc_decomposition_with_edge_order_internal(g, edge_order1);
      std::optional<CompleteBipartiteCompositeDecomposition> result2 = 
        get_cbc_decomposition_with_edge_order_internal(g, edge_order2);

      CHECK(result1 == result2);
    };

    SUBCASE("six-node diamond graph") {
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(2), n.at(4)},
                    DirectedEdge{n.at(3), n.at(5)},
                    DirectedEdge{n.at(4), n.at(5)},
                });

      std::optional<CompleteBipartiteCompositeDecomposition> result =
          get_cbc_decomposition(g);
      std::optional<CompleteBipartiteCompositeDecomposition> correct =
          CompleteBipartiteCompositeDecomposition{{
              BipartiteComponent{{n.at(0)}, {n.at(1), n.at(2)}},
              BipartiteComponent{{n.at(1)}, {n.at(3)}},
              BipartiteComponent{{n.at(2)}, {n.at(4)}},
              BipartiteComponent{{n.at(3), n.at(4)}, {n.at(5)}},
          }};

      CHECK(result == correct);

      check_cbc_decomposition_is_edge_order_invariant(g);
    }

    SUBCASE("graph without any edges") {
      std::vector<Node> n = add_nodes(g, 4);

      std::optional<CompleteBipartiteCompositeDecomposition> result =
          get_cbc_decomposition(g);
      std::optional<CompleteBipartiteCompositeDecomposition> correct =
          CompleteBipartiteCompositeDecomposition{{}};

      CHECK(result == correct);

      check_cbc_decomposition_is_edge_order_invariant(g);
    }

    SUBCASE("irreducible n-graph (non-cbc graph)") {
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                });

      std::optional<CompleteBipartiteCompositeDecomposition> result =
          get_cbc_decomposition(g);
      std::optional<CompleteBipartiteCompositeDecomposition> correct =
          std::nullopt;

      CHECK(result == correct);

      check_cbc_decomposition_is_edge_order_invariant(g);
    }
  }
}

#include <doctest/doctest.h>
#include "utils/graph/digraph/algorithms/complete_bipartite_composite/get_cbc_decomposition.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_cbc_decomposition") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    
    SUBCASE("six-node diamond graph") {
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g, {
        DirectedEdge{n.at(0), n.at(1)},
        DirectedEdge{n.at(0), n.at(2)},
        DirectedEdge{n.at(1), n.at(3)},
        DirectedEdge{n.at(2), n.at(4)},
        DirectedEdge{n.at(3), n.at(5)},
        DirectedEdge{n.at(4), n.at(5)},
      });
      
      std::optional<CompleteBipartiteCompositeDecomposition> result = get_cbc_decomposition(g);
      std::optional<CompleteBipartiteCompositeDecomposition> correct = CompleteBipartiteCompositeDecomposition{{
        BipartiteComponent{{n.at(0)}, {n.at(1), n.at(2)}},
        BipartiteComponent{{n.at(1)}, {n.at(3)}},
        BipartiteComponent{{n.at(2)}, {n.at(4)}},
        BipartiteComponent{{n.at(3),  n.at(4)}, {n.at(5)}},
      }};

      CHECK(result == correct);
    }
  }
}

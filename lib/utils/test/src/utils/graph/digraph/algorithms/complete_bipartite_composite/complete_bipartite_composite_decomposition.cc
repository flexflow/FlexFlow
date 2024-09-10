#include "utils/graph/digraph/algorithms/complete_bipartite_composite/complete_bipartite_composite_decomposition.h"
#include "utils/fmt/optional.h"
#include "utils/fmt/unordered_set.h"
#include "utils/hash/unordered_set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("CompleteBipartiteCompositeDecomposition") {
    BipartiteComponent bc1 = BipartiteComponent{
        {
            Node{1},
            Node{2},
        },
        {
            Node{3},
        },
    };
    BipartiteComponent bc2 = BipartiteComponent{
        {
            Node{3},
            Node{4},
        },
        {
            Node{5},
            Node{6},
            Node{7},
        },
    };
    CompleteBipartiteCompositeDecomposition cbc =
        CompleteBipartiteCompositeDecomposition{
            {bc1, bc2},
        };

    SUBCASE("get_component_containing_node_in_head") {
      std::optional<BipartiteComponent> result =
          get_component_containing_node_in_head(cbc, Node{3});
      std::optional<BipartiteComponent> correct = bc2;
      CHECK(result == correct);
    }

    SUBCASE("get_component_containing_node_in_tail") {
      std::optional<BipartiteComponent> result =
          get_component_containing_node_in_tail(cbc, Node{3});
      std::optional<BipartiteComponent> correct = bc1;
      CHECK(result == correct);
    }

    SUBCASE("get_head_subcomponents") {
      std::unordered_set<std::unordered_set<Node>> result =
          get_head_subcomponents(cbc);
      std::unordered_set<std::unordered_set<Node>> correct = {bc1.head_nodes,
                                                              bc2.head_nodes};
      CHECK(result == correct);
    }

    SUBCASE("get_tail_subcomponents") {
      std::unordered_set<std::unordered_set<Node>> result =
          get_tail_subcomponents(cbc);
      std::unordered_set<std::unordered_set<Node>> correct = {bc1.tail_nodes,
                                                              bc2.tail_nodes};
      CHECK(result == correct);
    }
  }
}

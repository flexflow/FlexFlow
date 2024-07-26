#include "utils/graph/serial_parallel/intermediate_sp_decomposition_tree.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("flatten_ast") {
    std::variant<IntermediateSpDecompositionTree, Node> input =
        IntermediateSpDecompositionTree{
            SplitType::SERIAL,
            {
                Node{1},
                IntermediateSpDecompositionTree{
                    SplitType::SERIAL,
                    {
                        Node{2},
                        Node{3},
                    },
                },
            },
        };

    std::variant<IntermediateSpDecompositionTree, Node> result =
        flatten_ast(input);
    std::variant<IntermediateSpDecompositionTree, Node> correct =
        IntermediateSpDecompositionTree{
            SplitType::SERIAL,
            {
                Node{1},
                Node{2},
                Node{3},
            },
        };

    CHECK(result == correct);
  }
}

#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"
#include <doctest/doctest.h>
#include "test/utils/doctest/fmt/unordered_multiset.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("to_final_ast (base case)") {
    std::variant<IntermediateSpDecompositionTree, Node> input = Node{1};
    SerialParallelDecomposition result = to_final_ast(input);
    SerialParallelDecomposition correct = SerialParallelDecomposition{Node{1}};
    CHECK(result == correct);
  }

  TEST_CASE("to_final_ast (serial)") {
    std::variant<IntermediateSpDecompositionTree, Node> input =
        IntermediateSpDecompositionTree{
            SplitType::SERIAL,
            {Node{1}, Node{2}},
        };
    SerialParallelDecomposition result = to_final_ast(input);
    SerialParallelDecomposition correct = SerialParallelDecomposition{
        SerialSplit{{
            Node{1},
            Node{2},
        }},
    };
  }

  TEST_CASE("to_final_ast (composite)") {
    std::variant<IntermediateSpDecompositionTree, Node> input =
        IntermediateSpDecompositionTree{
            SplitType::SERIAL,
            {
                Node{0},
                IntermediateSpDecompositionTree{
                    SplitType::SERIAL,
                    {
                        Node{1},
                        IntermediateSpDecompositionTree{
                            SplitType::PARALLEL,
                            {
                                IntermediateSpDecompositionTree{
                                    SplitType::PARALLEL,
                                    {
                                        Node{2},
                                        Node{3},
                                    },
                                },
                                Node{4},
                            },
                        },
                    },
                },
                Node{5},
            }};

    SerialParallelDecomposition result = to_final_ast(input);
    SerialParallelDecomposition correct =
        SerialParallelDecomposition{SerialSplit{{
            Node{0},
            Node{1},
            ParallelSplit{{
                Node{2},
                Node{3},
                Node{4},
            }},
            Node{5},
        }}};
    CHECK(result == correct);
  }

  TEST_CASE("get_nodes(SerialParallelDecomposition)") {
    SerialParallelDecomposition input =
        SerialParallelDecomposition{SerialSplit{{
            ParallelSplit{{
                Node{1},
                Node{2},
            }},
            Node{2},
            ParallelSplit{{
                Node{4},
                Node{5},
            }},
        }}};

    std::unordered_multiset<Node> result = get_nodes(input);
    std::unordered_multiset<Node> correct = {
        Node{1},
        Node{2},
        Node{2},
        Node{4},
        Node{5},
    };
    CHECK(result == correct);
  }

  TEST_CASE("get_nodes(SerialSplit)") {
    ParallelSplit input = ParallelSplit{{
        Node{1},
        SerialSplit{{
            Node{2},
            ParallelSplit{{
                Node{3},
                Node{4},
            }},
        }},
        SerialSplit{{
            Node{1},
            Node{6},
        }},
        Node{7},
    }};

    std::unordered_multiset<Node> result = get_nodes(input);
    std::unordered_multiset<Node> correct = {
        Node{1},
        Node{2},
        Node{3},
        Node{4},
        Node{1},
        Node{6},
        Node{7},
    };

    CHECK(result == correct);
  }

  TEST_CASE("get_nodes(ParallelSplit)") {
    ParallelSplit input = ParallelSplit{{
        Node{1},
        SerialSplit{{
            Node{2},
            Node{4},
            ParallelSplit{{
                Node{4},
                Node{5},
            }},
        }},
    }};

    std::unordered_multiset<Node> result = get_nodes(input);
    std::unordered_multiset<Node> correct = {
        Node{1},
        Node{2},
        Node{4},
        Node{4},
        Node{5},
    };

    CHECK(result == correct);
  }

  TEST_CASE("get_nodes(Node)") {
    Node input = Node{5};
    std::unordered_multiset<Node> result = get_nodes(input);
    std::unordered_multiset<Node> correct = {input};
    CHECK(result == correct);
  }
}

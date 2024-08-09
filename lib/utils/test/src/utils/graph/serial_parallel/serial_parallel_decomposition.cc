#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"
#include <doctest/doctest.h>

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
            Node{3},
            ParallelSplit{{
                Node{4},
                Node{5},
            }},
        }}};

    std::unordered_set<Node> result = get_nodes(input);
    std::unordered_set<Node> correct = {
        Node{1},
        Node{2},
        Node{3},
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
            Node{5},
            Node{6},
        }},
        Node{7},
    }};

    std::unordered_set<Node> result = get_nodes(input);
    std::unordered_set<Node> correct = {
        Node{1},
        Node{2},
        Node{3},
        Node{4},
        Node{5},
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
            Node{3},
            ParallelSplit{{
                Node{4},
                Node{5},
            }},
        }},
    }};

    std::unordered_set<Node> result = get_nodes(input);
    std::unordered_set<Node> correct = {
        Node{1},
        Node{2},
        Node{3},
        Node{4},
        Node{5},
    };

    CHECK(result == correct);
  }

  TEST_CASE("get_nodes(Node)") {
    Node input = Node{5};
    std::unordered_set<Node> result = get_nodes(input);
    std::unordered_set<Node> correct = {input};
    CHECK(result == correct);
  }

  TEST_CASE("is_empty(SerialParallelDecomposition)") {
    Node n1{1};
    Node n2{2};

    SUBCASE("Node Decomposition") {
      SerialParallelDecomposition sp{n1};
      CHECK_FALSE(is_empty(sp));
    }

    SUBCASE("Empty Serial") {
      SerialParallelDecomposition sp{SerialSplit{}};
      CHECK(is_empty(sp));
    }

    SUBCASE("Empty Parallel") {
      SerialParallelDecomposition sp{ParallelSplit{}};
      CHECK(is_empty(sp));
    }

    SUBCASE("Serial with Node") {
      SerialParallelDecomposition sp{SerialSplit{n1}};
      CHECK_FALSE(is_empty(sp));
    }

    SUBCASE("Parallel with Node") {
      SerialParallelDecomposition sp{ParallelSplit{n1}};
      CHECK_FALSE(is_empty(sp));
    }

    SUBCASE("Nested Serial") {
      SerialParallelDecomposition sp{SerialSplit{ParallelSplit{}}};
      CHECK(is_empty(sp));
    }

    SUBCASE("Nested Parallel") {
      SerialParallelDecomposition sp{ParallelSplit{SerialSplit{}}};
      CHECK(is_empty(sp));
    }

    SUBCASE("Sparse") {
      SerialSplit sp{ParallelSplit{}, ParallelSplit{SerialSplit{}}};
      CHECK(is_empty(sp));
    }

    SUBCASE("Sparse with Node") {
      SerialSplit sp{ParallelSplit{}, ParallelSplit{SerialSplit{}, n2}};
      CHECK_FALSE(is_empty(sp));
    }
  }
}

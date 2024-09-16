#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("to_final_ast (base case)") {
    std::variant<IntermediateSpDecompositionTree, Node> input = Node{1};
    SeriesParallelDecomposition result = to_final_ast(input);
    SeriesParallelDecomposition correct = SeriesParallelDecomposition{Node{1}};
    CHECK(result == correct);
  }

  TEST_CASE("to_final_ast (serial)") {
    std::variant<IntermediateSpDecompositionTree, Node> input =
        IntermediateSpDecompositionTree{
            SplitType::SERIES,
            {Node{1}, Node{2}},
        };
    SeriesParallelDecomposition result = to_final_ast(input);
    SeriesParallelDecomposition correct = SeriesParallelDecomposition{
        SeriesSplit{{
            Node{1},
            Node{2},
        }},
    };
  }

  TEST_CASE("to_final_ast (composite)") {
    std::variant<IntermediateSpDecompositionTree, Node> input =
        IntermediateSpDecompositionTree{
            SplitType::SERIES,
            {
                Node{0},
                IntermediateSpDecompositionTree{
                    SplitType::SERIES,
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

    SeriesParallelDecomposition result = to_final_ast(input);
    SeriesParallelDecomposition correct =
        SeriesParallelDecomposition{SeriesSplit{{
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

  TEST_CASE("get_nodes(SeriesParallelDecomposition)") {
    SeriesParallelDecomposition input =
        SeriesParallelDecomposition{SeriesSplit{{
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

  TEST_CASE("get_nodes(SeriesSplit)") {
    ParallelSplit input = ParallelSplit{{
        Node{1},
        SeriesSplit{{
            Node{2},
            ParallelSplit{{
                Node{3},
                Node{4},
            }},
        }},
        SeriesSplit{{
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
        SeriesSplit{{
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

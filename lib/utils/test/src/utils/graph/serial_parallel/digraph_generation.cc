#include "utils/graph/serial_parallel/digraph_generation.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("digraph_from_sp_decomposition") {
    SUBCASE("Empty") {
      SerialParallelDecomposition input =
          SerialParallelDecomposition(ParallelSplit{});
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 0);
      CHECK(num_edges(result) == 0);
    }
    SUBCASE("Complex Empty") {
      SerialParallelDecomposition input = SerialParallelDecomposition(
          ParallelSplit{SerialSplit{}, SerialSplit{ParallelSplit{}}});
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 0);
      CHECK(num_edges(result) == 0);
    }

    SUBCASE("Single Node") {
      SerialParallelDecomposition input = SerialParallelDecomposition(Node(1));
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 1);
      CHECK(num_edges(result) == 0);
    }

    SUBCASE("Simple SerialSplit") {
      SerialParallelDecomposition input =
          SerialParallelDecomposition{SerialSplit{Node(1), Node(2), Node(3)}};
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 3);
      CHECK(num_edges(result) == 2);
      CHECK(get_sources(result).size() == 1);
      CHECK(get_sinks(result).size() == 1);
    }

    SUBCASE("Simple ParallelSplit") {
      SerialParallelDecomposition input =
          SerialParallelDecomposition{ParallelSplit{Node(1), Node(2), Node(3)}};
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 3);
      CHECK(num_edges(result) == 0);
      CHECK(get_sources(result).size() == 3);
      CHECK(get_sinks(result).size() == 3);
    }

    SUBCASE("Mixed Serial-Parallel") {
      SerialParallelDecomposition input = SerialParallelDecomposition{
          SerialSplit{ParallelSplit{Node(1), Node(2)},
                      ParallelSplit{Node(3), Node(4)}}};
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 4);
      CHECK(num_edges(result) == 4);
      CHECK(get_sources(result).size() == 2);
      CHECK(get_sinks(result).size() == 2);
    }

    SUBCASE("Mixed Parallel-Serial") {
      SerialParallelDecomposition input =
          SerialParallelDecomposition{ParallelSplit{
              SerialSplit{Node(1), Node(2)}, SerialSplit{Node(3), Node(4)}}};
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 4);
      CHECK(num_edges(result) == 2);
      CHECK(get_sources(result).size() == 2);
      CHECK(get_sinks(result).size() == 2);
    }

    SUBCASE("Rhombus") {
      SerialParallelDecomposition input = SerialParallelDecomposition{
          SerialSplit{Node(1), ParallelSplit{Node(2), Node(3)}, Node(4)}};
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 4);
      CHECK(num_edges(result) == 4);
      CHECK(get_sources(result).size() == 1);
      CHECK(get_sinks(result).size() == 1);
    }

    SUBCASE("Duplicate Nodes") {
      SerialParallelDecomposition input = SerialParallelDecomposition{
          SerialSplit{Node(1), ParallelSplit{Node(1), Node(2)}, Node(1)}};
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 4);
      CHECK(num_edges(result) == 4);
      CHECK(get_sources(result).size() == 1);
      CHECK(get_sinks(result).size() == 1);
    }

    SUBCASE("Complex Graph") {
      SerialParallelDecomposition input = SerialParallelDecomposition{
          SerialSplit{ParallelSplit{SerialSplit{ParallelSplit{Node(1), Node(2)},
                                                ParallelSplit{Node(3), Node(4)},
                                                Node(5)},
                                    SerialSplit{Node(6), Node(7)}},
                      Node(8)}};

      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 8);
      CHECK(num_edges(result) == 9);
      CHECK(get_sources(result).size() == 3);
      CHECK(get_sinks(result).size() == 1);
    }
  }
}

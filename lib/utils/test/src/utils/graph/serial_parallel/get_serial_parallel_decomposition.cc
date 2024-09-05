#include "utils/graph/serial_parallel/get_serial_parallel_decomposition.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_serial_parallel_decomposition (base case)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    Node n = g.add_node();

    std::optional<SerialParallelDecomposition> result =
        get_serial_parallel_decomposition(g);
    std::optional<SerialParallelDecomposition> correct =
        SerialParallelDecomposition{n};
    CHECK(result == correct);
  }

  TEST_CASE("get_serial_parallel_decomposition (parallel)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 2);

    std::optional<SerialParallelDecomposition> result =
        get_serial_parallel_decomposition(g);
    std::optional<SerialParallelDecomposition> correct =
        SerialParallelDecomposition{ParallelSplit{
            n.at(0),
            n.at(1),
        }};
    CHECK(result == correct);
  }

  TEST_CASE("get_serial_parallel_decomposition (serial)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 2);
    g.add_edge(DirectedEdge{n.at(0), n.at(1)});

    std::optional<SerialParallelDecomposition> result =
        get_serial_parallel_decomposition(g);
    std::optional<SerialParallelDecomposition> correct =
        SerialParallelDecomposition{SerialSplit{
            n.at(0),
            n.at(1),
        }};
    CHECK(result == correct);
  }

  TEST_CASE("get_serial_parallel_decomposition (composite)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 3);
    add_edges(g,
              {
                  DirectedEdge{n.at(0), n.at(1)},
                  DirectedEdge{n.at(0), n.at(2)},
              });

    std::optional<SerialParallelDecomposition> result =
        get_serial_parallel_decomposition(g);
    std::optional<SerialParallelDecomposition> correct =
        SerialParallelDecomposition{
            SerialSplit{
                n.at(0),
                ParallelSplit{
                    n.at(1),
                    n.at(2),
                },
            },
        };
    CHECK(result == correct);
  }

  TEST_CASE("get_serial_parallel_decomposition (diamond graph)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

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

    std::optional<SerialParallelDecomposition> correct =
        SerialParallelDecomposition{SerialSplit{
            n.at(0),
            ParallelSplit{
                SerialSplit{
                    n.at(1),
                    n.at(3),
                },
                SerialSplit{
                    n.at(2),
                    n.at(4),
                },
            },
            n.at(5),
        }};

    std::optional<SerialParallelDecomposition> result =
        get_serial_parallel_decomposition(g);

    CHECK(result == correct);
  }

  TEST_CASE("get_serial_parallel_decomposition (all-to-all connection)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 4);

    add_edges(g,
              {
                  DirectedEdge{n.at(0), n.at(2)},
                  DirectedEdge{n.at(0), n.at(3)},
                  DirectedEdge{n.at(1), n.at(2)},
                  DirectedEdge{n.at(1), n.at(3)},
              });

    std::optional<SerialParallelDecomposition> correct =
        SerialParallelDecomposition{
            SerialSplit{
                ParallelSplit{
                    n.at(0),
                    n.at(1),
                },
                ParallelSplit{
                    n.at(2),
                    n.at(3),
                },
            },
        };

    std::optional<SerialParallelDecomposition> result =
        get_serial_parallel_decomposition(g);

    CHECK(result == correct);
  }

  TEST_CASE("get_serial_parallel_decomposition (non-sp graph)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 4);

    // N-graph
    add_edges(g,
              {
                  DirectedEdge{n.at(0), n.at(2)},
                  DirectedEdge{n.at(1), n.at(2)},
                  DirectedEdge{n.at(1), n.at(3)},
              });

    std::optional<SerialParallelDecomposition> correct = std::nullopt;
    std::optional<SerialParallelDecomposition> result =
        get_serial_parallel_decomposition(g);

    CHECK(result == correct);
  }

  TEST_CASE("get_serial_parallel_decomposition (transitive reduction)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 4);

    // N-graph
    add_edges(g,
              {
                  DirectedEdge{n.at(0), n.at(1)},
                  DirectedEdge{n.at(0), n.at(2)},
                  DirectedEdge{n.at(1), n.at(2)},
                  DirectedEdge{n.at(1), n.at(3)},
                  DirectedEdge{n.at(2), n.at(3)},
              });

    std::optional<SerialParallelDecomposition> correct =
        SerialParallelDecomposition{
            SerialSplit{
              n.at(0),
              n.at(1),
              n.at(2),
              n.at(3),
            },
        };
    std::optional<SerialParallelDecomposition> result =
        get_serial_parallel_decomposition(g);

    CHECK(result == correct);
  }
}

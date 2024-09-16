#include "utils/graph/series_parallel/get_series_parallel_decomposition.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_series_parallel_decomposition (base case)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    Node n = g.add_node();

    std::optional<SeriesParallelDecomposition> result =
        get_series_parallel_decomposition(g);
    std::optional<SeriesParallelDecomposition> correct =
        SeriesParallelDecomposition{n};
    CHECK(result == correct);
  }

  TEST_CASE("get_series_parallel_decomposition (parallel)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 2);

    std::optional<SeriesParallelDecomposition> result =
        get_series_parallel_decomposition(g);
    std::optional<SeriesParallelDecomposition> correct =
        SeriesParallelDecomposition{ParallelSplit{
            n.at(0),
            n.at(1),
        }};
    CHECK(result == correct);
  }

  TEST_CASE("get_series_parallel_decomposition (serial)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 2);
    g.add_edge(DirectedEdge{n.at(0), n.at(1)});

    std::optional<SeriesParallelDecomposition> result =
        get_series_parallel_decomposition(g);
    std::optional<SeriesParallelDecomposition> correct =
        SeriesParallelDecomposition{SeriesSplit{
            n.at(0),
            n.at(1),
        }};
    CHECK(result == correct);
  }

  TEST_CASE("get_series_parallel_decomposition (composite)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 3);
    add_edges(g,
              {
                  DirectedEdge{n.at(0), n.at(1)},
                  DirectedEdge{n.at(0), n.at(2)},
              });

    std::optional<SeriesParallelDecomposition> result =
        get_series_parallel_decomposition(g);
    std::optional<SeriesParallelDecomposition> correct =
        SeriesParallelDecomposition{
            SeriesSplit{
                n.at(0),
                ParallelSplit{
                    n.at(1),
                    n.at(2),
                },
            },
        };
    CHECK(result == correct);
  }

  TEST_CASE("get_series_parallel_decomposition (diamond graph)") {
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

    std::optional<SeriesParallelDecomposition> correct =
        SeriesParallelDecomposition{SeriesSplit{
            n.at(0),
            ParallelSplit{
                SeriesSplit{
                    n.at(1),
                    n.at(3),
                },
                SeriesSplit{
                    n.at(2),
                    n.at(4),
                },
            },
            n.at(5),
        }};

    std::optional<SeriesParallelDecomposition> result =
        get_series_parallel_decomposition(g);

    CHECK(result == correct);
  }

  TEST_CASE("get_series_parallel_decomposition (all-to-all connection)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 4);

    add_edges(g,
              {
                  DirectedEdge{n.at(0), n.at(2)},
                  DirectedEdge{n.at(0), n.at(3)},
                  DirectedEdge{n.at(1), n.at(2)},
                  DirectedEdge{n.at(1), n.at(3)},
              });

    std::optional<SeriesParallelDecomposition> correct =
        SeriesParallelDecomposition{
            SeriesSplit{
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

    std::optional<SeriesParallelDecomposition> result =
        get_series_parallel_decomposition(g);

    CHECK(result == correct);
  }

  TEST_CASE("get_series_parallel_decomposition (non-sp graph)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 4);

    // N-graph
    add_edges(g,
              {
                  DirectedEdge{n.at(0), n.at(2)},
                  DirectedEdge{n.at(1), n.at(2)},
                  DirectedEdge{n.at(1), n.at(3)},
              });

    std::optional<SeriesParallelDecomposition> correct = std::nullopt;
    std::optional<SeriesParallelDecomposition> result =
        get_series_parallel_decomposition(g);

    CHECK(result == correct);
  }

  TEST_CASE(
      "get_series_parallel_decomposition (requires transitive reduction)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 4);

    add_edges(g,
              {
                  DirectedEdge{n.at(0), n.at(1)},
                  DirectedEdge{n.at(0), n.at(2)},
                  DirectedEdge{n.at(1), n.at(2)},
                  DirectedEdge{n.at(1), n.at(3)},
                  DirectedEdge{n.at(2), n.at(3)},
              });

    std::optional<SeriesParallelDecomposition> correct =
        SeriesParallelDecomposition{
            SeriesSplit{
                n.at(0),
                n.at(1),
                n.at(2),
                n.at(3),
            },
        };
    std::optional<SeriesParallelDecomposition> result =
        get_series_parallel_decomposition(g);

    CHECK(result == correct);
  }
}

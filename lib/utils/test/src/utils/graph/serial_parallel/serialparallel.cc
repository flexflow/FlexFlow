#include "utils/graph/serial_parallel/serialparallel.h"
#include "test/utils/doctest.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/serial_parallel/serialparallel_internal.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  // TEST_CASE("get_serial_parallel_decomposition") {
  //   DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  //
  //   std::vector<Node> n = add_nodes(g, 6);
  //
  //   add_edges(g, {
  //     DirectedEdge{n.at(0), n.at(1)},
  //     DirectedEdge{n.at(0), n.at(2)},
  //     DirectedEdge{n.at(1), n.at(3)},
  //     DirectedEdge{n.at(2), n.at(4)},
  //     DirectedEdge{n.at(3), n.at(5)},
  //     DirectedEdge{n.at(4), n.at(5)},
  //   });
  //
  //   SerialParallelDecomposition correct = SerialParallelDecomposition{{
  //     SerialSplit{{
  //       n.at(0),
  //       ParallelSplit{{
  //         SerialSplit{{
  //           n.at(1),
  //           n.at(3),
  //         }},
  //         SerialSplit{{
  //           n.at(2),
  //           n.at(4),
  //         }},
  //       }},
  //       n.at(5),
  //     }}
  //   }};
  //
  //   SerialParallelDecomposition result =
  //   get_serial_parallel_decomposition(g);
  //
  //   CHECK(result == correct);
  // }
}

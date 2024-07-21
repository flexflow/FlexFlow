#include "test/utils/doctest.h"
#include "utils/graph/serial_parallel/serialparallel.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/serial_parallel/serialparallel_internal.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_serial_parallel_decomposition") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 6);

    add_edges(g, {
      DirectedEdge{n.at(0), n.at(1)},
      DirectedEdge{n.at(0), n.at(2)},
      DirectedEdge{n.at(1), n.at(3)},
      DirectedEdge{n.at(2), n.at(4)},
      DirectedEdge{n.at(3), n.at(5)},
      DirectedEdge{n.at(4), n.at(5)},
    });

    SerialParallelDecomposition correct = SerialParallelDecomposition{{
      SerialSplit{{
        n.at(0),
        ParallelSplit{{
          SerialSplit{{
            n.at(1),
            n.at(3),
          }},
          SerialSplit{{
            n.at(2),
            n.at(4),
          }},
        }},
        n.at(5),
      }}
    }};

    SerialParallelDecomposition result = get_serial_parallel_decomposition(g);

    CHECK(result == correct);
  }

  TEST_CASE("get_nodes(SerialParallelDecomposition)") {
    SerialParallelDecomposition input = SerialParallelDecomposition{
      SerialSplit{{
        ParallelSplit{{
          Node{1},
          Node{2},
        }},
        Node{3},
        ParallelSplit{{
          Node{4},
          Node{5},
        }},
      }}
    };

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
}

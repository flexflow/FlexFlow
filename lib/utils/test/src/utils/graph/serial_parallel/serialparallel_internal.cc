#include "test/utils/doctest.h"
#include "utils/graph/serial_parallel/serialparallel_internal.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/algorithms.h"
#include "utils/fmt/variant.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("to_final_ast (base case)") {
    std::variant<IntermediateSpDecompositionTree, Node> input = Node{1};
    SerialParallelDecomposition result = to_final_ast(input);
    SerialParallelDecomposition correct = SerialParallelDecomposition{Node{1}};
    CHECK(result == correct);
  }

  TEST_CASE("flatten_ast") {
    std::variant<IntermediateSpDecompositionTree, Node> input = IntermediateSpDecompositionTree{
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

    std::variant<IntermediateSpDecompositionTree, Node> result = flatten_ast(input);
    std::variant<IntermediateSpDecompositionTree, Node> correct = IntermediateSpDecompositionTree{
      SplitType::SERIAL,
      {
        Node{1},
        Node{2},
        Node{3},
      },
    };

    CHECK(result == correct);
  }

  TEST_CASE("to_final_ast (serial)") {
    std::variant<IntermediateSpDecompositionTree, Node> input = IntermediateSpDecompositionTree{
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
    std::variant<IntermediateSpDecompositionTree, Node> input = IntermediateSpDecompositionTree{
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
      }
    };

    SerialParallelDecomposition result = to_final_ast(input);
    SerialParallelDecomposition correct = SerialParallelDecomposition{
      SerialSplit{{
        Node{0},
        Node{1},
        ParallelSplit{{
          Node{2},
          Node{3},
          Node{4},
        }},
        Node{5},
      }}
    };
    CHECK(result == correct);
  }

  TEST_CASE("sp_decomposition (base case)") { 
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    Node n = g.add_node();
    std::variant<IntermediateSpDecompositionTree, Node> result = sp_decomposition(g);
    std::variant<IntermediateSpDecompositionTree, Node> correct = n;
    CHECK(result == correct);
  }

  // TEST_CASE("sp_decomposition (parallel)") { 
  //   DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  //   std::vector<Node> ns = add_nodes(g, 2);
  //   std::variant<IntermediateSpDecompositionTree, Node> result = sp_decomposition(g);
  //   std::variant<IntermediateSpDecompositionTree, Node> correct = IntermediateSpDecompositionTree{
  //     SplitType::PARALLEL,
  //     {ns.at(0), ns.at(1)},
  //   };
  //   CHECK(result == correct);
  // }
  //
  // TEST_CASE("sp_decomposition (serial)") {
  //   DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  //   std::vector<Node> ns = add_nodes(g, 2);
  //   g.add_edge(DirectedEdge{ns.at(0), ns.at(1)});
  //   std::variant<IntermediateSpDecompositionTree, Node> result = sp_decomposition(g);
  //   std::variant<IntermediateSpDecompositionTree, Node> correct = IntermediateSpDecompositionTree{
  //     SplitType::SERIAL,
  //     {ns.at(0), ns.at(1)},
  //   };
  // }
  //
  // TEST_CASE("sp_decomposition (composite)") {
  //   DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  //   std::vector<Node> ns = add_nodes(g, 3);
  //   add_edges(g, {
  //     DirectedEdge{ns.at(0), ns.at(1)},
  //     DirectedEdge{ns.at(0), ns.at(2)},
  //   });
  //   std::variant<IntermediateSpDecompositionTree, Node> result = sp_decomposition(g);
  //   std::variant<IntermediateSpDecompositionTree, Node> correct = IntermediateSpDecompositionTree{
  //     SplitType::SERIAL,
  //     {
  //       ns.at(0),
  //       IntermediateSpDecompositionTree{
  //           SplitType::PARALLEL, 
  //           ns.at(1),
  //           ns.at(2),
  //         }
  //       }
  //     }
  //   };
  // }
  // TEST_CASE("parallel_decomposition") {
  //   DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  //   std::vector<Node> ns = add_nodes(g, 5);
  //   add_edges(g, {
  //     DirectedEdge{ns.at(0), ns.at(2)},
  //     DirectedEdge{ns.at(1), ns.at(3)},
  //     DirectedEdge{ns.at(1), ns.at(4)},
  //     DirectedEdge{ns.at(3), ns.at(4)},
  //   });
  //
  //   IntermediateSpDecompositionTree result = parallel_decomposition(g);
  //   IntermediateSpDecompositionTree correct = IntermediateSpDecompositionTree{
  //     SplitType::PARALLEL,
  //
  // }
}

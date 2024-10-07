#include "utils/graph/dataflow_graph/algorithms/transitive_reduced_dataflow_graph/get_transitive_reduced_edges_across_split.h"
#include "utils/containers/get_only.h"
#include "utils/graph/dataflow_graph/algorithms/transitive_reduced_dataflow_graph/transitive_reduced_dataflow_graph.h"
#include "utils/graph/dataflow_graph/dataflow_graph.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_transitive_reduced_edges_across_split") {
    DataflowGraph g = DataflowGraph::create<UnorderedSetDataflowGraph>();

    auto make_series_split = [](BinarySPDecompositionTree const &lhs, BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinarySeriesSplit{lhs, rhs}};
    };

    auto make_parallel_split = [](BinarySPDecompositionTree const &lhs, BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinarySeriesSplit{lhs, rhs}};
    };

    auto make_leaf = [](Node const &n) {
      return BinarySPDecompositionTree{n};
    };

    SUBCASE("multiple nodes with edges across") {
      NodeAddedResult n1_added = g.add_node({}, 1);
      Node n1 = n1_added.node;
      DataflowOutput o1 = get_only(n1_added.outputs);

      NodeAddedResult n2_added = g.add_node({}, 1);
      Node n2 = n2_added.node;
      DataflowOutput o2 = get_only(n2_added.outputs);

      NodeAddedResult n3_added = g.add_node({o2, o1}, 1);
      Node n3 = n3_added.node;
      DataflowOutput o3 = get_only(n3_added.outputs);

      NodeAddedResult n4_added = g.add_node({o1}, 1);
      Node n4 = n4_added.node;
      DataflowOutput o4 = get_only(n4_added.outputs);

      TransitiveReducedDataflowGraphView tr_g =
          get_dataflow_graph_transitive_reduction(g);

      BinarySeriesSplit split = BinarySeriesSplit{
          make_parallel_split(make_leaf(n1), make_leaf(n2)),
          make_parallel_split(make_leaf(n3), make_leaf(n4)),
      };

      std::unordered_set<DataflowEdge> result =
          get_transitive_reduced_edges_across_split(tr_g, split);
      std::unordered_set<DataflowEdge> correct = {
          DataflowEdge{
              o1,
              DataflowInput{n3, 1},
          },
          DataflowEdge{
              o2,
              DataflowInput{n3, 0},
          },
          DataflowEdge{
              o1,
              DataflowInput{n4, 0},
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("nodes each have multiple edges across") {
      NodeAddedResult n1_added = g.add_node({}, 2);
      Node n1 = n1_added.node;
      DataflowOutput n1_o1 = n1_added.outputs.at(0);
      DataflowOutput n1_o2 = n1_added.outputs.at(1);

      NodeAddedResult n2_added = g.add_node({n1_o1, n1_o2, n1_o1}, 1);
      Node n2 = n2_added.node;

      TransitiveReducedDataflowGraphView tr_g =
          get_dataflow_graph_transitive_reduction(g);

      BinarySeriesSplit split = BinarySeriesSplit{
          make_leaf(n1), 
          make_leaf(n2),
      };

      std::unordered_set<DataflowEdge> result =
          get_transitive_reduced_edges_across_split(tr_g, split);
      std::unordered_set<DataflowEdge> correct = {
          DataflowEdge{
              n1_o1,
              DataflowInput{n2, 0},
          },
          DataflowEdge{
              n1_o2,
              DataflowInput{n2, 1},
          },
          DataflowEdge{
              n1_o1,
              DataflowInput{n2, 2},
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("does not return edges eliminated by transitive reduction") {
      NodeAddedResult n1_added = g.add_node({}, 1);
      Node n1 = n1_added.node;
      DataflowOutput o1 = get_only(n1_added.outputs);

      NodeAddedResult n2_added = g.add_node({o1}, 1);
      Node n2 = n2_added.node;
      DataflowOutput o2 = get_only(n2_added.outputs);

      NodeAddedResult n3_added = g.add_node({o1, o2}, 1);
      Node n3 = n3_added.node;
      DataflowOutput o3 = get_only(n3_added.outputs);

      NodeAddedResult n4_added = g.add_node({o2, o3}, 1);
      Node n4 = n4_added.node;
      DataflowOutput o4 = get_only(n4_added.outputs);

      TransitiveReducedDataflowGraphView tr_g =
          get_dataflow_graph_transitive_reduction(g);

      BinarySeriesSplit split = BinarySeriesSplit{
          make_series_split(make_leaf(n1), make_leaf(n2)),
          make_series_split(make_leaf(n3), make_leaf(n4)),
      };

      std::unordered_set<DataflowEdge> result =
          get_transitive_reduced_edges_across_split(tr_g, split);
      std::unordered_set<DataflowEdge> correct = {
          DataflowEdge{
              o2,
              DataflowInput{n3, 1},
          },
      };

      CHECK(result == correct);
    }
  }
}

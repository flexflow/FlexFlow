#include "utils/graph/dataflow_graph/algorithms/find_isomorphism.h"
#include "utils/containers/get_only.h"
#include "utils/graph/dataflow_graph/dataflow_graph.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("find_isomorphism(DataflowGraphView, DataflowGraphView)") {
    auto g1 = DataflowGraph::create<UnorderedSetDataflowGraph>();

    NodeAddedResult g1_n1_added = g1.add_node({}, 1);
    Node g1_n1_node = g1_n1_added.node;
    DataflowOutput g1_n1_output = get_only(g1_n1_added.outputs);

    NodeAddedResult g1_n2_added = g1.add_node({g1_n1_output}, 1);
    Node g1_n2_node = g1_n2_added.node;

    auto g2 = DataflowGraph::create<UnorderedSetDataflowGraph>();

    SUBCASE("input graphs are isomorphic") {
      NodeAddedResult g2_n1_added = g2.add_node({}, 1);
      Node g2_n1_node = g2_n1_added.node;
      DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);

      NodeAddedResult g2_n2_added = g2.add_node({g2_n1_output}, 1);
      Node g2_n2_node = g2_n2_added.node;

      std::optional<DataflowGraphIsomorphism> correct_isomorphism =
          DataflowGraphIsomorphism{
              bidict<Node, Node>{
                  {g1_n1_node, g2_n1_node},
                  {g1_n2_node, g2_n2_node},
              },
          };

      std::optional<DataflowGraphIsomorphism> result = find_isomorphism(g1, g2);

      CHECK(result == correct_isomorphism);
    }

    SUBCASE("input graphs are not isomorphic (different connectivity)") {
      NodeAddedResult g2_n1_added = g2.add_node({}, 1);
      Node g2_n1_node = g2_n1_added.node;
      DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);

      NodeAddedResult g2_n2_added =
          g2.add_node({g2_n1_output, g2_n1_output}, 1);
      Node g2_n2_node = g2_n2_added.node;

      std::optional<DataflowGraphIsomorphism> correct_isomorphism =
          std::nullopt;

      std::optional<DataflowGraphIsomorphism> result = find_isomorphism(g1, g2);

      CHECK(result == correct_isomorphism);
    }

    SUBCASE("input graphs are not isomorphic (different number of src and sink "
            "nodes)") {
      NodeAddedResult g2_n1_added = g2.add_node({}, 1);
      Node g2_n1_node = g2_n1_added.node;
      DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);

      NodeAddedResult g2_n2_added = g2.add_node({g2_n1_output}, 1);
      Node g2_n2_node = g2_n2_added.node;

      NodeAddedResult g2_n3_added = g2.add_node({}, 0);
      Node g2_n3_node = g2_n3_added.node;

      std::optional<DataflowGraphIsomorphism> correct_isomorphism =
          std::nullopt;

      std::optional<DataflowGraphIsomorphism> result = find_isomorphism(g1, g2);

      CHECK(result == correct_isomorphism);
    }

    SUBCASE("input graphs are not isomorphic (different number of internal "
            "nodes)") {
      NodeAddedResult g2_n1_added = g2.add_node({}, 1);
      Node g2_n1_node = g2_n1_added.node;
      DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);

      NodeAddedResult g2_n2_added = g2.add_node({g2_n1_output}, 1);
      Node g2_n2_node = g2_n2_added.node;
      DataflowOutput g2_n2_output = get_only(g2_n2_added.outputs);

      NodeAddedResult g2_n3_added = g2.add_node({g2_n2_output}, 1);
      Node g2_n3_node = g2_n3_added.node;

      std::optional<DataflowGraphIsomorphism> correct_isomorphism =
          std::nullopt;

      std::optional<DataflowGraphIsomorphism> result = find_isomorphism(g1, g2);

      CHECK(result == correct_isomorphism);
    }
  }
}

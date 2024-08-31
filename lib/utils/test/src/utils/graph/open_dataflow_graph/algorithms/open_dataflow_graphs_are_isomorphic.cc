#include "utils/graph/open_dataflow_graph/algorithms/open_dataflow_graphs_are_isomorphic.h"
#include "utils/containers/get_only.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("open_dataflow_graphs_are_isomorphic(OpenDataflowGraphView, "
            "OpenDataflowGraphView)") {
    auto g1 = OpenDataflowGraph::create<UnorderedSetDataflowGraph>();
    auto g2 = OpenDataflowGraph::create<UnorderedSetDataflowGraph>();

    SUBCASE("input graphs are empty") {
      bool correct = true;
      bool result = open_dataflow_graphs_are_isomorphic(g1, g2);

      CHECK(result == correct);
    }

    SUBCASE("input graphs are not empty") {
      DataflowGraphInput g1_i1 = g1.add_input();
      NodeAddedResult g1_n1_added = g1.add_node({OpenDataflowValue{g1_i1}}, 1);
      Node g1_n1_node = g1_n1_added.node;
      DataflowOutput g1_n1_output = get_only(g1_n1_added.outputs);

      NodeAddedResult g1_n2_added = g1.add_node(
          {OpenDataflowValue{g1_i1}, OpenDataflowValue{g1_n1_output}}, 1);
      Node g1_n2_node = g1_n2_added.node;

      SUBCASE("one input graph is empty") {
        bool correct = false;
        bool result = open_dataflow_graphs_are_isomorphic(g1, g2);

        CHECK(result == correct);
      }

      SUBCASE("input graphs are isomorphic") {
        DataflowGraphInput g2_i1 = g2.add_input();
        NodeAddedResult g2_n1_added =
            g2.add_node({OpenDataflowValue{g2_i1}}, 1);
        Node g2_n1_node = g2_n1_added.node;
        DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);
        NodeAddedResult g2_n2_added = g2.add_node(
            {OpenDataflowValue{g2_i1}, OpenDataflowValue{g2_n1_output}}, 1);
        Node g2_n2_node = g2_n2_added.node;

        bool correct = true;
        bool result = open_dataflow_graphs_are_isomorphic(g1, g2);

        CHECK(result == correct);
      }

      SUBCASE("input graphs are not isomorphic (different number of graph "
              "inputs)") {
        DataflowGraphInput g2_i1 = g2.add_input();
        DataflowGraphInput g2_i2 = g2.add_input();
        NodeAddedResult g2_n1_added =
            g2.add_node({OpenDataflowValue{g2_i1}}, 1);
        Node g2_n1_node = g2_n1_added.node;
        DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);
        NodeAddedResult g2_n2_added = g2.add_node(
            {OpenDataflowValue{g2_i1}, OpenDataflowValue{g2_n1_output}}, 1);
        Node g2_n2_node = g2_n2_added.node;

        bool correct = false;
        bool result = open_dataflow_graphs_are_isomorphic(g1, g2);

        CHECK(result == correct);
      }

      SUBCASE("input graphs are not isomorphic (different connectivity)") {
        DataflowGraphInput g2_i1 = g2.add_input();
        NodeAddedResult g2_n1_added =
            g2.add_node({OpenDataflowValue{g2_i1}}, 1);
        Node g2_n1_node = g2_n1_added.node;
        DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);
        NodeAddedResult g2_n2_added = g2.add_node(
            {OpenDataflowValue{g2_n1_output}, OpenDataflowValue{g2_n1_output}},
            1);
        Node g2_n2_node = g2_n2_added.node;

        bool correct = false;
        bool result = open_dataflow_graphs_are_isomorphic(g1, g2);

        CHECK(result == correct);
      }

      SUBCASE("input graphs are not isomorphic (different numbers of nodes)") {
        DataflowGraphInput g2_i1 = g2.add_input();
        NodeAddedResult g2_n1_added =
            g2.add_node({OpenDataflowValue{g2_i1}}, 1);
        Node g2_n1_node = g2_n1_added.node;
        DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);
        NodeAddedResult g2_n2_added = g2.add_node(
            {OpenDataflowValue{g2_i1}, OpenDataflowValue{g2_n1_output}}, 1);
        Node g2_n2_node = g2_n2_added.node;

        NodeAddedResult g2_n3_added = g2.add_node({}, 0);
        Node g2_n3_node = g2_n3_added.node;

        bool correct = false;
        bool result = open_dataflow_graphs_are_isomorphic(g1, g2);

        CHECK(result == correct);
      }
    }
  }
}

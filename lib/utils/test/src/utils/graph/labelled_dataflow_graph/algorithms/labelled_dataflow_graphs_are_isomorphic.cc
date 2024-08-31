#include <doctest/doctest.h>
#include "utils/containers/get_only.h"
#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/labelled_dataflow_graphs_are_isomorphic.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("labelled_dataflow_graphs_are_isomorphic(LabelledDataflowGraphView, LabelledDataflowGraphView)") {
    auto g1 = LabelledDataflowGraph<std::string, int>::create<
        UnorderedSetLabelledOpenDataflowGraph<std::string, int>>();
    auto g2 = LabelledDataflowGraph<std::string, int>::create<
        UnorderedSetLabelledOpenDataflowGraph<std::string, int>>();

    SUBCASE("duplicate labels") {
      std::string node_label = "n";
      int value_label = 1;

      NodeAddedResult g1_n1_added =
          g1.add_node(node_label, {}, {value_label});
      Node g1_n1_node = g1_n1_added.node;
      DataflowOutput g1_n1_output = get_only(g1_n1_added.outputs);

      NodeAddedResult g1_n2_added =
          g1.add_node(node_label, {}, {value_label});
      Node g1_n2_node = g1_n2_added.node;
      DataflowOutput g1_n2_output = get_only(g1_n2_added.outputs);

      NodeAddedResult g1_n3_added =
          g1.add_node(node_label,
                      {g1_n1_output, g1_n2_output},
                      {value_label});
      Node g1_n3_node = g1_n3_added.node;


      NodeAddedResult g2_n1_added =
          g2.add_node(node_label, {}, {value_label});
      Node g2_n1_node = g2_n1_added.node;
      DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);

      NodeAddedResult g2_n2_added =
          g2.add_node(node_label, {}, {value_label});
      Node g2_n2_node = g2_n2_added.node;
      DataflowOutput g2_n2_output = get_only(g2_n2_added.outputs);

      NodeAddedResult g2_n3_added =
          g2.add_node(node_label,
                      {g2_n1_output, g2_n2_output},
                      {value_label});
      Node g2_n3_node = g2_n3_added.node;

      
      bool correct = true;

      bool result = labelled_dataflow_graphs_are_isomorphic(g1, g2);

      CHECK(result == correct);
    }

    SUBCASE("non-duplicate labels") {
      std::string n1_label = "n1";
      std::string n2_label = "n2";
      std::string n3_label = "n3";
      int i1_label = 1;
      int n1_output_label = 2;
      int n2_output_label = 3;
      int n3_output_label = 4;

      NodeAddedResult g1_n1_added =
          g1.add_node(n1_label, {}, {n1_output_label});
      Node g1_n1_node = g1_n1_added.node;
      DataflowOutput g1_n1_output = get_only(g1_n1_added.outputs);

      NodeAddedResult g1_n2_added =
          g1.add_node(n2_label, {}, {n2_output_label});
      Node g1_n2_node = g1_n2_added.node;
      DataflowOutput g1_n2_output = get_only(g1_n2_added.outputs);

      NodeAddedResult g1_n3_added =
          g1.add_node(n3_label,
                      {g1_n1_output, g1_n2_output},
                      {n3_output_label});
      Node g1_n3_node = g1_n3_added.node;

      SUBCASE("input graphs are isomorphic") {
        NodeAddedResult g2_n1_added =
            g2.add_node(n1_label, {}, {n1_output_label});
        Node g2_n1_node = g2_n1_added.node;
        DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);

        NodeAddedResult g2_n2_added =
            g2.add_node(n2_label, {}, {n2_output_label});
        Node g2_n2_node = g2_n2_added.node;
        DataflowOutput g2_n2_output = get_only(g2_n2_added.outputs);

        NodeAddedResult g2_n3_added =
            g2.add_node(n3_label,
                        {g2_n1_output, g2_n2_output},
                        {n3_output_label});
        Node g2_n3_node = g2_n3_added.node;

        bool correct = true;

        bool result = 
            labelled_dataflow_graphs_are_isomorphic(g1, g2);

        CHECK(result == correct);
      }

      SUBCASE("input graphs are not isomorphic (mismatched node labels)") {
        std::string mismatched_node_label = "mismatched_node_label";

        NodeAddedResult g2_n1_added =
            g2.add_node(n1_label, {}, {n1_output_label});
        Node g2_n1_node = g2_n1_added.node;
        DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);

        NodeAddedResult g2_n2_added =
            g2.add_node(mismatched_node_label, {}, {n2_output_label});
        Node g2_n2_node = g2_n2_added.node;
        DataflowOutput g2_n2_output = get_only(g2_n2_added.outputs);

        NodeAddedResult g2_n3_added =
            g2.add_node(n3_label,
                        {g2_n1_output, g2_n2_output},
                        {n3_output_label});
        Node g2_n3_node = g2_n3_added.node;

        bool correct = false;

        bool result = labelled_dataflow_graphs_are_isomorphic(g1, g2);

        CHECK(result == correct);
      }

      SUBCASE("input graphs are not isomorphic (mismatched output label)") {
        int mismatched_output_label = 20000;

        NodeAddedResult g2_n1_added =
            g2.add_node(n1_label, {}, {n1_output_label});
        Node g2_n1_node = g2_n1_added.node;
        DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);

        NodeAddedResult g2_n2_added =
            g2.add_node(n2_label, {}, {mismatched_output_label});
        Node g2_n2_node = g2_n2_added.node;
        DataflowOutput g2_n2_output = get_only(g2_n2_added.outputs);

        NodeAddedResult g2_n3_added =
            g2.add_node(n3_label,
                        {g2_n1_output, g2_n2_output},
                        {n3_output_label});
        Node g2_n3_node = g2_n3_added.node;

        bool correct = false;

        bool result = labelled_dataflow_graphs_are_isomorphic(g1, g2);

        CHECK(result == correct);
      }

      SUBCASE("input graphs are not isomorphic (underlying unlabelled graphs not isomorphic)") {
        NodeAddedResult g2_n1_added =
            g2.add_node(n1_label, {}, {n1_output_label});
        Node g2_n1_node = g2_n1_added.node;
        DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);

        NodeAddedResult g2_n2_added =
            g2.add_node(n2_label, {}, {n2_output_label});
        Node g2_n2_node = g2_n2_added.node;
        DataflowOutput g2_n2_output = get_only(g2_n2_added.outputs);

        NodeAddedResult g2_n3_added =
            g2.add_node(n3_label,
                        {g2_n2_output, g2_n1_output},
                        {n3_output_label});
        Node g2_n3_node = g2_n3_added.node;

        bool correct = false;

        bool result = labelled_dataflow_graphs_are_isomorphic(g1, g2);

        CHECK(result == correct);
      }
    }
  }
}

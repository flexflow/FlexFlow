#include "utils/graph/labelled_open_dataflow_graph/algorithms/find_isomorphism.h"
#include "utils/containers/get_only.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("find_isomorphism") {
    auto g1 = LabelledOpenDataflowGraph<std::string, int>::create<
        UnorderedSetLabelledOpenDataflowGraph<std::string, int>>();

    auto g2 = LabelledOpenDataflowGraph<std::string, int>::create<
        UnorderedSetLabelledOpenDataflowGraph<std::string, int>>();

    SUBCASE("duplicate labels") {
      std::string node_label = "n";
      int value_label = 2;

      DataflowGraphInput g1_i1 = g1.add_input(value_label);
      NodeAddedResult g1_n1_added =
          g1.add_node(node_label, {OpenDataflowValue{g1_i1}}, {value_label});
      Node g1_n1_node = g1_n1_added.node;
      DataflowOutput g1_n1_output = get_only(g1_n1_added.outputs);
      NodeAddedResult g1_n2_added = g1.add_node(
          node_label,
          {OpenDataflowValue{g1_i1}, OpenDataflowValue{g1_n1_output}},
          {value_label});
      Node g1_n2_node = g1_n2_added.node;

      DataflowGraphInput g2_i1 = g2.add_input(value_label);
      NodeAddedResult g2_n1_added =
          g2.add_node(node_label, {OpenDataflowValue{g2_i1}}, {value_label});
      Node g2_n1_node = g2_n1_added.node;
      DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);
      NodeAddedResult g2_n2_added = g2.add_node(
          node_label,
          {OpenDataflowValue{g2_i1}, OpenDataflowValue{g2_n1_output}},
          {value_label});
      Node g2_n2_node = g2_n2_added.node;

      std::optional<OpenDataflowGraphIsomorphism> correct =
          OpenDataflowGraphIsomorphism{
              bidict<Node, Node>{
                  {g1_n1_node, g2_n1_node},
                  {g1_n2_node, g2_n2_node},
              },
              bidict<DataflowGraphInput, DataflowGraphInput>{
                  {g1_i1, g2_i1},
              },
          };

      std::optional<OpenDataflowGraphIsomorphism> result =
          find_isomorphism(g1, g2);

      CHECK(result == correct);
    }

    SUBCASE("differing labels") {
      std::string n1_label = "n1";
      std::string n2_label = "n2";
      int i1_label = 1;
      int n1_output_label = 2;
      int n2_output_label = 3;

      DataflowGraphInput g1_i1 = g1.add_input(i1_label);
      NodeAddedResult g1_n1_added =
          g1.add_node(n1_label, {OpenDataflowValue{g1_i1}}, {n1_output_label});
      Node g1_n1_node = g1_n1_added.node;
      DataflowOutput g1_n1_output = get_only(g1_n1_added.outputs);
      NodeAddedResult g1_n2_added = g1.add_node(
          n2_label,
          {OpenDataflowValue{g1_i1}, OpenDataflowValue{g1_n1_output}},
          {n2_output_label});
      Node g1_n2_node = g1_n2_added.node;

      SUBCASE("input graphs are isomorphic") {
        DataflowGraphInput g2_i1 = g2.add_input(i1_label);
        NodeAddedResult g2_n1_added = g2.add_node(
            n1_label, {OpenDataflowValue{g2_i1}}, {n1_output_label});
        Node g2_n1_node = g2_n1_added.node;
        DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);
        NodeAddedResult g2_n2_added = g2.add_node(
            n2_label,
            {OpenDataflowValue{g2_i1}, OpenDataflowValue{g2_n1_output}},
            {n2_output_label});
        Node g2_n2_node = g2_n2_added.node;

        std::optional<OpenDataflowGraphIsomorphism> correct =
            OpenDataflowGraphIsomorphism{
                bidict<Node, Node>{
                    {g1_n1_node, g2_n1_node},
                    {g1_n2_node, g2_n2_node},
                },
                bidict<DataflowGraphInput, DataflowGraphInput>{
                    {g1_i1, g2_i1},
                },
            };

        std::optional<OpenDataflowGraphIsomorphism> result =
            find_isomorphism(g1, g2);

        CHECK(result == correct);
      }

      SUBCASE("input graphs are not isomorphic (mismatched node labels)") {
        std::string mismatched_node_label = "mismatched_node_label";

        DataflowGraphInput g2_i1 = g2.add_input(i1_label);
        NodeAddedResult g2_n1_added = g2.add_node(
            n1_label, {OpenDataflowValue{g2_i1}}, {n1_output_label});
        DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);
        NodeAddedResult g2_n2_added = g2.add_node(
            mismatched_node_label,
            {OpenDataflowValue{g2_i1}, OpenDataflowValue{g2_n1_output}},
            {n2_output_label});

        std::optional<OpenDataflowGraphIsomorphism> correct = std::nullopt;

        std::optional<OpenDataflowGraphIsomorphism> result =
            find_isomorphism(g1, g2);

        CHECK(result == correct);
      }

      SUBCASE("input graphs are not isomorphic (mismatched output label)") {
        int mismatched_output_label = 20000;

        DataflowGraphInput g2_i1 = g2.add_input(i1_label);
        NodeAddedResult g2_n1_added = g2.add_node(
            n1_label, {OpenDataflowValue{g2_i1}}, {mismatched_output_label});
        DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);
        NodeAddedResult g2_n2_added = g2.add_node(
            n2_label,
            {OpenDataflowValue{g2_i1}, OpenDataflowValue{g2_n1_output}},
            {n2_output_label});

        std::optional<OpenDataflowGraphIsomorphism> correct = std::nullopt;

        std::optional<OpenDataflowGraphIsomorphism> result =
            find_isomorphism(g1, g2);

        CHECK(result == correct);
      }

      SUBCASE("input graphs are not isomorphic (mismatched input label)") {
        int mismatched_input_label = 10000;

        DataflowGraphInput g2_i1 = g2.add_input(mismatched_input_label);
        NodeAddedResult g2_n1_added = g2.add_node(
            n1_label, {OpenDataflowValue{g2_i1}}, {n1_output_label});
        DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);
        NodeAddedResult g2_n2_added = g2.add_node(
            n2_label,
            {OpenDataflowValue{g2_i1}, OpenDataflowValue{g2_n1_output}},
            {n2_output_label});

        std::optional<OpenDataflowGraphIsomorphism> correct = std::nullopt;

        std::optional<OpenDataflowGraphIsomorphism> result =
            find_isomorphism(g1, g2);

        CHECK(result == correct);
      }

      SUBCASE("input graphs are not isomorphic (underlying unlabelled graphs "
              "not isomorphic)") {
        DataflowGraphInput g2_i1 = g2.add_input(i1_label);
        NodeAddedResult g2_n1_added = g2.add_node(
            n1_label, {OpenDataflowValue{g2_i1}}, {n1_output_label});
        DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);
        NodeAddedResult g2_n2_added = g2.add_node(
            n2_label, {OpenDataflowValue{g2_n1_output}}, {n2_output_label});

        std::optional<OpenDataflowGraphIsomorphism> correct = std::nullopt;

        std::optional<OpenDataflowGraphIsomorphism> result =
            find_isomorphism(g1, g2);

        CHECK(result == correct);
      }
    }
  }
}

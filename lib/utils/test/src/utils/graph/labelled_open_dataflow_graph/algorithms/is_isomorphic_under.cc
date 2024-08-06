#include <doctest/doctest.h>
#include "utils/containers/get_only.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/is_isomorphic_under.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_isomorphic_under") {
    auto g1 = LabelledOpenDataflowGraph<std::string, int>::create<
      UnorderedSetLabelledOpenDataflowGraph<std::string, int>>();
  
    std::string n1_label = "n1";
    std::string n2_label = "n2";
    int i1_label = 1;
    int n1_output_label = 2;
    int n2_output_label = 3;

    DataflowGraphInput g1_i1 = g1.add_input(i1_label);
    NodeAddedResult g1_n1_added = g1.add_node(n1_label,
                                              {OpenDataflowValue{g1_i1}},
                                              {n1_output_label});
    Node g1_n1_node = g1_n1_added.node;
    DataflowOutput g1_n1_output = get_only(g1_n1_added.outputs);
    NodeAddedResult g1_n2_added = g1.add_node(n2_label, 
                                              {OpenDataflowValue{g1_i1}, OpenDataflowValue{g1_n1_output}},
                                              {n2_output_label});
    Node g1_n2_node = g1_n2_added.node;

    auto g2 = LabelledOpenDataflowGraph<std::string, int>::create<
      UnorderedSetLabelledOpenDataflowGraph<std::string, int>>();

    DataflowGraphInput g2_i1 = g2.add_input(i1_label);
    NodeAddedResult g2_n1_added = g2.add_node(n1_label,
                                              {OpenDataflowValue{g2_i1}},
                                              {n1_output_label});
    Node g2_n1_node = g2_n1_added.node;
    DataflowOutput g2_n1_output = get_only(g2_n1_added.outputs);
    NodeAddedResult g2_n2_added = g2.add_node(n2_label, 
                                              {OpenDataflowValue{g2_i1}, OpenDataflowValue{g2_n1_output}},
                                              {n2_output_label});
    Node g2_n2_node = g2_n2_added.node;
  
    
    OpenDataflowGraphIsomorphism correct_isomorphism = OpenDataflowGraphIsomorphism{
      bidict<Node, Node>{
        {g1_n1_node, g2_n1_node},
        {g1_n2_node, g2_n2_node},
      },
      bidict<DataflowGraphInput, DataflowGraphInput>{
        {g1_i1, g2_i1},
      },
    };

    bool result = is_isomorphic_under(g1, g2, correct_isomorphism);

    CHECK(result);
  }
}

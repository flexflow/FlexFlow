#include "utils/graph/open_dataflow_graph/algorithms/get_unused_open_dataflow_graph_inputs.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_unused_open_dataflow_graph_inputs(OpenDataflowGraphView)") {
    auto g = OpenDataflowGraph::create<UnorderedSetDataflowGraph>();
    SUBCASE("unused inputs exist") {
      DataflowGraphInput g_i1 = g.add_input();
      DataflowGraphInput g_i2 = g.add_input();
      DataflowGraphInput g_i3 = g.add_input();

      NodeAddedResult g_n1_added = g.add_node({OpenDataflowValue{g_i2}}, 1);

      std::unordered_set<DataflowGraphInput> result =
          get_unused_open_dataflow_graph_inputs(g);

      std::unordered_set<DataflowGraphInput> correct = {g_i1, g_i3};

      CHECK(result == correct);
    }

    SUBCASE("unused inputs don't exist") {
      DataflowGraphInput g_i1 = g.add_input();
      DataflowGraphInput g_i2 = g.add_input();

      NodeAddedResult g_n1_added =
          g.add_node({OpenDataflowValue{g_i1}, OpenDataflowValue{g_i2}}, 1);

      std::unordered_set<DataflowGraphInput> result =
          get_unused_open_dataflow_graph_inputs(g);

      std::unordered_set<DataflowGraphInput> correct = {};

      CHECK(result == correct);
    }
  }
}

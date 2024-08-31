#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_graph_inputs.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_open_dataflow_graph_inputs(OpenDataflowGraphView)") {
    OpenDataflowGraph g =
        OpenDataflowGraph::create<UnorderedSetDataflowGraph>();

    DataflowGraphInput i0 = g.add_input();
    DataflowGraphInput i1 = g.add_input();

    NodeAddedResult n0_added = g.add_node({}, 1);

    std::unordered_set<DataflowGraphInput> result =
        get_open_dataflow_graph_inputs(g);
    std::unordered_set<DataflowGraphInput> correct = {i0, i1};

    CHECK(result == correct);
  }
}

#include "utils/graph/open_dataflow_graph/algorithms/permute_input_ids.h"
#include "utils/containers/get_only.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_graph_data.h"
#include "utils/graph/open_dataflow_graph/algorithms/open_dataflow_graph_data.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("permute_input_ids(OpenDataflowGraphView, "
            "bidict<NewDataflowGraphInput, DataflowGraphInput>)") {
    OpenDataflowGraph g =
        OpenDataflowGraph::create<UnorderedSetDataflowGraph>();

    DataflowGraphInput i0 = g.add_input();
    DataflowGraphInput i1 = g.add_input();

    NodeAddedResult n0_added = g.add_node({OpenDataflowValue{i0}}, 1);
    Node n0 = n0_added.node;
    DataflowOutput n0_output = get_only(n0_added.outputs);

    NodeAddedResult n1_added = g.add_node({OpenDataflowValue{n0_output}}, 1);
    Node n1 = n1_added.node;
    DataflowOutput n1_output = get_only(n1_added.outputs);

    DataflowGraphInput new_i0 = DataflowGraphInput{6};
    DataflowGraphInput new_i1 = DataflowGraphInput{7};

    bidict<NewDataflowGraphInput, DataflowGraphInput> input_mapping = {
        {NewDataflowGraphInput{new_i0}, i0},
        {NewDataflowGraphInput{new_i1}, i1},
    };

    OpenDataflowGraphView result = permute_input_ids(g, input_mapping);
    OpenDataflowGraphData result_data = get_graph_data(result);

    OpenDataflowGraphData correct_data = OpenDataflowGraphData{
        {n0, n1},
        {
            OpenDataflowEdge{
                DataflowInputEdge{
                    new_i0,
                    DataflowInput{
                        n0,
                        0,
                    },
                },
            },
            OpenDataflowEdge{
                DataflowEdge{
                    DataflowOutput{
                        n0,
                        0,
                    },
                    DataflowInput{
                        n1,
                        0,
                    },
                },
            },
        },
        {new_i0, new_i1},
        {
            DataflowOutput{
                n0,
                0,
            },
            DataflowOutput{
                n1,
                0,
            },
        },
    };

    CHECK(result_data == correct_data);
  }
}

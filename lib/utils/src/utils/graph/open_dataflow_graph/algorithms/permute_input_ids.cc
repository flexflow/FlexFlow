#include "utils/graph/open_dataflow_graph/algorithms/permute_input_ids.h"
#include "utils/containers/transform.h"
#include "utils/graph/open_dataflow_graph/algorithms/from_open_dataflow_graph_data.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_graph_data.h"
#include "utils/overload.h"

namespace FlexFlow {

OpenDataflowGraphView permute_input_ids(
    OpenDataflowGraphView const &g,
    bidict<NewDataflowGraphInput, DataflowGraphInput> const &input_mapping) {
  auto new_input_from_old =
      [&](DataflowGraphInput const &old_input) -> DataflowGraphInput {
    return input_mapping.at_r(old_input).raw_input;
  };

  auto new_edge_from_old = [&](OpenDataflowEdge const &e) {
    return e.visit<OpenDataflowEdge>(overload{
        [&](DataflowInputEdge const &input_e) {
          return OpenDataflowEdge{
              DataflowInputEdge{
                  new_input_from_old(input_e.src),
                  input_e.dst,
              },
          };
        },
        [&](DataflowEdge const &standard_e) {
          return OpenDataflowEdge{standard_e};
        },
    });
  };

  OpenDataflowGraphData old_data = get_graph_data(g);
  OpenDataflowGraphData permuted_data = OpenDataflowGraphData{
      old_data.nodes,
      transform(old_data.edges, new_edge_from_old),
      transform(old_data.inputs, new_input_from_old),
      old_data.outputs,
  };

  return from_open_dataflow_graph_data(permuted_data);
}

} // namespace FlexFlow

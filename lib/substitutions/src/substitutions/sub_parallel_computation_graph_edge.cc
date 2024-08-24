#include "substitutions/sub_parallel_computation_graph_edge.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"

namespace FlexFlow {

SubParallelComputationGraphEdge subpcg_edge_from_tensor_and_dst(parallel_tensor_guid_t const &tensor, 
                                                                parallel_layer_guid_t const &layer,
                                                                int input_idx) {
  return SubParallelComputationGraphEdge{
    OpenDataflowEdge{
      DataflowEdge{
        tensor.raw_graph_output,
        DataflowInput{
          layer.raw_graph_node,
          input_idx,
        },
      },
    },
  };
}

SubParallelComputationGraphEdge subpcg_edge_from_tensor_and_use(open_parallel_tensor_guid_t const &tensor,
                                                                parallel_tensor_use_t const &use) {
  return SubParallelComputationGraphEdge{
    open_dataflow_edge_from_src_and_dst(tensor.raw_open_dataflow_value, use.raw_dataflow_input),
  };
}

open_parallel_tensor_guid_t get_parallel_tensor(SubParallelComputationGraphEdge const &e) {
  OpenDataflowValue raw_value = get_open_dataflow_edge_source(e.raw_edge);
  return open_parallel_tensor_guid_t{raw_value};
}

} // namespace FlexFlow

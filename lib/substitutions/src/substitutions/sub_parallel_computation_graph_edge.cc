#include "substitutions/sub_parallel_computation_graph_edge.h"

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

} // namespace FlexFlow

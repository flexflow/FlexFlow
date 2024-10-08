#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"

namespace FlexFlow {

UnmappedOpCostEstimateKey get_unmapped_op_cost_estimate_key_for_layer(
    ParallelComputationGraph const &pcg, parallel_layer_guid_t const &layer) {
  auto get_tensor_shape = [&](parallel_tensor_guid_t const &t) {
    return get_parallel_tensor_shape(pcg, t);
  };

  return UnmappedOpCostEstimateKey{
      /*op_attrs=*/pcg_get_op_attrs(pcg, layer),
      /*input_shapes=*/
      transform(get_incoming_inputs(pcg, layer), get_tensor_shape),
      /*weight_shapes=*/
      transform(get_incoming_weights(pcg, layer), get_tensor_shape),
      /*output_shapes=*/
      transform(get_layer_outputs(pcg, layer), get_tensor_shape),
  };
}

OpCostEstimateKey
    map_unmapped_op_cost_estimate_key(UnmappedOpCostEstimateKey const &unmapped,
                                      MachineView const &machine_view) {
  return OpCostEstimateKey{
      /*op_attrs=*/unmapped.op_attrs,
      /*input_shapes=*/unmapped.input_shapes,
      /*weight_shapes=*/unmapped.weight_shapes,
      /*output_shapes=*/unmapped.output_shapes,
      /*machine_view=*/machine_view,
  };
}

} // namespace FlexFlow

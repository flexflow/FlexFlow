#include "compiler/machine_mapping/estimate_layer_cost.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"

namespace FlexFlow {

float estimate_layer_cost(CostEstimator const &cost_estimator,
                          PCGOperatorAttrs const &layer,
                          MachineView const &machine_view) {
  PCGOperatorAttrs op_attrs = get_parallel_layer_attrs(pcg, layer).op_attrs;

  auto get_tensor_shape = [&](parallel_tensor_guid_t const &t) { 
    return get_parallel_tensor_shape(pcg, t);
  };

  std::vector<parallel_tensor_guid_t> input_tensors = get_incoming_inputs(pcg, layer);
  std::vector<parallel_tensor_guid_t> weight_tensors = get_incoming_weights(pcg, layer);
  std::vector<parallel_tensor_guid_t> output_tensors = get_layer_outputs(pcg, layer);

  OpCostEstimateKey key = OpCostEstimateKey{
    /*op_attrs=*/op_attrs,
    /*input_shapes=*/transform(input_tensors, get_tensor_shape),
    /*weight_shapes=*/transform(weight_tensors, get_tensor_shape),
    /*output_shapes=*/transform(output_tensors, get_tensor_shape),
    /*machine_view=*/machine_view,
  };

  return cost_estimator.estimate_cost(key);
}

} // namespace FlexFlow

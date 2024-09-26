#include "compiler/machine_mapping/estimate_layer_cost.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"

namespace FlexFlow {

float estimate_layer_cost(ParallelComputationGraph const &pcg,
                          CostEstimator const &cost_estimator,
                          parallel_layer_guid_t const &layer,
                          MachineView const &machine_view) {
  PCGOperatorAttrs op_attrs = get_parallel_layer_attrs(pcg, layer).op_attrs;

  auto get_tensor_shape = [&](parallel_tensor_guid_t const &t) { 
    return get_parallel_tensor_shape(pcg, t);
  };

  std::vector<parallel_tensor_guid_t> input_tensors = get_incoming_inputs(pcg, layer);
  std::vector<parallel_tensor_guid_t> weight_tensors = get_incoming_weights(pcg, layer);
  std::vector<parallel_tensor_guid_t> output_tensors = get_layer_outputs(pcg, layer);

  return cost_estimator.estimate_cost(op_attrs,
                                      transform(input_tensors, get_tensor_shape),
                                      transform(weight_tensors, get_tensor_shape),
                                      transform(output_tensors, get_tensor_shape),
                                      machine_view);
}

} // namespace FlexFlow

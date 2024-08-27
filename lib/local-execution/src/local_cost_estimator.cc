#include "local-execution/local_cost_estimator.h"
#include "kernels/device.h"
#include "kernels/local_cuda_allocator.h"
#include "local-execution/tracked_allocator.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/parallel_tensor_attrs.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

static float get_total_elapsed_time(PerLayerElapsedTime const &fwd,
                                    PerLayerElapsedTime const &bwd) {
  float total_elapsed_time = 0;
  for (auto const &layer_elapsed_time : fwd) {
    layer_guid_t layer_id = layer_elapsed_time.first;
    float fwd_time = layer_elapsed_time.second.value();
    float bwd_time = bwd.at(layer_id).value();
    total_elapsed_time += fwd_time + bwd_time;
  }
  return total_elapsed_time;
}

LocalCostEstimator::LocalCostEstimator(RuntimeArgConfig const &config)
    : runtime_arg_config(config) {}

CostDetails LocalCostEstimator::estimate_cost(
    PCGOperatorAttrs const &op,
    std::vector<ParallelTensorShape> const &inputs,
    std::vector<ParallelTensorAttrs> const &weights,
    std::vector<ParallelTensorAttrs> const &outputs,
    MachineView const &mv) const {

  if (is_parallel_op(op) || op.has<InputAttrs>() || op.has<NoopAttrs>() ||
      op.has<WeightAttrs>()) {
    return CostDetails{0, 0};
  }

  LayerAttrs layer_attrs =
      LayerAttrs{compgraph_op_attrs_from_pcg_op_attrs(op), std::nullopt};

  // allocate memory for inputs
  std::shared_ptr<TrackedAllocator> tracked_allocator_ptr =
      std::make_shared<TrackedAllocator>(create_local_cuda_memory_allocator());
  Allocator allocator = Allocator(tracked_allocator_ptr);
  TensorBackingMap tensor_backing_map;
  std::vector<tensor_guid_t> input_tensor_ids;

  ComputationGraphBuilder cg_builder;
  for (ParallelTensorShape const &input : inputs) {
    TensorShape tensor_shape = get_piece_shape(input);
    tensor_guid_t tensor_id =
        cg_builder.create_tensor(tensor_shape, CreateGrad::YES);
    GenericTensorAccessorW tensor_backing =
        allocator.allocate_tensor(tensor_shape);
    tensor_backing_map.insert({tensor_id, tensor_backing});
    input_tensor_ids.push_back(tensor_id);
  }

  auto get_vector_piece_attrs =
      [](std::vector<ParallelTensorAttrs> const &parallel_attrs) {
        return transform(parallel_attrs, [](ParallelTensorAttrs const &p) {
          return get_piece_attrs(p);
        });
      };

  // add operator to graph
  std::vector<tensor_guid_t> output_tensor_ids =
      cg_builder.add_layer(layer_attrs,
                           input_tensor_ids,
                           get_vector_piece_attrs(weights),
                           get_vector_piece_attrs(outputs));

  LocalTrainingBacking local_backing(allocator,
                                     cg_builder.computation_graph,
                                     tensor_backing_map,
                                     this->runtime_arg_config,
                                     std::nullopt);

  local_backing.execute_init();
  PerLayerElapsedTime fwd = local_backing.execute_forward();
  PerLayerElapsedTime bwd = local_backing.execute_backward();

  return CostDetails{get_total_elapsed_time(fwd, bwd),
                     tracked_allocator_ptr->get_current_mem_usage()};
}

float LocalCostEstimator::estimate_cost(ParallelTensorShape const &tensor_shape,
                                        MachineView const &src,
                                        MachineView const &dst) const {
  // TODO: model communication cost analytically
  // https://github.com/flexflow/FlexFlow/issues/1414
  // temporarily return 0

  return 0.0;
}

CostEstimator
    get_local_cost_estimator(RuntimeArgConfig const &runtime_arg_config) {
  return CostEstimator::create<LocalCostEstimator>(runtime_arg_config);
}

} // namespace FlexFlow

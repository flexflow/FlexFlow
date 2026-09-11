#include "local-execution/cost_estimator/local_cost_estimator.h"
#include "compiler/machine_mapping/machine_view.dtg.h"
#include "kernels/create_local_allocator_for_device_type.h"
#include "kernels/device.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/local_cuda_allocator.h"
#include "local-execution/computation_graph_instance.h"
#include "local-execution/cost_estimator/tracked_allocator.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph/layer_added_result.dtg.h"
#include "pcg/parallel_tensor_attrs.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/map_values.h"
#include "utils/containers/maximum.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/set_of.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/exception.h"
#include "utils/optional.h"
#include <optional>

namespace FlexFlow {

LocalCostEstimator::LocalCostEstimator(
    MachineInterconnectSpecification const &interconnect_specification,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &device_handle,
    global_device_id_t device_idx)
    : interconnect_specification(interconnect_specification),
      allocator(allocator), profiling_settings(profiling_settings),
      device_handle(device_handle), device_idx(device_idx) {}

static ComputationGraph computation_graph_for_local_cost_estimation(
    ComputationGraphOpAttrs const &op,
    std::map<TensorSlotName, ParallelTensorShape> const &inputs,
    std::map<TensorSlotName, ParallelTensorShape> const &weights,
    std::map<TensorSlotName, ParallelTensorShape> const &outputs) {
  ComputationGraph computation_graph = make_empty_computation_graph();

  std::map<TensorSlotName, tensor_guid_t> input_tensors =
      map_values(inputs, [&](ParallelTensorShape const &shape) {
        LayerAddedResult inputs_layer =
            add_layer(computation_graph,
                      LayerAttrs{ComputationGraphOpAttrs{
                                     InputAttrs{get_piece_shape(shape)}},
                                 std::nullopt},
                      {},
                      {});
        return require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);
      });

  std::map<TensorSlotName, tensor_guid_t> weight_tensors =
      map_values(weights, [&](ParallelTensorShape const &shape) {
        LayerAddedResult weights_layer =
            add_layer(computation_graph,
                      LayerAttrs{ComputationGraphOpAttrs{WeightAttrs{
                                     get_piece_shape(shape),
                                     InitializerAttrs{ZeroInitializerAttrs{}}}},
                                 std::nullopt},
                      {},
                      {});
        return require_only_key(weights_layer.outputs, TensorSlotName::OUTPUT);
      });

  // create operator layer
  LayerAddedResult operator_layer = add_layer(computation_graph,
                                              LayerAttrs{
                                                  /*op_attrs=*/op,
                                                  /*name=*/"operator",
                                              },
                                              input_tensors,
                                              weight_tensors);

  return computation_graph;
}

OpCostMetrics LocalCostEstimator::estimate_cost(
    OpCostEstimateKey const &op_cost_estimate_key) const {

  PCGOperatorAttrs op = op_cost_estimate_key.op_attrs;
  std::map<TensorSlotName, ParallelTensorShape> inputs =
      op_cost_estimate_key.input_shapes;
  std::map<TensorSlotName, ParallelTensorShape> weights =
      op_cost_estimate_key.weight_shapes;
  std::map<TensorSlotName, ParallelTensorShape> outputs =
      op_cost_estimate_key.output_shapes;
  OptimizerAttrs optimizer_attrs = op_cost_estimate_key.optimizer_attrs;

  if (is_parallel_op(op) || op.has<InputAttrs>() || op.has<NoopAttrs>() ||
      op.has<WeightAttrs>()) {
    return OpCostMetrics{
        /*forward_runtime=*/0_ms,
        /*backward_runtime=*/0_ms,
        /*memory=*/0_bytes,
    };
  }

  // allocate memory
  std::shared_ptr<TrackedAllocator> tracked_allocator_ptr =
      std::make_shared<TrackedAllocator>(
          create_local_allocator_for_device_type(this->device_idx.device_type));

  layer_guid_t layer_guid = layer_guid_t{Node{0}};

  Allocator allocator = Allocator(tracked_allocator_ptr);

  ComputationGraph cg = computation_graph_for_local_cost_estimation(
      /*op=*/assert_unwrap(compgraph_op_attrs_from_pcg_op_attrs(op)),
      /*inputs=*/inputs,
      /*weights=*/weights,
      /*outputs=*/outputs);

  ComputationGraphInstance instance = create_computation_graph_instance(
      /*compgraph=*/cg,
      /*optimizer_attrs=*/optimizer_attrs,
      /*loss=*/std::nullopt,
      /*input_tensors=*/{},
      /*allocator=*/allocator,
      /*device_handle=*/this->device_handle,
      /*device_idx=*/this->device_idx);

  // execute layer
  dynamic_layer_guid_t operator_layer_guid{get_layer_by_name(cg, "operator")};

  std::map<dynamic_layer_guid_t, std::optional<milliseconds_t>> fwd_timing =
      perform_forward_pass_for_computation_graph_instance(
          instance,
          this->profiling_settings,
          this->device_handle,
          this->device_idx);
  milliseconds_t fwd = fwd_timing.at(operator_layer_guid).value();
  std::map<dynamic_layer_guid_t, std::optional<milliseconds_t>> bwd_timing =
      perform_backward_pass_for_computation_graph_instance(
          instance,
          this->profiling_settings,
          this->device_handle,
          this->device_idx);
  milliseconds_t bwd = bwd_timing.at(operator_layer_guid).value();

  return OpCostMetrics{
      /*forward_runtime=*/fwd,
      /*backward_runtime=*/bwd,
      /*memory=*/tracked_allocator_ptr->get_current_mem_usage(),
  };
}

milliseconds_t LocalCostEstimator::estimate_cost(
    TensorSetMovement const &tensor_set_movement) const {
  auto estimate_single_comm_cost =
      [&](MachineSpaceCoordinate const &src,
          MachineSpaceCoordinate const &dst,
          num_bytes_t num_bytes) -> milliseconds_t {
    if (src == dst) {
      return 0_ms;
    } else if (src.node_idx == dst.node_idx) {
      return (num_bytes /
              this->interconnect_specification.intra_node_bandwidth);
    } else {
      return (num_bytes /
              this->interconnect_specification.inter_node_bandwidth);
    }
  };

  return maximum(
      transform(set_of(tensor_set_movement.edge_to_size),
                [&](std::pair<CommunicationEdge, num_bytes_t> const &p) {
                  return estimate_single_comm_cost(
                      p.first.get_src(), p.first.get_dst(), p.second);
                }));
}

CostEstimator get_local_cost_estimator(
    MachineInterconnectSpecification const &interconnect_specification,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &device_handle,
    global_device_id_t device_idx) {
  return CostEstimator::create<LocalCostEstimator>(interconnect_specification,
                                                   allocator,
                                                   profiling_settings,
                                                   device_handle,
                                                   device_idx);
}

} // namespace FlexFlow

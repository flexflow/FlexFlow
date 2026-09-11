#include "local-execution/computation_graph_instance.h"
#include "local-execution/per_device_op_state_initialization.h"
#include "local-execution/task_execution.h"
#include "local-execution/tensor_allocation.h"
#include "pcg/optimizer_attrs.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/loss_insertion.h"
#include "task-spec/dynamic_graph/make_dynamic_open_dataflow_graph_from_cg.h"
#include "task-spec/dynamic_graph/pass_expansion.h"
#include "task-spec/dynamic_graph/update_insertion.h"
#include "task-spec/per_device_op_state.h"
#include "task-spec/task_argument_accessor/task_argument_accessor.h"
#include "utils/containers/map_from_pairs.h"
#include "utils/containers/transform.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/optional.h"
#include <optional>

namespace FlexFlow {

ComputationGraphInstance::ComputationGraphInstance(
    std::vector<DynamicNodeInvocation> const &execution_order,
    Allocator &allocator,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<GenericTensorAccessorW> logit_grad_tensor)
    : execution_order(execution_order), allocator(allocator),
      optimizer_attrs(optimizer_attrs), logit_grad_tensor(logit_grad_tensor) {}

std::vector<DynamicNodeInvocation> const &
    ComputationGraphInstance::get_execution_order() const {
  return this->execution_order;
}
Allocator &ComputationGraphInstance::get_allocator() const {
  return this->allocator;
}
OptimizerAttrs const &ComputationGraphInstance::get_optimizer_attrs() const {
  return this->optimizer_attrs;
}
void ComputationGraphInstance::update_optimizer_attrs_for_next_iter() {
  this->optimizer_attrs =
      get_optimizer_attrs_for_next_iter(this->optimizer_attrs);
}
std::optional<GenericTensorAccessorR>
    ComputationGraphInstance::get_loss_tensor_accessor() const {
  return this->logit_grad_tensor;
}

static GenericTensorAccessorW
    get_loss_tensor_accessor(DynamicOpenDataflowGraph const &dg,
                             DynamicValueAttrs const &value) {
  std::optional<DynamicTensorAccessor> accessor =
      assert_unwrap(find_output_value_attrs(dg, value.tensor_guid, value.role))
          .accessor;
  return assert_unwrap(accessor).require_write();
}

ComputationGraphInstance create_computation_graph_instance(
    ComputationGraph const &cg,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<LossConfig> const &loss,
    std::map<DynamicValueAttrs, DynamicTensorAccessor> const &input_tensors,
    Allocator &allocator,
    device_handle_t const &device_handle,
    global_device_id_t device_idx) {
  DynamicOpenDataflowGraph dg = make_dynamic_open_dataflow_graph_from_cg(cg);
  dg = perform_pass_expansion(dg);

  std::map<DynamicValueAttrs, DynamicTensorAccessor> inputs = input_tensors;
  std::optional<DynamicValueAttrs> logit_grad_value;
  if (loss.has_value()) {
    auto [loss_attrs, label_tensor, logit_tensor] = assert_unwrap(loss);
    auto [loss_inserted_dg, label_v, logit_grad_v] = perform_loss_insertion(
        dg, loss_attrs, dynamic_tensor_guid_t{logit_tensor}, std::nullopt);
    dg = loss_inserted_dg;
    logit_grad_value = logit_grad_v;
    inputs.insert(std::pair{label_v, label_tensor});
  }

  dg = perform_update_insertion(dg, optimizer_attrs);
  dg = perform_tensor_allocation(dg, inputs, allocator);

  std::optional<GenericTensorAccessorW> logit_grad_tensor =
      transform(logit_grad_value, [&](DynamicValueAttrs const &lgv) {
        return get_loss_tensor_accessor(dg, lgv);
      });

  dg = perform_per_device_op_state_initialization(
      dg, allocator, device_handle, optimizer_attrs, device_idx);

  // Compute the topological ordering of the graph
  auto [kwarg_graph, node_map] =
      labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph(dg);
  std::vector<Node> node_topo_order = get_topological_ordering(kwarg_graph);
  std::vector<DynamicNodeInvocation> invocation_topo_order = transform(
      node_topo_order, [&](Node node) { return node_map.at_l(node); });

  return ComputationGraphInstance{
      invocation_topo_order, allocator, optimizer_attrs, logit_grad_tensor};
}

static std::map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    execute_dynamic_node_invocation_set(
        std::vector<DynamicNodeInvocation> const &invocations,
        Allocator &allocator,
        OptimizerAttrs const &optimizer_attrs,
        std::optional<ProfilingSettings> const &profiling_settings,
        device_handle_t const &ff_handle,
        global_device_id_t device_idx) {
  return map_from_pairs(
      transform(invocations, [&](DynamicNodeInvocation const &invocation) {
        std::optional<milliseconds_t> timing = execute_dynamic_node_invocation(
            /*invocation=*/invocation,
            /*allocator=*/allocator,
            /*profiling_settings=*/profiling_settings,
            /*ff_handle=*/ff_handle,
            /*per_device_op_state=*/
            transform(invocation.node_attrs.per_device_op_state,
                      [&](DeviceSpecificPerDeviceOpState const &op_state) {
                        return get_per_device_op_state_from_device_specific(
                            op_state, device_idx);
                      }),
            /*optimizer_attrs=*/optimizer_attrs,
            /*device_idx=*/device_idx);
        return std::pair{invocation.node_attrs.layer_guid, timing};
      }));
}

std::map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_all_passes_for_computation_graph_instance(
        ComputationGraphInstance &instance,
        std::optional<ProfilingSettings> const &profiling_settings,
        device_handle_t const &ff_handle,
        global_device_id_t device_idx) {
  std::vector<DynamicNodeInvocation> execution_order =
      instance.get_execution_order();
  std::map<dynamic_layer_guid_t, std::optional<milliseconds_t>> result =
      execute_dynamic_node_invocation_set(
          /*invocations=*/execution_order,
          /*allocator=*/instance.get_allocator(),
          /*optimizer_attrs=*/instance.get_optimizer_attrs(),
          /*profiling_settings=*/profiling_settings,
          /*ff_handle=*/ff_handle,
          /*device_idx=*/device_idx);
  instance.update_optimizer_attrs_for_next_iter();
  return result;
}

std::map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_forward_pass_for_computation_graph_instance(
        ComputationGraphInstance const &instance,
        std::optional<ProfilingSettings> const &profiling_settings,
        device_handle_t const &ff_handle,
        global_device_id_t device_idx) {
  std::vector<DynamicNodeInvocation> execution_order =
      filter(instance.get_execution_order(),
             [](DynamicNodeInvocation const &invocation) {
               DynamicTaskType task_type =
                   assert_unwrap(invocation.node_attrs.task_type);
               return task_type == DynamicTaskType::FWD;
             });

  return execute_dynamic_node_invocation_set(
      /*invocations=*/execution_order,
      /*allocator=*/instance.get_allocator(),
      /*optimizer_attrs=*/instance.get_optimizer_attrs(),
      /*profiling_settings=*/profiling_settings,
      /*ff_handle=*/ff_handle,
      /*device_idx=*/device_idx);
}

std::map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_backward_pass_for_computation_graph_instance(
        ComputationGraphInstance const &instance,
        std::optional<ProfilingSettings> const &profiling_settings,
        device_handle_t const &ff_handle,
        global_device_id_t device_idx) {
  std::vector<DynamicNodeInvocation> execution_order =
      filter(instance.get_execution_order(),
             [](DynamicNodeInvocation const &invocation) {
               DynamicTaskType task_type =
                   assert_unwrap(invocation.node_attrs.task_type);
               return task_type == DynamicTaskType::BWD;
             });

  return execute_dynamic_node_invocation_set(
      /*invocations=*/execution_order,
      /*allocator=*/instance.get_allocator(),
      /*optimizer_attrs=*/instance.get_optimizer_attrs(),
      /*profiling_settings=*/profiling_settings,
      /*ff_handle=*/ff_handle,
      /*device_idx=*/device_idx);
}

void perform_update_pass_for_computation_graph_instance(
    ComputationGraphInstance &instance,
    std::optional<ProfilingSettings> const &profiling_settings,
    device_handle_t const &ff_handle,
    global_device_id_t device_idx) {
  std::vector<DynamicNodeInvocation> execution_order =
      filter(instance.get_execution_order(),
             [](DynamicNodeInvocation const &invocation) {
               DynamicTaskType task_type =
                   assert_unwrap(invocation.node_attrs.task_type);
               return task_type == DynamicTaskType::UPD;
             });

  execute_dynamic_node_invocation_set(
      /*invocations=*/execution_order,
      /*allocator=*/instance.get_allocator(),
      /*optimizer_attrs=*/instance.get_optimizer_attrs(),
      /*profiling_settings=*/profiling_settings,
      /*ff_handle=*/ff_handle,
      /*device_idx=*/device_idx);
  instance.update_optimizer_attrs_for_next_iter();
}

} // namespace FlexFlow

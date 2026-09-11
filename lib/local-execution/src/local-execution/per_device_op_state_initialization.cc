#include "local-execution/per_device_op_state_initialization.h"
#include "local-execution/local_task_registry.h"
#include "local-execution/task_execution.h"
#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "utils/containers/all_are_true.h"
#include "utils/containers/transform.h"
#include "utils/optional.h"
#include <optional>

namespace FlexFlow {

bool no_nodes_are_initialized(DynamicOpenDataflowGraph const &g) {
  return all_are_true(
      transform(get_dynamic_nodes(g), [](DynamicNodeAttrs const &n) -> bool {
        return !n.per_device_op_state.has_value();
      }));
}

DynamicNodeInvocation initialize_node(DynamicNodeInvocation const &i,
                                      Allocator &allocator,
                                      device_handle_t const &device_handle,
                                      OptimizerAttrs const &optimizer_attrs,
                                      global_device_id_t device_idx) {
  if (!i.node_attrs.op_attrs.has_value() ||
      !i.node_attrs.op_attrs.value().is_pcg_op()) {
    return i;
  }

  // Get op
  ComputationGraphOpAttrs op_attrs =
      assert_unwrap(compgraph_op_attrs_from_pcg_op_attrs(
          assert_unwrap(i.node_attrs.op_attrs).require_pcg_op()));

  // Prepare arguments
  TaskArgumentAccessor arg_accessor =
      make_task_argument_accessor_for_invocation(
          /*invocation=*/i,
          /*allocator=*/allocator,
          /*profiling_settings=*/std::nullopt,
          /*ff_handle=*/device_handle,
          /*per_device_op_state=*/std::nullopt,
          /*optimizer_attrs=*/optimizer_attrs,
          /*device_idx=*/device_idx);

  // Run task init
  std::optional<DeviceSpecificPerDeviceOpState> per_device_op_state =
      call_init_task_impl(op_attrs, arg_accessor);

  DynamicNodeInvocation result = i;
  result.node_attrs.per_device_op_state = per_device_op_state;
  return result;
}

DynamicOpenDataflowGraph perform_per_device_op_state_initialization(
    DynamicOpenDataflowGraph const &dg,
    Allocator &allocator,
    device_handle_t const &device_handle,
    OptimizerAttrs const &optimizer_attrs,
    global_device_id_t device_idx) {

  ASSERT(no_nodes_are_initialized(dg));
  DynamicOpenDataflowGraph result = transform_dynamic_invocation_set(
      dg, [&](DynamicNodeInvocation const &invocation) {
        return initialize_node(
            invocation, allocator, device_handle, optimizer_attrs, device_idx);
      });

  return result;
}

} // namespace FlexFlow

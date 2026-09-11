#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COMPUTATION_GRAPH_INSTANCE_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COMPUTATION_GRAPH_INSTANCE_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/profiling_settings.dtg.h"
#include "local-execution/loss_config.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_layer_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/global_device_id_t.dtg.h"
#include "utils/units/milliseconds_t.h"
#include <map>
#include <optional>

namespace FlexFlow {

struct ComputationGraphInstance {
public:
  ComputationGraphInstance() = delete;
  explicit ComputationGraphInstance(
      std::vector<DynamicNodeInvocation> const &execution_order,
      Allocator &allocator,
      OptimizerAttrs const &optimizer_attrs,
      std::optional<GenericTensorAccessorW> logit_grad_tensor);
  std::vector<DynamicNodeInvocation> const &get_execution_order() const;
  Allocator &get_allocator() const;
  OptimizerAttrs const &get_optimizer_attrs() const;
  void update_optimizer_attrs_for_next_iter();
  std::optional<GenericTensorAccessorR> get_loss_tensor_accessor() const;

private:
  std::vector<DynamicNodeInvocation> execution_order;
  Allocator &allocator;
  OptimizerAttrs optimizer_attrs;
  std::optional<GenericTensorAccessorW> logit_grad_tensor;
};

ComputationGraphInstance create_computation_graph_instance(
    ComputationGraph const &cg,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<LossConfig> const &loss,
    std::map<DynamicValueAttrs, DynamicTensorAccessor> const &input_tensors,
    Allocator &allocator,
    device_handle_t const &device_handle,
    global_device_id_t global_device_id);

std::map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_all_passes_for_computation_graph_instance(
        ComputationGraphInstance &instance,
        std::optional<ProfilingSettings> const &profiling_settings,
        device_handle_t const &ff_handle,
        global_device_id_t global_device_id);
std::map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_forward_pass_for_computation_graph_instance(
        ComputationGraphInstance const &instance,
        std::optional<ProfilingSettings> const &profiling_settings,
        device_handle_t const &ff_handle,
        global_device_id_t global_device_id);
std::map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_backward_pass_for_computation_graph_instance(
        ComputationGraphInstance const &instance,
        std::optional<ProfilingSettings> const &profiling_settings,
        device_handle_t const &ff_handle,
        global_device_id_t global_device_id);
void perform_update_pass_for_computation_graph_instance(
    ComputationGraphInstance &instance,
    std::optional<ProfilingSettings> const &profiling_settings,
    device_handle_t const &ff_handle,
    global_device_id_t global_device_id);

} // namespace FlexFlow

#endif

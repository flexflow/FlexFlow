#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_BACKING_H

#include "arg_backing.h"
#include "device_specific.h"
#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "kernels/linear_kernels.h"
#include "op-attrs/operator_attrs.h"
#include "op_task_invocation.h"
#include "op_task_signature.h"
#include "pcg/computation_graph.h"
#include "task_argument_accessor.h"
#include <functional>
#include <unordered_map>
#include <vector>

namespace FlexFlow {

struct TaskSignatureImpl {
  TaskImplFunction impl_function;
  OpTaskSignature task_signature;
};

struct TaskRegistry {
  TaskRegistry(std::unordered_map<tensor_guid_t, GenericTensorAccessorW &>);
  void register_task(task_id_t, operator_guid_t);
  // void register_args(operator_guid_t, OpArgBacking);
  bool is_tensor_allocated(tensor_guid_t tensor_id);
  GenericTensorAccessorW &get_tensor_backing(tensor_guid_t op_slot_id);
  OpArgBacking get_arg_backing(operator_guid_t op_slot_id);

  OpTaskSignature get_init_signature(operator_guid_t);
  OpTaskSignature get_fwd_signature(operator_guid_t);
  OpTaskSignature get_bwd_signature(operator_guid_t);

  std::unordered_map<operator_guid_t, task_id_t> init_task_ids;
  std::unordered_map<operator_guid_t, task_id_t> forward_task_ids;
  std::unordered_map<operator_guid_t, task_id_t> backward_task_ids;
  // std::unordered_map<operator_guid_t, OpArgBacking> arg_mapping;
  // manage tensor slots

  std::unordered_map<operator_guid_t, std::vector<tensor_guid_t>>
      input_tensor_slots;
  std::unordered_map<operator_guid_t, std::vector<tensor_guid_t>>
      weight_tensor_slots;
  std::unordered_map<operator_guid_t, std::vector<tensor_guid_t>>
      output_tensor_slots;

  std::unordered_map<task_id_t, TaskSignatureImpl &> task_mapping;
  std::unordered_map<tensor_guid_t, GenericTensorAccessorW &> tensor_mapping;
};

struct LocalTrainingBacking {
  LocalTrainingBacking(
      ComputationGraph,
      Allocator,
      std::unordered_map<tensor_guid_t, GenericTensorAccessorW &>,
      PerDeviceFFHandle,
      EnableProfiling,
      ProfilingSettings);
  ~LocalTrainingBacking() = default;

  void execute_init();
  void execute_forward();
  void execute_backward();
  void execute_update();

  DeviceSpecific<DeviceStates> call_init_task_impl(task_id_t,
                                                   TaskArgumentAccessor);
  void call_task_impl(task_id_t, TaskArgumentAccessor);

  TaskArgumentAccessor get_task_arg_accessor(OpTaskInvocation);

private:
  Allocator allocator;
  ComputationGraph computation_graph;
  PerDeviceFFHandle ff_handle;
  EnableProfiling enable_profiling;
  ProfilingSettings profiling_settings;

  TaskRegistry task_registry;

  ArgBackingMapping arg_backing_mapping;
};

} // namespace FlexFlow

#endif

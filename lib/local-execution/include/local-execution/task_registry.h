
#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "local-execution/device_specific.h"
#include "local-execution/device_states.h"
#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/op_task_invocation.h"
#include "local-execution/op_task_signature.h"
#include "local-execution/runtime_arg_config.h"
#include "op-attrs/operator_attrs.h"
#include "pcg/computation_graph.h"
#include <functional>
#include <unordered_map>
#include <vector>

namespace FlexFlow {

using TensorBackingMapping =
    std::unordered_map<tensor_guid_t, GenericTensorAccessorW>;

struct TaskSignatureImpl {
  TaskImplFunction impl_function;
  OpTaskSignature task_signature;
};

struct TaskRegistry {
  TaskRegistry(TensorBackingMapping const &, RuntimeArgConfig const &);

  // tasks
  std::unordered_map<operator_guid_t, task_id_t> init_task_ids;
  std::unordered_map<operator_guid_t, task_id_t> forward_task_ids;
  std::unordered_map<operator_guid_t, task_id_t> backward_task_ids;
  std::unordered_map<task_id_t, TaskSignatureImpl> task_mapping;

  // tensors
  TensorBackingMapping tensor_mapping;

  // manage tensor slots
  std::unordered_map<operator_guid_t, std::vector<tensor_guid_t>>
      input_tensor_slots;
  std::unordered_map<operator_guid_t, std::vector<tensor_guid_t>>
      weight_tensor_slots;
  std::unordered_map<operator_guid_t, std::vector<tensor_guid_t>>
      output_tensor_slots;

  // arguments
  std::unordered_map<operator_guid_t,
                     std::optional<DeviceSpecific<DeviceStates>>>
      per_device_op_states;
  RuntimeArgConfig runtime_arg_config;

  // signatures
  OpTaskSignature get_init_signature(operator_guid_t const &);
  OpTaskSignature get_fwd_signature(operator_guid_t const &);
  OpTaskSignature get_bwd_signature(operator_guid_t const &);

  void register_task(task_id_t const &, operator_guid_t const &);
  void insert_per_device_op_state(operator_guid_t const &,
                                  DeviceSpecific<DeviceStates> const &);
  bool is_tensor_allocated(tensor_guid_t const &) const;
  GenericTensorAccessorW const &get_tensor_backing(tensor_guid_t const &) const;
  SlotTensorBackingMapping
      construct_slot_tensor_backing_map(OpTaskBinding const &,
                                        operator_guid_t const &) const;
  SlotArgBackingMapping
      construct_slot_argument_mapping(OpTaskBinding const &,
                                      operator_guid_t const &) const;
  ConcreteArgSpec compile_op_arg_ref_spec(OpArgRefSpec const &,
                                          operator_guid_t const &) const;
  ConcreteArgSpec compile_runtime_arg_ref_spec(RuntimeArgRefSpec const &) const;
};

} // namespace FlexFlow

#endif

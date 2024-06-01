
#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H

#include "device_specific.h"
#include "device_states.h"
#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "op-attrs/operator_attrs.h"
#include "op_task_invocation.h"
#include "op_task_signature.h"
#include "pcg/computation_graph.h"
#include "local_task_argument_accessor.h"
#include "runtime_arg_config.h"
#include <functional>
#include <unordered_map>
#include <vector>

namespace FlexFlow {

using TensorBackingMapping = std::unordered_map<tensor_guid_t, GenericTensorAccessorW const &>;

struct TaskSignatureImpl {
  TaskImplFunction impl_function;
  OpTaskSignature task_signature;
};

struct TaskRegistry {
  TaskRegistry(TensorBackingMapping, RuntimeArgConfig);

  // tasks
  std::unordered_map<operator_guid_t, task_id_t> init_task_ids;
  std::unordered_map<operator_guid_t, task_id_t> forward_task_ids;
  std::unordered_map<operator_guid_t, task_id_t> backward_task_ids;
  std::unordered_map<task_id_t, TaskSignatureImpl &> task_mapping;

  // tensors
  std::unordered_map<tensor_guid_t, GenericTensorAccessorW const &> tensor_mapping;
  
  // manage tensor slots
  std::unordered_map<operator_guid_t, std::vector<tensor_guid_t>>
      input_tensor_slots;
  std::unordered_map<operator_guid_t, std::vector<tensor_guid_t>>
      weight_tensor_slots;
  std::unordered_map<operator_guid_t, std::vector<tensor_guid_t>>
      output_tensor_slots;

  // arguments
  std::unordered_map<operator_guid_t, std::optional<DeviceSpecific<DeviceStates>>> per_device_op_states;
  RuntimeArgConfig runtime_arg_config;

  // signatures
  OpTaskSignature get_init_signature(operator_guid_t);
  OpTaskSignature get_fwd_signature(operator_guid_t);
  OpTaskSignature get_bwd_signature(operator_guid_t);

  void register_task(task_id_t, operator_guid_t);
  void insert_per_device_op_state(operator_guid_t, DeviceSpecific<DeviceStates>);
  bool is_tensor_allocated(tensor_guid_t);
  GenericTensorAccessorW const &get_tensor_backing(tensor_guid_t);
  void construct_slot_tensor_backing_map(SlotTensorBackingMapping &, OpTaskBinding const &, operator_guid_t const &);
  void construct_slot_argument_map(SlotArgBackingMap &, OpTaskBinding const &, operator_guid_t const &);
  ConcreteArgSpec compile_op_arg_ref_spec(OpArgRefSpec, operator_guid_t const &);
  ConcreteArgSpec compile_runtime_arg_ref_spec(RuntimeArgRefSpec);
};

}

#endif
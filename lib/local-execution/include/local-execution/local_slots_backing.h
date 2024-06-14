
#ifndef _FLEXFLOW_LOCAL_EXECUTION_SLOT_REGISTRY_H
#define _FLEXFLOW_LOCAL_EXECUTION_SLOT_REGISTRY_H

#include "kernels/accessor.h"
#include "local-execution/device_states.h"
#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/op_task_invocation.h"
#include "local-execution/runtime_arg_config.h"

namespace FlexFlow {

using TensorBackingMap =
    std::unordered_map<tensor_guid_t, GenericTensorAccessorW>;

struct LocalSlotsBacking {
  LocalSlotsBacking(TensorBackingMap const &, RuntimeArgConfig const &);

public:
  void add_per_device_op_state(layer_guid_t const &,
                               DeviceSpecific<DeviceStates> const &);
  bool is_tensor_allocated(tensor_guid_t const &) const;
  GenericTensorAccessorW const &get_tensor_backing(tensor_guid_t const &,
                                                   IsGrad const) const;
  TensorSlotsBacking construct_tensor_slots_backing(OpTaskBinding const &,
                                                    layer_guid_t const &) const;
  ArgSlotsBacking construct_arg_slots_backing(OpTaskBinding const &,
                                              layer_guid_t const &) const;
  ConcreteArgSpec resolve_op_arg_ref_spec(OpArgRefSpec const &,
                                          layer_guid_t const &) const;
  ConcreteArgSpec resolve_runtime_arg_ref_spec(RuntimeArgRefSpec const &) const;

public:
  // tensors
  TensorBackingMap tensor_mapping;
  TensorBackingMap gradient_tensor_mapping;
  std::unordered_map<layer_guid_t, std::vector<tensor_guid_t>>
      input_tensor_slots;
  std::unordered_map<layer_guid_t, std::vector<tensor_guid_t>>
      weight_tensor_slots;
  std::unordered_map<layer_guid_t, std::vector<tensor_guid_t>>
      output_tensor_slots;

  // arguments
  std::unordered_map<layer_guid_t, std::optional<DeviceSpecific<DeviceStates>>>
      per_device_op_states;
  RuntimeArgConfig runtime_arg_config;
};

} // namespace FlexFlow

#endif

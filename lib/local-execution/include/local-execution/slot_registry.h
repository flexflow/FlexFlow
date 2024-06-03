
#ifndef _FLEXFLOW_LOCAL_EXECUTION_SLOT_REGISTRY_H
#define _FLEXFLOW_LOCAL_EXECUTION_SLOT_REGISTRY_H

#include "kernels/accessor.h"
#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/device_states.h"
#include "local-execution/op_task_invocation.h"
#include "local-execution/runtime_arg_config.h"
#include "pcg/computation_graph.h"

namespace FlexFlow {

using TensorBackingMapping =
    std::unordered_map<tensor_guid_t, GenericTensorAccessorW>;

struct SlotRegistry {
  SlotRegistry(TensorBackingMapping const &, RuntimeArgConfig const &);

public:
  void add_per_device_op_state(operator_guid_t const &,
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

public:
  // tensors
  TensorBackingMapping tensor_mapping;
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
};

} // namespace FlexFlow

#endif

#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TASK_ARGUMENT_ACCESSOR_H

#include "local-execution/task_argument_accessor.h"
#include <unordered_map>
#include <variant>

namespace FlexFlow {

using SlotGradId = std::pair<slot_id, IsGrad>;
using TensorSlotsBacking = std::unordered_map<
    SlotGradId,
    std::variant<GenericTensorAccessorW, std::vector<GenericTensorAccessorW>>>;
using ArgSlotsBacking = std::unordered_map<slot_id, ConcreteArgSpec>;

struct LocalTaskArgumentAccessor : public ITaskArgumentAccessor {
  LocalTaskArgumentAccessor(Allocator const &allocator,
                            TensorSlotsBacking const &tensor_slots_backing,
                            ArgSlotsBacking const &arg_slots_backing)
      : allocator(allocator), tensor_slots_backing(tensor_slots_backing),
        arg_slots_backing(arg_slots_backing){};
  LocalTaskArgumentAccessor(LocalTaskArgumentAccessor const &) = delete;
  LocalTaskArgumentAccessor(LocalTaskArgumentAccessor &&) = delete;

  ConcreteArgSpec const &get_concrete_arg(slot_id) const override;

  GenericTensorAccessor
      get_tensor(slot_id slot, Permissions priv, IsGrad is_grad) const override;
  VariadicGenericTensorAccessor get_variadic_tensor(
      slot_id slot, Permissions priv, IsGrad is_grad) const override;

  Allocator get_allocator() const override;

  size_t get_device_idx() const override {
    return 0;
  }

private:
  Allocator allocator;
  TensorSlotsBacking tensor_slots_backing;
  ArgSlotsBacking arg_slots_backing;
};

bool are_slots_backings_equivalent_up_to_tensor_allocation_addresses(
    TensorSlotsBacking const &slots_1, TensorSlotsBacking const &slots_2);

CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalTaskArgumentAccessor);

} // namespace FlexFlow

#endif

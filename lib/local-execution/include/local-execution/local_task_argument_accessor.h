#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TASK_ARGUMENT_ACCESSOR_H

#include "local-execution/slot_grad_id.dtg.h"
#include "local-execution/task_argument_accessor.h"
#include <unordered_map>
#include <variant>

namespace FlexFlow {

using TensorSlotsBacking = std::unordered_map<
    SlotGradId,
    std::variant<GenericTensorAccessorW, std::vector<GenericTensorAccessorW>>>;
using ArgSlotsBacking = std::unordered_map<slot_id_t, ConcreteArgSpec>;

struct LocalTaskArgumentAccessor : public ITaskArgumentAccessor {
  LocalTaskArgumentAccessor(Allocator const &allocator,
                            TensorSlotsBacking const &tensor_slots_backing,
                            ArgSlotsBacking const &arg_slots_backing);

  LocalTaskArgumentAccessor(LocalTaskArgumentAccessor const &) = delete;
  LocalTaskArgumentAccessor(LocalTaskArgumentAccessor &&) = delete;

  ConcreteArgSpec const &get_concrete_arg(slot_id_t) const override;

  GenericTensorAccessor get_tensor(slot_id_t slot,
                                   Permissions priv,
                                   IsGrad is_grad) const override;
  VariadicGenericTensorAccessor get_variadic_tensor(
      slot_id_t slot, Permissions priv, IsGrad is_grad) const override;

  Allocator get_allocator() const override;

  size_t get_device_idx() const override;

private:
  Allocator allocator;
  TensorSlotsBacking tensor_slots_backing;
  ArgSlotsBacking arg_slots_backing;
};

bool are_slots_backings_equivalent_up_to_tensor_allocation_addresses(
    TensorSlotsBacking const &slots_1, TensorSlotsBacking const &slots_2);

CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalTaskArgumentAccessor);

std::string format_as(std::unordered_map<slot_id_t, ConcreteArgSpec> const &x);
std::ostream &
    operator<<(std::ostream &s,
               std::unordered_map<slot_id_t, ConcreteArgSpec> const &x);

} // namespace FlexFlow

#endif

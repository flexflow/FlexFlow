#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TASK_ARGUMENT_ACCESSOR_H

#include "local-execution/task_argument_accessor.h"
#include "local-execution/slot_grad_id.dtg.h"
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

  GenericTensorAccessor
      get_tensor(slot_id_t slot, Permissions priv, IsGrad is_grad) const override;
  VariadicGenericTensorAccessor get_variadic_tensor(
      slot_id_t slot, Permissions priv, IsGrad is_grad) const override;

  Allocator get_allocator() const override;

  size_t get_device_idx() const override;

private:
  Allocator allocator;
  TensorSlotsBacking tensor_slots_backing;
  ArgSlotsBacking arg_slots_backing;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalTaskArgumentAccessor);

} // namespace FlexFlow

#endif

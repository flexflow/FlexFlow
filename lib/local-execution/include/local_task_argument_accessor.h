#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TASK_ARGUMENT_ACCESSOR_H

#include "task_argument_accessor.h"
#include <unordered_map>
#include <variant>

namespace FlexFlow {

using SlotGradId = std::pair<slot_id, IsGrad>;
using TensorBackingMap = std::unordered_map<SlotGradId, 
                                            std::variant<GenericTensorAccessorW, 
                                                         std::vector<GenericTensorAccessorW>>>;
using ArgBackingMap = std::unordered_map<slot_id, ConcreteArgSpec>;

struct LocalTaskArgumentAccessor : public ITaskArgumentAccessor {
  LocalTaskArgumentAccessor(
      Allocator allocator,
      TensorBackingMap tensor_backing_map,
      ArgBackingMap argument_map)
      : allocator(allocator), tensor_backing_map(tensor_backing_map),
        argument_map(argument_map){};
  LocalTaskArgumentAccessor(LocalTaskArgumentAccessor const &) = delete;
  LocalTaskArgumentAccessor(LocalTaskArgumentAccessor &&) = delete;

  ConcreteArgSpec const &get_concrete_arg(slot_id) const override;

  PrivilegeTensorAccessor
      get_tensor(slot_id slot, Permissions priv, IsGrad is_grad) const override;
  PrivilegeVariadicTensorAccessor get_variadic_tensor(slot_id slot,
                                            Permissions priv,
                                            IsGrad is_grad) const override;

  Allocator get_allocator() const override;

  size_t get_device_idx() const override {
    return 0;
  }

private:
  Allocator allocator;
  TensorBackingMap tensor_backing_map;
  ArgBackingMap argument_map;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalTaskArgumentAccessor);

} // namespace FlexFlow

#endif

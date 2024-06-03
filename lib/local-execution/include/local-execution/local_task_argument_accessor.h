#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TASK_ARGUMENT_ACCESSOR_H

#include "local-execution/task_argument_accessor.h"
#include <unordered_map>
#include <variant>

namespace FlexFlow {

using SlotGradId = std::pair<slot_id, IsGrad>;
using SlotTensorBackingMapping = std::unordered_map<
    SlotGradId,
    std::variant<GenericTensorAccessorW, std::vector<GenericTensorAccessorW>>>;
using SlotArgBackingMapping = std::unordered_map<slot_id, ConcreteArgSpec>;

struct LocalTaskArgumentAccessor : public ITaskArgumentAccessor {
  LocalTaskArgumentAccessor(
      Allocator allocator,
      SlotTensorBackingMapping slot_tensor_backing_mapping,
      SlotArgBackingMapping slot_argument_mapping)
      : allocator(allocator),
        slot_tensor_backing_mapping(slot_tensor_backing_mapping),
        slot_argument_mapping(slot_argument_mapping){};
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
  SlotTensorBackingMapping slot_tensor_backing_mapping;
  SlotArgBackingMapping slot_argument_mapping;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalTaskArgumentAccessor);

} // namespace FlexFlow

#endif
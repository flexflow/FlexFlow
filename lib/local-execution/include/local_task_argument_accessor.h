#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_ARGUMENT_ACCESSOR_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "task_argument_accessor.h"
#include "config.h"
#include <unordered_map>
#include <variant>

namespace FlexFlow {

using SlotGradId = std::pair<slot_id, IsGrad>;
using TensorBackingOption = std::variant<std::unordered_map<SlotGradId, GenericTensorAccessorW>,
                                         std::unordered_map<SlotGradId, std::vector<GenericTensorAccessorW>>>;

struct LocalTaskArgumentAccessor : public ITaskArgumentAccessor {

  LocalTaskArgumentAccessor(
    Allocator allocator, 
    TensorBackingOption tensor_backing_map,
    std::unordered_map<slot_id, ArgRefBacking> argument_map)
      : allocator(allocator), tensor_backing_map(tensor_backing_map), argument_map(argument_map) {};
  LocalTaskArgumentAccessor(LocalTaskArgumentAccessor const &) = delete;
  LocalTaskArgumentAccessor(LocalTaskArgumentAccessor &&) = delete;

  ConcreteArgSpec const & get_concrete_arg(slot_id) const override;
  OpArgRefTypeBacking const & get_op_arg_ref(slot_id) const override;
  RuntimeArgRefTypeBacking const & get_runtime_arg(slot_id) const override;

  PrivilegeType get_tensor(slot_id slot, Permissions priv, IsGrad is_grad) const override;
  PrivilegeVariadicType get_variadic_tensor(slot_id slot,
                                                    Permissions priv, IsGrad is_grad) const override;

  Allocator get_allocator() const override;

  size_t get_device_idx() const override { return 0; }

private:
  Allocator allocator;
  TensorBackingOption tensor_backing_map;
  std::unordered_map<slot_id, ArgRefBacking> argument_map;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalTaskArgumentAccessor);

} // namespace FlexFlow

#endif

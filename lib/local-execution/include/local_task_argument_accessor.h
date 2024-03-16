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
using TensorBackingOption = std::variant<GenericTensorAccessorW, std::vector<GenericTensorAccessorW>>;

struct LocalTaskArgumentAccessor : public ITaskArgumentAccessor {

  LocalTaskArgumentAccessor() = default;

  ConcreteArgSpec const & get_concrete_arg(slot_id) const override;
  OpArgRefTypeBacking const & get_op_arg_ref(slot_id) const override;
  RuntimeArgRefTypeBacking const & get_runtime_arg(slot_id) const override;

  PrivilegeType get_tensor(slot_id slot, Permissions priv) const override;
  PrivilegeVariadicType get_variadic_tensor(slot_id slot,
                                                    Permissions priv) const override;
  PrivilegeType get_tensor_grad(slot_id slot,
                                        Permissions priv) const override;
  PrivilegeVariadicType
      get_variadic_tensor_grad(slot_id slot, Permissions priv) const override;

  Allocator get_allocator();

  void insert_tensor(SlotGradId tensor_id,
                     GenericTensorAccessorW tensor_backing) {
    this->tensor_backing_map.insert({tensor_id, tensor_backing});
  }

private:
  Allocator allocator;
  std::unordered_map<SlotGradId, TensorBackingOption> tensor_backing_map;
  std::unordered_map<slot_id, ArgRefBacking> argument_map;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalTaskArgumentAccessor);

} // namespace FlexFlow

#endif

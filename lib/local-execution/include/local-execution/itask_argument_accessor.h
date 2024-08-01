#ifndef _FLEXFLOW_LOCAL_EXECUTION_ITASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_ITASK_ARGUMENT_ACCESSOR_H

#include "kernels/allocation.h"
#include "local-execution/concrete_arg.h"
#include "local-execution/op_task_signature.h"
#include "local-execution/privilege_tensor_accessor.h"

namespace FlexFlow {

struct ITaskArgumentAccessor {
  ITaskArgumentAccessor &operator=(ITaskArgumentAccessor const &) = delete;

  virtual ~ITaskArgumentAccessor() = default;

  virtual ConcreteArgSpec const &get_concrete_arg(slot_id_t) const = 0;

  virtual GenericTensorAccessor
      get_tensor(slot_id_t slot, Permissions priv, IsGrad is_grad) const = 0;
  virtual VariadicGenericTensorAccessor get_variadic_tensor(
      slot_id_t slot, Permissions priv, IsGrad is_grad) const = 0;

  virtual Allocator get_allocator() const = 0;
  virtual size_t get_device_idx() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ITaskArgumentAccessor);

} // namespace FlexFlow

#endif

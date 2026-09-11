#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TASK_ARGUMENT_ACCESSOR_ITASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TASK_ARGUMENT_ACCESSOR_ITASK_ARGUMENT_ACCESSOR_H

#include "kernels/allocation.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/profiling_settings.dtg.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/concrete_arg_spec.h"
#include "task-spec/global_device_id_t.dtg.h"
#include "task-spec/ops/arg_slot_id_t.dtg.h"
#include "task-spec/per_device_op_state.dtg.h"
#include "task-spec/privilege_tensor_accessor.h"
#include "task-spec/task_argument_accessor/task_tensor_parameter.dtg.h"
#include "task-spec/training_tensor_type.dtg.h"

namespace FlexFlow {

struct ITaskArgumentAccessor {
  ITaskArgumentAccessor &operator=(ITaskArgumentAccessor const &) = delete;

  virtual ~ITaskArgumentAccessor() = default;

  virtual TensorShape get_tensor_shape(TensorSlotName) const = 0;

  virtual GenericTensorAccessor get_tensor(TaskTensorParameter,
                                           Permissions priv) const = 0;

  virtual std::optional<ProfilingSettings> get_profiling_settings() const = 0;
  virtual device_handle_t get_ff_handle() const = 0;
  virtual DeviceType get_kernel_device_type() const = 0;
  virtual PCGOperatorAttrs get_op_attrs() const = 0;
  virtual LossAttrs get_loss_attrs() const = 0;
  virtual PerDeviceOpState get_per_device_op_state() const = 0;
  virtual OptimizerAttrs get_optimizer_attrs() const = 0;

  virtual Allocator get_allocator() const = 0;
  virtual global_device_id_t get_device_idx() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ITaskArgumentAccessor);

} // namespace FlexFlow

#endif

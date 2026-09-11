#include "task-spec/optimizer.h"
#include "kernels/optimizer_kernels.h"
#include "task-spec/profiling.h"
#include "utils/containers/get_only.h"
#include "utils/overload.h"
#include "utils/units/milliseconds_t.h"

namespace FlexFlow {

static void sgd_update_task_impl(TaskArgumentAccessor const &acc) {
  SGDOptimizerAttrs attrs = acc.get_optimizer_attrs().require_sgd_optimizer();
  auto weight_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);
  auto weight = acc.get_tensor<Permissions::RW>(TensorSlotName::OUTPUT);
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  ASSERT(weight.shape == weight_grad.shape);

  ASSERT(get_num_elements(weight_grad.shape.dims).int_from_positive_int() %
             get_num_elements(weight.shape.dims).int_from_positive_int() ==
         0);
  int num_replicas =
      get_num_elements(weight_grad.shape.dims).int_from_positive_int() /
      get_num_elements(weight.shape.dims).int_from_positive_int();

  std::optional<GenericTensorAccessorW> sgd_v = std::nullopt;
  if (attrs.momentum > 0.0f) {
    sgd_v = acc.get_optimizer_tensor<Permissions::RW>(TensorSlotName::OUTPUT,
                                                      OptimizerSlotName::SGD_V);
    ASSERT(sgd_v.value().shape == weight.shape);
  }

  device_handle_t handle = acc.get_ff_handle();
  profile(sgd_update_task,
          profiling,
          kernel_device_type,
          "[SGD] update_time = %.2lfms\n",
          handle,
          attrs.lr,
          attrs.momentum,
          attrs.nesterov,
          attrs.weight_decay,
          weight_grad,
          num_replicas,
          weight,
          sgd_v);
}

TaskImplFunction get_sgd_update_task_impl() {
  return TaskImplFunction{GenericTaskImplFunction{sgd_update_task_impl}};
}

static void adam_update_task_impl(TaskArgumentAccessor const &acc) {
  AdamOptimizerAttrs attrs = acc.get_optimizer_attrs().require_adam_optimizer();
  auto weight_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);
  auto weight = acc.get_tensor<Permissions::RW>(TensorSlotName::OUTPUT);
  auto v_tensor = acc.get_optimizer_tensor<Permissions::RW>(
      TensorSlotName::WEIGHT, OptimizerSlotName::ADAM_V);
  auto m_tensor = acc.get_optimizer_tensor<Permissions::RW>(
      TensorSlotName::WEIGHT, OptimizerSlotName::ADAM_M);

  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  ASSERT(weight.shape == weight_grad.shape);
  int size = get_num_elements(weight_grad.shape.dims).int_from_positive_int();

  ASSERT(get_num_elements(weight_grad.shape.dims).int_from_positive_int() %
             get_num_elements(weight.shape.dims).int_from_positive_int() ==
         0);
  int num_replicas =
      get_num_elements(weight_grad.shape.dims).int_from_positive_int() /
      get_num_elements(weight.shape.dims).int_from_positive_int();

  device_handle_t handle = acc.get_ff_handle();
  profile(adam_update_task,
          profiling,
          kernel_device_type,
          "[Adam NCCL] update_time = %.2lfms\n",
          handle,
          attrs.alpha_t,
          attrs.beta1,
          attrs.beta2,
          attrs.weight_decay,
          attrs.epsilon,
          weight_grad.get_float_ptr(),
          size,
          num_replicas,
          m_tensor.get_float_ptr(),
          v_tensor.get_float_ptr(),
          weight.get_float_ptr());
}

TaskImplFunction get_adam_update_task_impl() {
  return TaskImplFunction{GenericTaskImplFunction{adam_update_task_impl}};
}

TaskImplFunction get_update_task_impl(OptimizerAttrs const &attrs) {
  return attrs.visit<TaskImplFunction>(overload{
      [&](SGDOptimizerAttrs const &) { return get_sgd_update_task_impl(); },
      [&](AdamOptimizerAttrs const &) { return get_adam_update_task_impl(); }});
}

} // namespace FlexFlow

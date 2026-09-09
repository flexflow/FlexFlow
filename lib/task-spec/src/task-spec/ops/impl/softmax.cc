/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "task-spec/ops/impl/softmax.h"
#include "kernels/softmax_kernels.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "task-spec/profiling.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  SoftmaxAttrs attrs = acc.get_op_attrs().require_softmax();

  TensorShape input_shape = acc.get_tensor_shape(TensorSlotName::INPUT);
  TensorShape output_shape = acc.get_tensor_shape(TensorSlotName::OUTPUT);

  std::optional<SoftmaxPerDeviceState> per_device_state =
      softmax_init_kernel(kernel_device_type, attrs, input_shape, output_shape);

  return DeviceSpecificPerDeviceOpState{
      acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  GenericTensorAccessorR input =
      acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  GenericTensorAccessorW output =
      acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);
  SoftmaxAttrs attrs = acc.get_op_attrs().require_softmax();

  device_handle_t handle = acc.get_ff_handle();

  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  std::optional<SoftmaxPerDeviceState> per_device_state =
      acc.get_per_device_op_state().require_softmax();

  return profile(softmax_forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Softmax] forward_time = {:.2lf}ms\n",
                 handle,
                 per_device_state,
                 attrs,
                 input,
                 output);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  GenericTensorAccessorR input =
      acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  GenericTensorAccessorW input_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::INPUT);
  GenericTensorAccessorR output =
      acc.get_tensor<Permissions::RO>(TensorSlotName::OUTPUT);
  GenericTensorAccessorR output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);

  SoftmaxAttrs attrs = acc.get_op_attrs().require_softmax();
  device_handle_t handle = acc.get_ff_handle();

  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  std::optional<SoftmaxPerDeviceState> per_device_state =
      acc.get_per_device_op_state().require_softmax();

  return profile(softmax_backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Softmax] backward_time = {:.2lf}ms\n",
                 handle,
                 per_device_state,
                 attrs,
                 output,
                 output_grad,
                 input,
                 input_grad);
}

TaskImplFunction get_softmax_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_softmax_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_softmax_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

} // namespace FlexFlow

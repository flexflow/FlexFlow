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

#include "task-spec/ops/impl/layer_norm.h"
#include "kernels/layer_norm_kernels.h"
#include "op-attrs/ff_ordered/ff_ordered_transform.h"
#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "task-spec/profiling.h"
#include "utils/containers/product.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include <type_traits>

namespace FlexFlow {

using namespace FlexFlow::Kernels::LayerNorm;

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);
  auto gamma = acc.get_tensor<Permissions::RW>(TensorSlotName::GAMMA);
  auto beta = acc.get_tensor<Permissions::RW>(TensorSlotName::BETA);

  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  LayerNormPerDeviceState state =
      acc.get_per_device_op_state().require_layer_norm().value();

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[LayerNorm] forward time = {:.2lf}ms\n",
                 state,
                 input,
                 output,
                 gamma,
                 beta);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto gamma = acc.get_tensor<Permissions::RO>(TensorSlotName::GAMMA);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::INPUT);
  auto gamma_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::GAMMA);
  auto beta_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::BETA);
  auto output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);

  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  LayerNormPerDeviceState state =
      acc.get_per_device_op_state().require_layer_norm().value();

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[LayerNorm] backward time = {:.2lf}ms\n",
                 state,
                 output_grad,
                 input,
                 input_grad,
                 gamma,
                 gamma_grad,
                 beta_grad);
}

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  LayerNormAttrs attrs = acc.get_op_attrs().require_layer_norm();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  Allocator allocator = acc.get_allocator();
  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  device_handle_t handle = acc.get_ff_handle();

  positive_int M = product(transform(attrs.axes, [&](ff_dim_t dim) {
    return dim_at_idx(input.shape.dims, dim);
  }));

  positive_int num_replicas = get_num_elements(input.shape.dims);

  positive_int effective_num_elements = M;
  positive_int effective_batch_size =
      positive_int{get_num_elements(input.shape.dims) / M};

  std::optional<LayerNormPerDeviceState> per_device_state =
      init_kernel(kernel_device_type,
                  handle,
                  allocator,
                  attrs.elementwise_affine,
                  effective_batch_size.int_from_positive_int(),
                  effective_num_elements.int_from_positive_int(),
                  attrs.eps);

  return DeviceSpecificPerDeviceOpState{
      acc.make_device_specific(per_device_state),
  };
}

TaskImplFunction get_layer_norm_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_layer_norm_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_layer_norm_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

} // namespace FlexFlow

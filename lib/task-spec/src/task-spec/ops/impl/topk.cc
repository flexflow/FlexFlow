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

#include "task-spec/ops/impl/topk.h"
#include "kernels/topk_kernels.h"
#include "task-spec/profiling.h"
#include "utils/exception.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::TopK;

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  TopKAttrs attrs = acc.get_op_attrs().require_topk();
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  positive_int length = dim_at_idx(input.shape.dims, legion_dim_t{0_n});
  positive_int batch_size =
      positive_int{get_num_elements(input.shape.dims) / length};
  auto indices = acc.get_tensor<Permissions::WO>(TensorSlotName::INDEX);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[TopK] forward_time = {:.2lf}ms\n",
                 input.get_float_ptr(),
                 output.get_float_ptr(),
                 indices.get_int32_ptr(),
                 batch_size.int_from_positive_int(),
                 length.int_from_positive_int(),
                 attrs.k.int_from_positive_int(),
                 attrs.sorted);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_op_attrs().require_topk();
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::INPUT);
  auto output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);

  auto indices = acc.get_tensor<Permissions::RO>(TensorSlotName::INDEX);

  positive_int length = dim_at_idx(input_grad.shape.dims, legion_dim_t{0_n});
  positive_int batch_size =
      positive_int{get_num_elements(input_grad.shape.dims) / length};

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[TopK] backward_time = {:.2lf}ms\n",
                 output_grad.get_float_ptr(),
                 indices.get_int32_ptr(),
                 input_grad.get_float_ptr(),
                 batch_size.int_from_positive_int(),
                 length.int_from_positive_int(),
                 attrs.k.int_from_positive_int());
}

TaskImplFunction get_topk_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_topk_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow

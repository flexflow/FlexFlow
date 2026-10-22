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

#include "task-spec/ops/impl/split.h"
#include "kernels/split_kernels.h"
#include "op-attrs/tensor_slot_name.h"
#include "task-spec/profiling.h"
#include "utils/containers/slice.h"
#include "utils/containers/transform.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"
#include "utils/nonnegative_int/nonnegative_range.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Split;

static std::vector<TensorSlotName> get_output_slots(SplitAttrs const &attrs) {
  std::vector<TensorSlotName> output_slots =
      get_variadic_outputs_slot_name_sequence();
  ASSERT(attrs.splits.size() <= output_slots.size());
  return slice(output_slots, 0, attrs.splits.size());
}

static std::pair<positive_int, positive_int>
    calc_block_size(TensorShape const &tensor_shape, ff_dim_t axis) {
  positive_int num_blocks = 1_p;
  positive_int block_size = 1_p;
  for (nonnegative_int d :
       nonnegative_range(get_num_elements(tensor_shape.dims)
                             .nonnegative_int_from_positive_int())) {
    if (d <= axis.value) {
      block_size *= dim_at_idx(tensor_shape.dims, legion_dim_t{d});
    } else {
      num_blocks *= dim_at_idx(tensor_shape.dims, legion_dim_t{d});
    }
  }
  return {num_blocks, block_size};
}

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  SplitAttrs attrs = acc.get_op_attrs().require_split();

  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  std::vector<GenericTensorAccessorW> outputs =
      transform(get_output_slots(attrs),
                [&](TensorSlotName slot) -> GenericTensorAccessorW {
                  return acc.get_tensor<Permissions::WO>(slot);
                });
  ASSERT(outputs.size() == attrs.splits.size());
  ASSERT(outputs.size() <= MAX_NUM_OUTPUTS);

  auto [num_blocks, in_block_size] = calc_block_size(input.shape, attrs.axis);
  std::vector<int> out_block_sizes =
      transform(outputs, [&](GenericTensorAccessorW const &output) -> int {
        auto [output_num_blocks, out_block_size] =
            calc_block_size(output.shape, attrs.axis);
        ASSERT(output_num_blocks == num_blocks);
        return out_block_size.int_from_positive_int();
      });
  std::vector<float *> output_ptrs =
      transform(outputs, [](GenericTensorAccessorW const &output) -> float * {
        return output.get_float_ptr();
      });
  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Split] forward_time = {:.2lf}ms\n",
                 output_ptrs.data(),
                 input.get_float_ptr(),
                 out_block_sizes.data(),
                 in_block_size.int_from_positive_int(),
                 num_blocks.int_from_positive_int(),
                 outputs.size());
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  std::optional<ProfilingSettings> profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  SplitAttrs attrs = acc.get_op_attrs().require_split();

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::INPUT);
  std::vector<GenericTensorAccessorR> output_grads =
      transform(get_output_slots(attrs),
                [&](TensorSlotName slot) -> GenericTensorAccessorR {
                  return acc.get_tensor_grad<Permissions::RO>(slot);
                });
  ASSERT(output_grads.size() == attrs.splits.size());
  ASSERT(output_grads.size() <= MAX_NUM_OUTPUTS);

  auto [num_blocks, in_block_size] =
      calc_block_size(input_grad.shape, attrs.axis);
  std::vector<int> out_block_sizes = transform(
      output_grads, [&](GenericTensorAccessorR const &output_grad) -> int {
        auto [output_num_blocks, out_block_size] =
            calc_block_size(output_grad.shape, attrs.axis);
        ASSERT(output_num_blocks == num_blocks);
        return out_block_size.int_from_positive_int();
      });
  std::vector<float const *> output_grad_ptrs =
      transform(output_grads,
                [](GenericTensorAccessorR const &output_grad) -> float const * {
                  return output_grad.get_float_ptr();
                });
  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Split] backward_time = {:.2lf}ms\n",
                 input_grad.get_float_ptr(),
                 output_grad_ptrs.data(),
                 out_block_sizes.data(),
                 in_block_size.int_from_positive_int(),
                 num_blocks.int_from_positive_int(),
                 output_grads.size());
}

TaskImplFunction get_split_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_split_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow

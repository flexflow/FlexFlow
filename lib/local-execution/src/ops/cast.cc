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

#include "cast.h"
#include "kernels/cast_kernels.h"

#include "local-execution/op_task_signature.h"
#include "utils/hash-utils.h"

using namespace FlexFlow::Kernels::Cast;

namespace FlexFlow {

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING };

OpTaskInvocation forward(CastAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(ATTRS, attrs);

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {CAST_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(CastAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {CAST_BWD_TASK_ID, binding};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto const &attrs = acc.get_argument<CastAttrs>(ATTRS);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Cast] forward_time = {:.2lf}ms\n",
                 input,
                 output,
                 input.data_type,
                 attrs.dtype);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto const &attrs = acc.get_argument<CastAttrs>(ATTRS);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);

  auto input_grad = acc.get_tensor_grad<Permissions::RO>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::WO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Cast] forward_time = {:.2lf}ms\n",
                 input_grad,
                 output_grad,
                 input.data_type,
                 attrs.dtype);
}

TaskImplFunction get_cast_fwd_task_impl() {
  return TaskImplFunction{forward_task_impl};
}
TaskImplFunction get_cast_bwd_task_impl() {
  return TaskImplFunction{backward_task_impl};
}

OpTaskSignature get_cast_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<CastAttrs>(ATTRS);
  fwd.add_arg_slot<bool>(PROFILING);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  return fwd;
}

OpTaskSignature get_cast_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_cast_fwd_signature());

  return bwd;
}

std::vector<task_id_t> get_task_ids(CastAttrs const &) {
  return {CAST_FWD_TASK_ID, CAST_BWD_TASK_ID};
}

}; // namespace FlexFlow

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

#include "combine.h"
#include "kernels/combine_kernels.h"
#include "local-execution/op_task_invocation.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Combine;

enum Slots { INPUT, OUTPUT, PROFILING };

OpTaskInvocation forward(CombineAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PROFILING, profiling_settings());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {task_id_t::COMBINE_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(CombineAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {task_id_t::COMBINE_BWD_TASK_ID, b};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Combine] forward_time = {:.2lf}ms\n",
                 input,
                 output);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input_grad = acc.get_tensor_grad<Permissions::RO>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::WO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Combine] backward_time = {:.2lf}ms\n",
                 input_grad,
                 output_grad);
}

OpTaskSignature get_combine_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  return fwd;
}

OpTaskSignature get_combine_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_combine_fwd_signature());

  return bwd;
}

TaskImplFunction get_combine_fwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_combine_bwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow

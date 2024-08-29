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

#include "reduction.h"
#include "kernels/reduction_kernels.h"
#include "op-attrs/get_output_shapes.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Reduction;

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING };

OpTaskInvocation forward(ReductionAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(ATTRS, attrs);
  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {task_id_t::REDUCTION_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(ReductionAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {task_id_t::REDUCTION_BWD_TASK_ID, binding};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling_settings =
      acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto attrs = acc.get_argument<ReductionAttrs>(ATTRS);

  size_t num_replicas = attrs.reduction_degree;

  return profile(forward_kernel,
                 profiling_settings,
                 "[Reduction] forward_time = {:.2lf}ms\n",
                 input,
                 output,
                 num_replicas);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input_grad = acc.get_tensor_grad<Permissions::WO>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);
  return profile(backward_kernel,
                 profiling,
                 "[Reduction] backward_time = {:.2lf}ms\n",
                 input_grad,
                 output_grad);
}

TaskImplFunction get_reduction_fwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_reduction_bwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_reduction_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_arg_slot<ReductionAttrs>(ATTRS);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  return fwd;
}
OpTaskSignature get_reduction_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_reduction_fwd_signature());
  return bwd;
}

std::vector<task_id_t> get_task_ids(ReductionAttrs const &) {
  return {task_id_t::REDUCTION_FWD_TASK_ID, task_id_t::REDUCTION_BWD_TASK_ID};
}

}; // namespace FlexFlow

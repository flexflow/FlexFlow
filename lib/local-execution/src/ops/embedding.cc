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

#include "embedding.h"
#include "kernels/embedding_kernels.h"
#include "op-attrs/get_output_shapes.h"
#include "op-attrs/ops/embedding.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Embedding;

enum Slots { INPUT, WEIGHT, OUTPUT, ATTRS, PROFILING };

OpTaskInvocation forward(EmbeddingAttrs const &attrs) {
  OpTaskBinding b;

  b.bind(INPUT, input_tensor(0));
  b.bind(WEIGHT, weight_tensor(0));
  b.bind(OUTPUT, output_tensor(0));

  b.bind_arg(ATTRS, attrs);
  b.bind_arg(PROFILING, profiling_settings());

  return {EMBED_FWD_TASK_ID, b};
}

OpTaskInvocation backward(EmbeddingAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {EMBED_BWD_TASK_ID, b};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  EmbeddingAttrs attrs = acc.get_argument<EmbeddingAttrs>(ATTRS);

  return profile(forward_kernel,
                 profiling,
                 "[Embedding] forward_time = {:.2lf}ms\n",
                 input,
                 output,
                 weight,
                 input.data_type,
                 output.data_type,
                 attrs.aggr,
                 input.shape.get_dim(),
                 output.shape.get_dim(),
                 input.shape[legion_dim_t(1)]);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);
  auto weight_grad = acc.get_tensor_grad<Permissions::RW>(WEIGHT);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  EmbeddingAttrs attrs = acc.get_argument<EmbeddingAttrs>(ATTRS);

  return profile(backward_kernel,
                 profiling,
                 "[Embedding] backward_time = {:.2lf}ms\n",
                 input,
                 output,
                 weight_grad,
                 input.data_type,
                 output.data_type,
                 attrs.aggr,
                 input.shape.get_dim(),
                 output.shape.get_dim(),
                 input.shape.at(ff_dim_t(0)));
}

TaskImplFunction get_embedding_fwd_task_impl() {
  return forward_task_impl;
}
TaskImplFunction get_embedding_bwd_task_impl() {
  return backward_task_impl;
}

OpTaskSignature get_embedding_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(INPUT);
  fwd.add_input_slot(OUTPUT);
  fwd.add_input_slot(WEIGHT);

  fwd.add_arg_slot<EmbeddingAttrs>(ATTRS);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);

  return fwd;
}

OpTaskSignature get_embedding_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_embedding_fwd_signature());
  return bwd;
}

std::vector<task_id_t> get_task_ids(EmbeddingAttrs const &) {
  return {EMBED_FWD_TASK_ID, EMBED_BWD_TASK_ID};
}

} // namespace FlexFlow

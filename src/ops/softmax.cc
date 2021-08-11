/* Copyright 2017 Stanford, NVIDIA
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

#include "flexflow/ops/softmax.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;

Tensor FFModel::softmax(const Tensor _input, int dim, const char *name)
{
  if (dim < 0)
    dim += _input->num_dims;
  Softmax *sm = new Softmax(*this, _input, _input->num_dims-1-dim, name);
  layers.push_back(sm);
  return sm->outputs[0];
}

Softmax::Softmax(FFModel& model,
                 const Tensor _input,
                 int _dim,
                 const char* name)
: Op(model, OP_SOFTMAX, name, 1/*inputs*/, 0/*weights*/, 1/*outputs*/, _input),
  dim(_dim)
{
  // Currently assume we always perform softmax along the inner most dim
  assert(dim == 0);
  ParallelDim dims[MAX_TENSOR_DIM];
  int numdim = _input->num_dims;
  for (int i = 0; i < numdim; i++)
    dims[i] = _input->dims[numdim-1-i];
  outputs[0] = model.create_tensor(numdim, dims, DT_FLOAT, this);
}

void Softmax::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(SOFTMAX_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Softmax)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void Softmax::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(SOFTMAX_FWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Softmax::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(SOFTMAX_BWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

bool Softmax::get_int_parameter(PMParameter para, int* value) const
{
  switch(para) {
    case PM_SOFTMAX_DIM:
      *value = dim;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

size_t Softmax::get_params_hash() const {
  size_t hash = this->inputs[0]->get_owner_independent_hash();
  hash_combine(hash, this->dim);

  return hash;
}

using PCG::Node;
Node FFModel::get_or_create_softmax_node(const Tensor input,
                                         int softmax_dim)
{
  size_t hash = input->get_owner_independent_hash();
  hash = hash * 31 + std::hash<int>()(softmax_dim);
  const auto& it = cached_softmax_ops.find(hash);
  Softmax* softmax = NULL;
  if (it != cached_softmax_ops.end()) {
    softmax = it->second;
  } else {
    softmax = new Softmax(*this, input, softmax_dim, NULL);
    cached_softmax_ops[hash] = softmax;
  }
  Node ret;
  ret.guid = node_global_guid ++;
  ret.ptr = softmax;
  return ret;
}

}; // namespace FlexFlow
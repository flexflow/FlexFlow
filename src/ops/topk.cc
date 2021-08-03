/* Copyright 2021 Facebook
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

#include "flexflow/ops/topk.h"

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

// For an input tensor, computes the top k entries in each row
// (resp. vector along the last dimension). Thus,
// values.shape = indices.shape = input.shape[:-1] + [k]
void FFModel::top_k(const Tensor input,
                    Tensor* outputs,
                    int k,
                    bool sorted,
                    const char *name)
{
  TopK* topk = new TopK(*this, input, k, sorted, name);
  layers.push_back(topk);
  assert(topk->numOutputs == 2);
  outputs[0] = topk->outputs[0];
  outputs[1] = topk->outputs[1];
}

TopK::TopK(FFModel& model,
           const Tensor _input,
           int _k, bool _sorted,
           const char* name)
: Op(model, OP_TOPK, name, 1/*inputs*/, 0/*weights*/, 2/*outputs*/, _input),
  k(_k), sorted(_sorted)
{
  int numdim = inputs[0]->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++)
    dims[i] = inputs[0]->dims[i];
  dims[0].size = k;
  assert(inputs[0]->dims[0].degree == 1);
  assert(inputs[0]->dims[0].parallel_idx == -1);
  outputs[0] = model.create_tensor_legion_ordering(
      numdim, dims, _input->data_type,
      this, 0/*owner_idx*/);
  outputs[1] = model.create_tensor_legion_ordering(
      numdim, dims, DT_INT32,
      this, 1/*owner_idx*/);
}

void TopK::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(TOPK_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(TopK)), argmap,
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
  launcher.add_region_requirement(
    RegionRequirement(outputs[1]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[1]->region));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void TopK::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(TOPK_FWD_TASK_ID, parallel_is,
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
  launcher.add_region_requirement(
    RegionRequirement(outputs[1]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[1]->region));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void TopK::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(TOPK_BWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0](I): value_grad
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  // regions[1](I): indices
  launcher.add_region_requirement(
    RegionRequirement(outputs[1]->part, 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, outputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // regions[2](I/O): input_grad
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
                      READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

}; // namespace FlexFlow
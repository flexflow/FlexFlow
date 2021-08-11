/* Copyright 2020 Facebook
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

#include "flexflow/ops/split.h"
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

void FFModel::split(const Tensor input,
                    Tensor* outputs,
                    const std::vector<int>& splits,
                    int axis,
                    const char* name)
{
  Split* split = new Split(*this, input, splits, axis, name);
  layers.push_back(split);
  for (size_t i = 0; i < splits.size(); i++)
    outputs[i] = split->outputs[i];
}

size_t Split::get_params_hash() const {
  size_t hash = 0;
  for (int i = 0; i < this->numInputs; i++) {
    hash_combine(hash, this->inputs[i]->get_owner_independent_hash()); 
  }
  hash_combine(hash, this->axis);

  return hash;
}

Split::Split(FFModel& model,
             const Tensor input,
             const std::vector<int>& splits,
             int _axis,
             const char* name)
: Op(model, OP_SPLIT, name, 1/*inputs*/, 0/*weights*/, splits.size()/*outputs*/, input),
  axis(input->num_dims-1-_axis)
{
  numOutputs = splits.size();
  // Note that we use the Legion dim ordering
  // axis = input->num_dims-1-_axis
  assert(axis >= 0);
  numWeights = 0;
  int split_size = 0;
  for (int i = 0; i < numOutputs; i++) {
    split_size += splits[i];
    int numdim = input->num_dims;
    ParallelDim dims[MAX_TENSOR_DIM];
    for (int j = 0; j < numdim; j++)
      dims[j] = input->dims[j];
    dims[axis].size = splits[i];
    // Assert the _axis dim cannot be parallelized
    assert(dims[axis].degree == 1);
    assert(dims[axis].parallel_idx == -1);
    outputs[i] = model.create_tensor_legion_ordering(
        numdim, dims, input->data_type,
        this/*owner_op*/, i/*owner_idx*/);
  }
  // Check split sizes
  assert(split_size == input->dims[axis].size);
}

void Split::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(SPLIT_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Split)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i]->part, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[i]->region));
    launcher.add_field(i+1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void Split::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(SPLIT_FWD_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Split)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i]->part, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[i]->region));
    launcher.add_field(i+1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void Split::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(SPLIT_BWD_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Split)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i]->part_grad, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, outputs[i]->region_grad));
    launcher.add_field(i+1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

}; // namespace FlexFlow
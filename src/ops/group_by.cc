/* Copyright 2019 Stanford
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

#include "flexflow/ops/groupby.h"
#include <math.h>
#include <stdio.h>

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

void FFModel::group_by(const Tensor input,
                       const Tensor assign,
                       Tensor* outputs,
                       int n, float alpha,
                       const char* name)
{
  assert(false);
#ifdef DEADCODE
  Group_by* group_by = new Group_by(*this, input, assign, n, alpha, name);
  layers.push_back(group_by);
  for (int i = 0; i < n; i++)
    outputs[i] = group_by->outputs[i];
#endif
}


Group_by::Group_by(FFModel& model,
                   const ParallelTensor _input,
                   const ParallelTensor _assign,
                   int _n, float _alpha,
                   const char* name)
: Op(model, OP_GROUP_BY, name, 2/*inputs*/, 0/*weights*/, _n/*outputs*/, _input, _assign),
  n(_n),
  alpha(_alpha)
{
  assert(_input->num_dims == 2); // NOTE: Is that a problem if you e.g. want to pass in images
  assert(_input->num_dims == 2);
  assert(_input->dims[1] == _assign->dims[1]);
  assert(n > 0);

  // List of outputs
  int k = _assign->dims[0].size;
  for(int i = 0; i < n; i++) {
    outputs[i]->num_dims = 2;
    outputs[i]->dims[0].size = inputs[0]->dims[0].size;
    outputs[i]->dims[1].size = (int)ceil(alpha*k/n*inputs[0]->dims[1].size);
  }

  numWeights = 0;
}

void Group_by::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(GROUP_BY_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Group_by)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // data
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // assign
  launcher.add_region_requirement(
    RegionRequirement(inputs[1]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(1, FID_DATA);

  // output
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i]->part, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[i]->region));
    launcher.add_field(i+2, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void Group_by::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(GROUP_BY_FWD_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Group_by)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // data
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);

  // assign
  launcher.add_region_requirement(
    RegionRequirement(inputs[1]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(1, FID_DATA);

  // output
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i]->part, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[i]->region));
    launcher.add_field(i+2, FID_DATA);
  }

  runtime->execute_index_space(ctx, launcher);
}

void Group_by::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(GROUP_BY_BWD_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Group_by)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());

  // input_grad
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);

  // assign
  launcher.add_region_requirement(
    RegionRequirement(inputs[1]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(1, FID_DATA);

  // output grad
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i]->part_grad, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[i]->region_grad));
    launcher.add_field(i+2, FID_DATA);
  }

  runtime->execute_index_space(ctx, launcher);
}

}; // namespace FlexFlow

/* Copyright 2021 Stanford, Facebook
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

#include "flexflow/ops/aggregate_spec.h"

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

Tensor FFModel::aggregate_spec(const Tensor* inputs, /* gate_preds, gate_assign, full_gate_pred, n * exp_pred */
                          int n, float lambda_bal, const char* name)
{
  AggregateSpec* aggr = new AggregateSpec(*this, inputs, n, lambda_bal, name);
  layers.push_back(aggr);
  return aggr->outputs[0];
}

AggregateSpec::AggregateSpec(FFModel& model,
                    const Tensor* _inputs,
                    int _n, float _lambda_bal, const char* name)
: Op(model, OP_AGG_SPEC, name, _n+4/*numInputs*/, 0/*numWeights*/, 1/*numOutputs*/, _inputs),
  n(_n), lambda_bal(_lambda_bal)
{
  // FIXME: For now, set upper limits Better: Do as follows, but memory is
  // assigned per block, so requires to check that
  // https://stackoverflow.com/questions/5531247/allocating-shared-memory/5531640#5531640
  assert(n <= AGGREGATE_SPEC_MAX_N && "Increase AGGREGATE_SPEC_MAX_N in #define");
  assert(inputs[0]->dims[0].size <= AGGREGATE_SPEC_MAX_K && "Increase AGGREGATE_SPEC_MAX_K in #define");
  assert(inputs[0]->dims[1].size <= AGGREGATE_SPEC_MAX_BATCH_SIZE && "Increase AGGREGATE_SPEC_MAX_BATCH_SIZE in #define");

  assert(n+4 == numInputs);
  assert(n > 0);
  assert(inputs[0]->num_dims == 2);
  assert(inputs[1]->num_dims == 2);
  assert(inputs[2]->num_dims == 2);
  assert(inputs[3]->num_dims == 2);

  for(int i = 0; i < inputs[0]->num_dims; i++) {
    assert(inputs[0]->dims[i] == inputs[1]->dims[i]);
    assert(inputs[0]->dims[i] == inputs[2]->dims[i]);
  }
  assert(inputs[0]->dims[1] == inputs[3]->dims[1]);
  assert(inputs[3]->dims[0].size == n);

  // expert inputs
  int num_dim = inputs[4]->num_dims;
  int out_dim = inputs[4]->dims[0].size;
  for(int i = 1; i < n; i++) {
    assert(inputs[i+4]->num_dims == num_dim);
    assert(inputs[i+4]->dims[0].size == out_dim);
  }
  // Set output shape
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < num_dim-1; i++)
    dims[i] = inputs[4]->dims[i];
  dims[num_dim-1] = inputs[0]->dims[num_dim-1];
  numOutputs = 1;
  outputs[0] = model.create_tensor_legion_ordering(num_dim, dims, DT_FLOAT, this);

  numWeights = 0;
}

void AggregateSpec::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  parallel_is = outputs[0]->parallel_is;
  IndexLauncher launcher(AGG_SPEC_INIT_TASK_ID, parallel_is,
    TaskArgument(this, sizeof(AggregateSpec)), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    outputs[0]->machine_view.hash());
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void AggregateSpec::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  parallel_is = outputs[0]->parallel_is;
  IndexLauncher launcher(AGG_SPEC_FWD_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(AggregateSpec)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // gate_preds
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // gate_assign
  launcher.add_region_requirement(
    RegionRequirement(inputs[1]->part, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // exp_preds
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(inputs[i+4]->part, 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[i+4]->region));
    launcher.add_field(i+2, FID_DATA);
  }
  // output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(n+2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void AggregateSpec::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  parallel_is = outputs[0]->parallel_is;
  IndexLauncher launcher(AGG_SPEC_BWD_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(AggregateSpec)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());

  // gate_preds
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);

  // gate_assign
  launcher.add_region_requirement(
    RegionRequirement(inputs[1]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(1, FID_DATA);

  // true gate_assign
  launcher.add_region_requirement(
    RegionRequirement(inputs[2]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[2]->region));
  launcher.add_field(2, FID_DATA);

  // gate gradients full
  launcher.add_region_requirement(
    RegionRequirement(inputs[3]->part_grad, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[3]->region_grad));
  launcher.add_field(3, FID_DATA);

  // exp gradients
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(inputs[i+4]->part_grad, 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[i+4]->region_grad));
    launcher.add_field(i+4, FID_DATA);
  }

  // output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(n+4, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

}; // namespace FlexFlow
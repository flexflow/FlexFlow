/* Copyright 2021 CMU, Facebook, LANL, MIT, and Stanford (alphabetical)
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

#include "model.h"

using namespace Legion;

Tensor FFModel::replicate(
    const Tensor input,
    int replicate_legion_dim,
    int replicate_degree,
    const char* name)
{
  Replicate *repl = new Replicate(*this, input,
      replicate_legion_dim, replicate_degree, name);
  layers.push_back(repl);
  return repl->outputs[0];
}

Replicate::Replicate(
    FFModel& model,
    const Tensor _input,
    int _replicate_legion_dim,
    int _replicate_degree,
    const char* name)
: ParallelOp(model, OP_REPLICATE, name, _input),
  replicate_dim(_replicate_legion_dim),
  replicate_degree(_replicate_degree)
{
  int numdim = _input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = _input->dims[i];
  }
  dims[replicate_dim].size *= replicate_degree;
  dims[replicate_dim].degree *= replicate_degree;
  TensorBase::update_parallel_ids(numdim, dims);
  for (int i = 0; i < numdim; i++)
    if (i != replicate_dim) {
      register_output_input_parallel_dims(outputs[0], i, inputs[0], i);
    }
  outputs[0] = model.create_tensor_legion_ordering(
      numdim, dims, DT_FLOAT, this);
  // Check correctness
  assert(check_output_input_weight_parallel_dims());
}

void Replicate::init(const FFModel& ff)
{
  // Do nothing
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexSpace task_is = outputs[0]->parallel_is;
  IndexLauncher launcher(REPLICATE_FWD_TASK_ID, task_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Replicate::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexSpace task_is = outputs[0]->parallel_is;
  IndexLauncher launcher(REPLICATE_FWD_TASK_ID, task_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      FFConfig::get_hash_id(std::string(name)));
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

void Replicate::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexSpace task_is = outputs[0]->parallel_is;
  IndexLauncher launcher(REPLICATE_BWD_TASK_ID, task_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

bool Replicate::measure_operator_cost(
    Simulator* sim,
    const ParallelConfig& pc,
    CostMetrics& cost_metrics)
{
  assert(false);
  return false;
}

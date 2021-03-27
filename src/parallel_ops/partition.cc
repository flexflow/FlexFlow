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

Tensor FFModel::repartition(
    const Tensor input,
    int repartition_legion_dim,
    int repartition_degree,
    const char* name)
{
  Repartition *part = new Repartition(*this, input,
      repartition_legion_dim, repartition_degree, name);
  layers.push_back(part);
  return part->outputs[0];
}

Repartition::Repartition(
    FFModel& model,
    const Tensor _input,
    int _repartition_legion_dim,
    int _repartition_degree,
    const char* name)
: ParallelOp(model, OP_REPARTITION, name, _input),
  repartition_dim(_repartition_legion_dim),
  repartition_degree(_repartition_degree)
{
  int numdim = _input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = _input->dims[i];
  }
  dims[repartition_dim].degree *= repartition_degree;
  TensorBase::update_parallel_ids(numdim, dims);
  for (int i = 0; i < numdim; i++)
    if (i != repartition_dim) {
      register_output_input_parallel_dims(outputs[0], i, inputs[0], i);
    }
  outputs[0] = model.create_tensor_legion_ordering(
      numdim, dims, DT_FLOAT, this);
  outputs[0]->print("Repartition::output");
  // Check correctness
  // assert(check_output_input_weight_parallel_dims());
}

void Repartition::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexSpace task_is = outputs[0]->parallel_is;
  IndexLauncher launcher(REPARTITION_FWD_TASK_ID, task_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Repartition::create_input_partition(FFModel& ff)
{
  assert(outputs[0]->part != LogicalPartition::NO_PART);
  assert(inputs[0]->part != LogicalPartition::NO_PART);
  ff.create_disjoint_partition(outputs[0]->num_dims, outputs[0]->dims,
      outputs[0]->parallel_is, inputs[0]->region, input_lp);
  ff.create_disjoint_partition(inputs[0]->num_dims, inputs[0]->dims,
      inputs[0]->parallel_is, outputs[0]->region_grad, output_grad_lp);
}

void Repartition::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexSpace task_is = outputs[0]->parallel_is;
  IndexLauncher launcher(REPARTITION_FWD_TASK_ID, task_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Repartition::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexSpace task_is = outputs[0]->parallel_is;
  IndexLauncher launcher(REPARTITION_BWD_TASK_ID, task_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(output_grad_lp, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));

  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

bool Repartition::measure_operator_cost(
    Simulator* sim,
    const ParallelConfig& pc,
    CostMetrics& cost_metrics) const
{
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;
  return true;
}

bool Repartition::get_int_parameter(PMParameter para, int* value) const
{
  switch(para) {
    case PM_REPARTITION_DIM:
      *value = repartition_dim;
      return true;
    case PM_REPARTITION_DEGREE:
      *value = repartition_degree;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

Node FFModel::get_or_create_repartition_node(const Tensor input,
                                             int repartition_dim,
                                             int repartition_degree)
{
  size_t hash = input->get_owner_independent_hash();
  hash = hash * 31 + std::hash<int>()(repartition_dim);
  hash = hash * 31 + std::hash<int>()(repartition_degree);
  const auto& it = cached_repartition_ops.find(hash);
  Repartition* repartition = NULL;
  if (it != cached_repartition_ops.end()) {
    repartition = it->second;
  } else {
    repartition = new Repartition(*this, input, repartition_dim,
                                  repartition_degree, NULL);
    cached_repartition_ops[hash] = repartition;
  }
  Node ret;
  ret.ptr = repartition;
  ret.guid = node_global_guid++;
  return ret;
}

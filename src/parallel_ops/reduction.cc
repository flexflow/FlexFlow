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

Tensor FFModel::reduction(
    const Tensor input,
    int reduction_legion_dim,
    int reduction_degree,
    const char* name)
{
  Reduction* reduce = new Reduction(*this, input,
      reduction_legion_dim, reduction_degree, name);
  layers.push_back(reduce);
  return reduce->outputs[0];
}

Reduction::Reduction(
    FFModel& model,
    const Tensor _input,
    int _reduction_legion_dim,
    int _reduction_degree,
    const char* name)
: ParallelOp(model, OP_REDUCTION, name, _input),
  reduction_dim(_reduction_legion_dim),
  reduction_degree(_reduction_degree)
{
  int numdim = _input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = _input->dims[i];
  }
  assert(dims[reduction_dim].degree % reduction_degree == 0);
  assert(dims[reduction_dim].size % reduction_degree == 0);
  dims[reduction_dim].degree /= reduction_degree;
  dims[reduction_dim].size /= reduction_degree;
  TensorBase::update_parallel_ids(numdim, dims);
  outputs[0] = model.create_tensor_legion_ordering(
      numdim, dims, DT_FLOAT, this);
  for (int i = 0; i < numdim; i++) {
    register_output_input_parallel_dims(outputs[0], i, inputs[0], i);
  }
  // Check correctness
  // assert(check_output_input_weight_parallel_dims());
}

void Reduction::create_input_partition(FFModel& ff)
{
  assert(outputs[0]->part != LogicalPartition::NO_PART);
  assert(inputs[0]->part != LogicalPartition::NO_PART);
  // input_lp is a disjoint partition
  ff.create_disjoint_partition(outputs[0]->num_dims, outputs[0]->dims,
      outputs[0]->parallel_is, inputs[0]->region, input_lp);
  // output_grad_lp is an aliased partitioning along the replica dim
  ff.create_aliased_partition(inputs[0]->num_dims, inputs[0]->dims,
      reduction_dim, inputs[0]->parallel_is, outputs[0]->region_grad, output_grad_lp);
}

void Reduction::init(const FFModel& ff)
{
  // Do nothing
}

void Reduction::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexSpace task_is = outputs[0]->parallel_is;
  IndexLauncher launcher(REDUCTION_FWD_TASK_ID, task_is,
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

void Reduction::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexSpace task_is = inputs[0]->parallel_is;
  IndexLauncher launcher(REDUCTION_BWD_TASK_ID, task_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

bool Reduction::measure_operator_cost(
    Simulator* sim,
    const ParallelConfig& pc,
    CostMetrics& cost_metrics) const
{
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;
  return true;
}

bool Reduction::get_int_parameter(PMParameter para, int* value) const
{
  switch(para) {
    case PM_REDUCTION_DIM:
      *value = reduction_dim;
      return true;
    case PM_REDUCTION_DEGREE:
      *value = reduction_degree;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

bool Reduction::append_parallel_op_info(std::vector<ParallelOpInfo>& parallel_ops) const
{
  ParallelOpInfo ret;
  ret.op_type = op_type;
  ret.parallel_dim = reduction_dim;
  ret.parallel_degree = reduction_degree;
  parallel_ops.push_back(ret);
  return true;
}

Node FFModel::get_or_create_reduction_node(const Tensor input,
                                           int reduction_dim,
                                           int reduction_degree)
{
  size_t hash = input->get_owner_independent_hash();
  hash = hash * 31 + std::hash<int>()(reduction_dim);
  hash = hash * 31 + std::hash<int>()(reduction_degree);
  const auto& it = cached_reduction_ops.find(hash);
  Reduction* reduction = NULL;
  if (it != cached_reduction_ops.end()) {
    reduction = it->second;
  } else {
    reduction = new Reduction(*this, input, reduction_dim,
                              reduction_degree, NULL);
    cached_reduction_ops[hash] = reduction;
  }
  Node ret;
  ret.ptr = reduction;
  ret.guid = node_global_guid++;
  return ret;
}

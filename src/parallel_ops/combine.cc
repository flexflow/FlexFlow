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

Tensor FFModel::combine(
    const Tensor input,
    int combine_legion_dim,
    int combine_degree,
    const char* name)
{
  Combine* comb = new Combine(*this, input,
      combine_legion_dim, combine_degree, name);
  layers.push_back(comb);
  return comb->outputs[0];
}

Combine::Combine(
    FFModel& model,
    const Tensor _input,
    int _combine_legion_dim,
    int _combine_degree,
    const char* name)
: ParallelOp(model, OP_COMBINE, name, _input),
  combine_dim(_combine_legion_dim),
  combine_degree(_combine_degree)
{
  int numdim = _input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = _input->dims[i];
  }
  assert (combine_degree > 0 && "Must use combine_degree > 0");
  assert(dims[combine_dim].degree % combine_degree == 0);
  dims[combine_dim].degree /= combine_degree;
  TensorBase::update_parallel_ids(numdim, dims);
  outputs[0] = model.create_tensor_legion_ordering(
      numdim, dims, DT_FLOAT, this);
  for (int i = 0; i < numdim; i++) {
    register_output_input_parallel_dims(outputs[0], i, inputs[0], i);
  }
  // Check correctness
  //assert(check_output_input_weight_parallel_dims());
}

void Combine::init(const FFModel& ff)
{
  // Do nothing
}

void Combine::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexSpace task_is = outputs[0]->parallel_is;
  IndexLauncher launcher(COMBINE_FWD_TASK_ID, task_is,
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

void Combine::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexSpace task_is = inputs[0]->parallel_is;
  IndexLauncher launcher(COMBINE_BWD_TASK_ID, task_is,
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

bool Combine::measure_operator_cost(
    Simulator* sim,
    const ParallelConfig& pc,
    CostMetrics& cost_metrics) const
{
  //TODO: to be implemented
  cost_metrics.forward_time = 0.05f;
  cost_metrics.backward_time = 0.05f;
  return true;
}

bool Combine::get_int_parameter(PMParameter para, int* value) const
{
  switch(para) {
    case PM_COMBINE_DIM:
      *value = combine_dim;
      return true;
    case PM_NUM_PARTITIONS:
      *value = combine_degree;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

Node FFModel::create_combine_node(const Tensor input,
                                  int combine_dim,
                                  int combine_degree)
{
  size_t hash = input->get_owner_independent_hash();
  hash = hash * 31 + std::hash<int>()(combine_dim);
  hash = hash * 31 + std::hash<int>()(combine_degree);
  const auto& it = cached_combine_ops.find(hash);
  Combine* combine = NULL;
  if (it != cached_combine_ops.end()) {
    combine = it->second;
  } else {
    combine = new Combine(*this, input, combine_dim,
                          combine_degree, NULL);
    cached_combine_ops[hash] = combine;
  }
  Node ret;
  ret.ptr = combine;
  ret.guid = node_global_guid ++;
  return ret;
}

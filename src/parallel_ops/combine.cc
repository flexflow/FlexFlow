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

#include "flexflow/parallel_ops/combine.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::Rect;
using Legion::PhysicalRegion;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;
using Legion::coord_t;
using Legion::Memory;
using Legion::Machine;
using Legion::LogicalRegion;
using Legion::LogicalPartition;

ParallelTensor FFModel::combine(
    const ParallelTensor input,
    int combine_legion_dim,
    int combine_degree,
    const char* name)
{
  assert(false);
#ifdef DEADCODE
  Combine* comb = new Combine(*this, input,
      combine_legion_dim, combine_degree, name);
  layers.push_back(comb);
  return comb->outputs[0];
#endif
}

Combine::Combine(
    FFModel& model,
    const ParallelTensor _input,
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
  ParallelTensorBase::update_parallel_ids(numdim, dims);
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, DT_FLOAT, this);
  //inputs[0]->print("Combine::input");
  //outputs[0]->print("Combine::output");
}

void Combine::init(const FFModel& ff)
{
  forward(ff);
}

void Combine::create_input_partition(FFModel& ff)
{
  assert(outputs[0]->part != LogicalPartition::NO_PART);
  assert(inputs[0]->part != LogicalPartition::NO_PART);
  ff.create_disjoint_partition(outputs[0]->num_dims, outputs[0]->dims,
      outputs[0]->parallel_is, inputs[0]->region, input_lp);
  ff.create_disjoint_partition(inputs[0]->num_dims, inputs[0]->dims,
      inputs[0]->parallel_is, outputs[0]->region_grad, output_grad_lp);
}

void Combine::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexLauncher launcher(COMBINE_FWD_TASK_ID, outputs[0]->parallel_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
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

void Combine::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexLauncher launcher(COMBINE_BWD_TASK_ID, inputs[0]->parallel_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      inputs[0]->machine_view.hash());
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

bool Combine::measure_operator_cost(
    Simulator* sim,
    const MachineView& mv,
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
    case PM_COMBINE_DEGREE:
      *value = combine_degree;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

bool Combine::append_parallel_op_info(std::vector<ParallelOpInfo>& parallel_ops) const
{
  ParallelOpInfo ret;
  ret.op_type = op_type;
  ret.parallel_dim = combine_dim;
  ret.parallel_degree = combine_degree;
  parallel_ops.push_back(ret);
  return true;
}

size_t Combine::get_params_hash() const {
  size_t hash = this->inputs[0]->get_owner_independent_hash();
  hash_combine(hash, this->combine_dim);
  hash_combine(hash, this->combine_degree);

  return hash;
}

using PCG::Node;
Node FFModel::get_or_create_combine_node(const ParallelTensor input,
                                         int combine_dim,
                                         int combine_degree)
{
  if (input->dims[combine_dim].degree % combine_degree != 0) {
    return Node::INVALID_NODE;
  }

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

tl::optional<RecordFormatter> Combine::as_dot() const {
  RecordFormatter rf;
  {
    std::ostringstream oss;
    oss << "dim(" << this->combine_dim << ")";
    rf << oss.str();
  }
  {
    std::ostringstream oss;
    oss << "deg(" << this->combine_degree << ")";
    rf << oss.str();
  }
  return rf;
}

/*static*/
void Combine::forward_task(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  Domain input_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(output_domain == input_domain);

  const float* input_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* output_ptr = helperGetTensorPointerWO<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);

  forward_kernel<float>(input_ptr, output_ptr, output_domain.get_volume());
}

void Combine::backward_task(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  Domain output_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Domain input_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  assert(output_grad_domain == input_grad_domain);

  const float* output_grad_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float* input_grad_ptr = helperGetTensorPointerRW<float>(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);

  backward_kernel<float>(output_grad_ptr, input_grad_ptr, output_grad_domain.get_volume());
}

}; // namespace FlexFlow

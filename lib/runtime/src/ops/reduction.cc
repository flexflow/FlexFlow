/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

#include "ops/reduction.h"
#include "kernels/reduction_kernels.h"
#include "utils/hash-utils.h"

namespace FlexFlow {
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::LogicalPartition;
using Legion::LogicalRegion;
using Legion::Machine;
using Legion::Memory;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

using namespace FlexFlow::Kernels::Reduction;

ReductionParams Reduction::get_params() const {
  ReductionParams params;
  params.reduction_legion_dim = this->reduction_dim;
  params.reduction_degree = this->reduction_degree;
  return params;
}

Reduction::Reduction(FFModel &model,
                     const ParallelTensor _input,
                     int _reduction_legion_dim,
                     int _reduction_degree,
                     char const *name)
    : ParallelOp(model, OP_REDUCTION, name, _input),
      reduction_dim(_reduction_legion_dim),
      reduction_degree(_reduction_degree) {
  int numdim = _input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = _input->dims[i];
  }
  assert(dims[reduction_dim].degree % reduction_degree == 0);
  assert(dims[reduction_dim].size % reduction_degree == 0);
  dims[reduction_dim].degree /= reduction_degree;
  dims[reduction_dim].size /= reduction_degree;
  ParallelTensorBase::update_parallel_ids(numdim, dims);
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, DT_FLOAT, this);
}

Reduction::Reduction(FFModel &model,
                     ReductionParams const &params,
                     ParallelTensor const input,
                     char const *name)
    : Reduction(model,
                input,
                params.reduction_legion_dim,
                params.reduction_degree,
                name) {}

void Reduction::create_input_partition(FFModel &ff) {
  assert(outputs[0]->part != LogicalPartition::NO_PART);
  assert(inputs[0]->part != LogicalPartition::NO_PART);
  // input_lp is a disjoint partition
  ff.create_disjoint_partition(outputs[0]->num_dims,
                               outputs[0]->dims,
                               outputs[0]->parallel_is,
                               inputs[0]->region,
                               input_lp);
  // output_grad_lp is an aliased partitioning along the replica dim
  ff.create_aliased_partition(inputs[0]->num_dims,
                              inputs[0]->dims,
                              reduction_dim,
                              inputs[0]->parallel_is,
                              outputs[0]->region_grad,
                              output_grad_lp);
}

void Reduction::init(FFModel const &ff) {
  forward(ff);
}

void Reduction::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexLauncher launcher(REDUCTION_FWD_TASK_ID,
                         outputs[0]->parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(
      input_lp, 0 /*projection id*/, READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Reduction::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexLauncher launcher(REDUCTION_BWD_TASK_ID,
                         inputs[0]->parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         inputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(output_grad_lp,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

bool Reduction::measure_operator_cost(Simulator *sim,
                                      MachineView const &pc,
                                      CostMetrics &cost_metrics) const {
  cost_metrics = CostMetrics();
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;

  cost_metrics.sync_time = 0;
  cost_metrics.inputs_memory = 0;
  cost_metrics.outputs_memory = 0;
  cost_metrics.weights_memory = 0;
  return true;
}

bool Reduction::get_int_parameter(PMParameter para, int *value) const {
  switch (para) {
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

bool Reduction::append_parallel_op_info(
    std::vector<ParallelOpInfo> &parallel_ops) const {
  ParallelOpInfo ret;
  ret.op_type = op_type;
  ret.parallel_dim = reduction_dim;
  ret.parallel_degree = reduction_degree;
  parallel_ops.push_back(ret);
  return true;
}

/*static*/
void Reduction::forward_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  // Currently only support the outter most dimension
  for (int i = 0; i < output_domain.get_dim() - 1; i++) {
    assert(output_domain.lo()[i] == input_domain.lo()[i]);
    assert(output_domain.hi()[i] == input_domain.hi()[i]);
  }
  size_t num_elements = output_domain.get_volume();
  size_t num_replicas = input_domain.get_volume() / num_elements;
  float const *input_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float *output_ptr = helperGetTensorPointerRW<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);

  forward_kernel<float>(input_ptr, output_ptr, num_elements, num_replicas);
}

void Reduction::backward_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  Domain output_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain input_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(input_grad_domain.get_volume() == output_grad_domain.get_volume());
  float const *output_grad_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float *input_grad_ptr = helperGetTensorPointerWO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);

  backward_kernel<float>(
      output_grad_ptr, input_grad_ptr, output_grad_domain.get_volume());
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ReductionParams>::operator()(
    FlexFlow::ReductionParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.reduction_legion_dim);
  hash_combine(key, params.reduction_degree);
  return key;
}
}; // namespace std

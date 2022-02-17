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

#include "flexflow/parallel_ops/fused_parallel_op.h"
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

FusedParallelOp::FusedParallelOp(
    FFModel& model,
    const ParallelTensor _input,
    const std::vector<ParallelOpInfo>& _parallel_ops)
: ParallelOp(model, OP_FUSED_PARALLEL, NULL, _input),
  num_parallel_ops(0)
{
  set_parallel_ops(_parallel_ops);
  assert(check_no_redundant_parallel_ops());
  int numdim = _input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++)
    dims[i] = _input->dims[i];
  for (int i = 0; i < num_parallel_ops; i++) {
    ParallelOpInfo info = parallel_ops[i];
    switch (info.op_type) {
      case OP_REPARTITION:
      {
        dims[info.parallel_dim].degree *= info.parallel_degree;
        break;
      }
      case OP_COMBINE:
      {
        assert(dims[info.parallel_dim].degree % info.parallel_degree == 0);
        dims[info.parallel_dim].degree /= info.parallel_degree;
        break;
      }
      case OP_REPLICATE:
      {
        dims[info.parallel_dim].size *= info.parallel_degree;
        dims[info.parallel_dim].degree *= info.parallel_degree;
        break;
      }
      case OP_REDUCTION:
      {
        assert(dims[info.parallel_dim].degree % info.parallel_degree == 0);
        assert(dims[info.parallel_dim].size % info.parallel_degree == 0);
        dims[info.parallel_dim].degree /= info.parallel_degree;
        dims[info.parallel_dim].size /= info.parallel_degree;
        break;
      }
      default:
      {
        assert(false && "Unsupported parallel op");
      }
    }
    ParallelTensorBase::update_parallel_ids(numdim, dims);
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, inputs[0]->data_type, this);
}

void FusedParallelOp::set_parallel_ops(const std::vector<ParallelOpInfo>& _parallel_ops)
{
  for (size_t i = 0; i < _parallel_ops.size(); i++)
    parallel_ops[num_parallel_ops++] = _parallel_ops[i];
}

bool FusedParallelOp::check_no_redundant_parallel_ops(void) const
{
  //for (int i = 1; i < num_parallel_ops; i++)
  //  if (parallel_ops[i-1].parallel_dim > parallel_osp[i].parallel_dim)
  //    return false;
  // Check there are no redundant combine/repartition
  for (int i = 1; i < num_parallel_ops; i ++) {
    if (parallel_ops[i].op_type == OP_COMBINE) {
      if (parallel_ops[i-1].op_type == OP_REPARTITION) {
        if (parallel_ops[i].parallel_dim == parallel_ops[i-1].parallel_dim)
          return false;
      }
    }
    if (parallel_ops[i].op_type == OP_REPARTITION) {
      if (parallel_ops[i-1].op_type == OP_COMBINE) {
        if (parallel_ops[i].parallel_dim == parallel_ops[i-1].parallel_dim)
          return false;
      }
    }
    if (parallel_ops[i].op_type == parallel_ops[i-1].op_type) {
      if (parallel_ops[i].parallel_dim == parallel_ops[i-1].parallel_dim)
        return false;
    }
  }
  return true;
}

void FusedParallelOp::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexLauncher launcher(FUSED_PARALLELOP_FWD_TASK_ID, outputs[0]->parallel_is,
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

void FusedParallelOp::create_input_partition(FFModel& ff)
{
  assert(outputs[0]->part != LogicalPartition::NO_PART);
  assert(inputs[0]->part != LogicalPartition::NO_PART);
  ff.create_disjoint_partition(outputs[0]->num_dims, outputs[0]->dims,
      outputs[0]->parallel_is, inputs[0]->region, input_lp);
  ff.create_disjoint_partition(inputs[0]->num_dims, inputs[0]->dims,
      inputs[0]->parallel_is, outputs[0]->region_grad, output_grad_lp);
}

void FusedParallelOp::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexLauncher launcher(FUSED_PARALLELOP_FWD_TASK_ID, outputs[0]->parallel_is,
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

void FusedParallelOp::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexLauncher launcher(FUSED_PARALLELOP_BWD_TASK_ID, inputs[0]->parallel_is,
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

bool FusedParallelOp::measure_operator_cost(
    Simulator* sim,
    const ParallelConfig& pc,
    CostMetrics& cost_metrics) const
{
  cost_metrics.forward_time = 0.1f;
  cost_metrics.backward_time = 0.1f;
  return true;
}

bool FusedParallelOp::append_parallel_op_info(std::vector<ParallelOpInfo>& _parallel_ops) const
{
  for (int i = 0; i < num_parallel_ops; i++) {
    _parallel_ops.push_back(parallel_ops[i]);
  }
  return true;
}

size_t FusedParallelOp::get_params_hash() const {
  size_t hash = this->inputs[0]->get_owner_independent_hash();
  hash_combine(hash, this->num_parallel_ops);
  for (ParallelOpInfo const &p : this->parallel_ops) {
    hash_combine(hash, p.op_type);
    hash_combine(hash, p.parallel_dim);
    hash_combine(hash, p.parallel_degree);
  }

  return hash;
}

using PCG::Node;
Node FFModel::get_or_create_fused_parallel_node(const ParallelTensor input,
                                                const std::vector<ParallelOpInfo>& parallel_ops)
{
  // Try to combine _parallel_ops's dimensions
  if (parallel_ops.size() == 0) {
    return get_or_create_noop_node(input);
  } else if (parallel_ops.size() == 1) {
    return this->get_or_create_parallel_op_node(input, parallel_ops[0]);
  }
  size_t hash = input->get_owner_independent_hash();
  for (size_t i = 0; i < parallel_ops.size(); i++) {
    hash = hash * 31 + std::hash<int>()(parallel_ops[i].op_type);
    hash = hash * 31 + std::hash<int>()(parallel_ops[i].parallel_dim);
    hash = hash * 31 + std::hash<int>()(parallel_ops[i].parallel_degree);
  }
  const auto& it = cached_fused_parallel_ops.find(hash);
  FusedParallelOp* fused = NULL;
  if (it != cached_fused_parallel_ops.end()) {
    fused = it->second;
  } else {
    fused = new FusedParallelOp(*this, input, parallel_ops);
    cached_fused_parallel_ops[hash] = fused;
  }

  Node ret;
  ret.ptr = fused;
  ret.guid = node_global_guid++;
  return ret;
}

void FusedParallelOp::forward_task(
    const Task *task,
    const std::vector<PhysicalRegion>& regions,
    Context ctx, Runtime* runtime)
{}

void FusedParallelOp::backward_task(
    const Task *task,
    const std::vector<PhysicalRegion>& regions,
    Context ctx, Runtime* runtime)
{}

}; // namespace FlexFlow

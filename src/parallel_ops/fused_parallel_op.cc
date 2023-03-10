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

#include "flexflow/parallel_ops/fused_parallel_op.h"
#include "flexflow/model.h"
#include "flexflow/utils/hash_utils.h"

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

/* Params */
bool operator==(ParallelOpInfo const &lhs, ParallelOpInfo const &rhs) {
  return lhs.op_type == rhs.op_type &&
         lhs.parallel_degree == rhs.parallel_degree &&
         lhs.parallel_dim == rhs.parallel_dim;
}

bool operator==(FusedParallelOpParams const &lhs,
                FusedParallelOpParams const &rhs) {
  return lhs.parallel_ops == rhs.parallel_ops;
}

bool FusedParallelOpParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

FusedParallelOpParams FusedParallelOp::get_params() const {
  FusedParallelOpParams params;
  std::vector<ParallelOpInfo> ops(std::begin(this->parallel_ops),
                                  std::end(this->parallel_ops));
  params.parallel_ops = ops;
  return params;
}

FusedParallelOp::FusedParallelOp(
    FFModel &model,
    const ParallelTensor _input,
    std::vector<ParallelOpInfo> const &_parallel_ops)
    : ParallelOp(model, OP_FUSED_PARALLEL, NULL, _input), num_parallel_ops(0) {
  set_parallel_ops(_parallel_ops);
  assert(check_no_redundant_parallel_ops());
  int numdim = _input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = _input->dims[i];
  }
  for (int i = 0; i < num_parallel_ops; i++) {
    ParallelOpInfo info = parallel_ops[i];
    switch (info.op_type) {
      case OP_REPARTITION: {
        dims[info.parallel_dim].degree *= info.parallel_degree;
        break;
      }
      case OP_COMBINE: {
        assert(dims[info.parallel_dim].degree % info.parallel_degree == 0);
        dims[info.parallel_dim].degree /= info.parallel_degree;
        break;
      }
      case OP_REPLICATE: {
        dims[info.parallel_dim].size *= info.parallel_degree;
        dims[info.parallel_dim].degree *= info.parallel_degree;
        break;
      }
      case OP_REDUCTION: {
        assert(dims[info.parallel_dim].degree % info.parallel_degree == 0);
        assert(dims[info.parallel_dim].size % info.parallel_degree == 0);
        dims[info.parallel_dim].degree /= info.parallel_degree;
        dims[info.parallel_dim].size /= info.parallel_degree;
        break;
      }
      default: {
        assert(false && "Unsupported parallel op");
      }
    }
    ParallelTensorBase::update_parallel_ids(numdim, dims);
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, inputs[0]->data_type, this);
}

FusedParallelOp::FusedParallelOp(FFModel &model,
                                 FusedParallelOpParams const &params,
                                 const ParallelTensor input)
    : FusedParallelOp(model, input, params.parallel_ops) {}

void FusedParallelOp::set_parallel_ops(
    std::vector<ParallelOpInfo> const &_parallel_ops) {
  for (size_t i = 0; i < _parallel_ops.size(); i++) {
    parallel_ops[num_parallel_ops++] = _parallel_ops[i];
  }
}

bool FusedParallelOp::check_no_redundant_parallel_ops(void) const {
  // for (int i = 1; i < num_parallel_ops; i++)
  //   if (parallel_ops[i-1].parallel_dim > parallel_osp[i].parallel_dim)
  //     return false;
  //  Check there are no redundant combine/repartition
  for (int i = 1; i < num_parallel_ops; i++) {
    if (parallel_ops[i].op_type == OP_COMBINE) {
      if (parallel_ops[i - 1].op_type == OP_REPARTITION) {
        if (parallel_ops[i].parallel_dim == parallel_ops[i - 1].parallel_dim) {
          return false;
        }
      }
    }
    if (parallel_ops[i].op_type == OP_REPARTITION) {
      if (parallel_ops[i - 1].op_type == OP_COMBINE) {
        if (parallel_ops[i].parallel_dim == parallel_ops[i - 1].parallel_dim) {
          return false;
        }
      }
    }
    if (parallel_ops[i].op_type == parallel_ops[i - 1].op_type) {
      if (parallel_ops[i].parallel_dim == parallel_ops[i - 1].parallel_dim) {
        return false;
      }
    }
  }
  return true;
}

void FusedParallelOp::init(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexLauncher launcher(FUSED_PARALLELOP_FWD_TASK_ID,
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

void FusedParallelOp::create_input_partition(FFModel &ff) {
  assert(outputs[0]->part != LogicalPartition::NO_PART);
  assert(inputs[0]->part != LogicalPartition::NO_PART);
  ff.create_disjoint_partition(outputs[0]->num_dims,
                               outputs[0]->dims,
                               outputs[0]->parallel_is,
                               inputs[0]->region,
                               input_lp);
  ff.create_disjoint_partition(inputs[0]->num_dims,
                               inputs[0]->dims,
                               inputs[0]->parallel_is,
                               outputs[0]->region_grad,
                               output_grad_lp);
}

void FusedParallelOp::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexLauncher launcher(FUSED_PARALLELOP_FWD_TASK_ID,
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

void FusedParallelOp::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexLauncher launcher(FUSED_PARALLELOP_BWD_TASK_ID,
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

bool FusedParallelOp::measure_operator_cost(Simulator *sim,
                                            MachineView const &pc,
                                            CostMetrics &cost_metrics) const {
  cost_metrics = CostMetrics();
  cost_metrics.forward_time = 0.1f;
  cost_metrics.backward_time = 0.1f;

  cost_metrics.sync_time = 0;
  cost_metrics.inputs_memory = 0;
  cost_metrics.outputs_memory = 0;
  cost_metrics.weights_memory = 0;
  return true;
}

bool FusedParallelOp::append_parallel_op_info(
    std::vector<ParallelOpInfo> &_parallel_ops) const {
  for (int i = 0; i < num_parallel_ops; i++) {
    _parallel_ops.push_back(parallel_ops[i]);
  }
  return true;
}

using PCG::Node;
Node FFModel::get_or_create_fused_parallel_node(
    const ParallelTensor input,
    std::vector<ParallelOpInfo> const &parallel_ops) {
  // Try to combine _parallel_ops's dimensions
  if (parallel_ops.size() == 0) {
    return get_or_create_noop_node(input);
  } else if (parallel_ops.size() == 1) {
    return this->get_or_create_parallel_op_node(input, parallel_ops[0]);
  }
  FusedParallelOpParams params;
  params.parallel_ops = parallel_ops;
  return get_or_create_node<FusedParallelOp>(input, params);
}

void FusedParallelOp::forward_task(Task const *task,
                                   std::vector<PhysicalRegion> const &regions,
                                   Context ctx,
                                   Runtime *runtime) {}

void FusedParallelOp::backward_task(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime) {}

}; // namespace FlexFlow

namespace std {

size_t hash<FlexFlow::FusedParallelOpParams>::operator()(
    FlexFlow::FusedParallelOpParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.parallel_ops.size());
  for (FlexFlow::ParallelOpInfo const &p : params.parallel_ops) {
    hash_combine(key, p.op_type);
    hash_combine(key, p.parallel_dim);
    hash_combine(key, p.parallel_degree);
  }
  return key;
}
}; // namespace std

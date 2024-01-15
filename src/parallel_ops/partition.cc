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

#include "flexflow/parallel_ops/partition.h"
#include "flexflow/model.h"
#include "flexflow/parallel_ops/kernels/partition_kernels.h"
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

using namespace FlexFlow::Kernels::Repartition;

/* Params */
bool operator==(RepartitionParams const &lhs, RepartitionParams const &rhs) {
  return lhs.repartition_legion_dim == rhs.repartition_legion_dim &&
         lhs.repartition_degree == rhs.repartition_degree;
}

bool RepartitionParams::is_valid(ParallelTensorShape const &input) const {
  bool valid = input.is_valid();
  valid &= (input.dims[this->repartition_legion_dim].size %
                (this->repartition_degree *
                 input.dims[this->repartition_legion_dim].degree) ==
            0);
  return valid;
}

RepartitionParams Repartition::get_params() const {
  RepartitionParams params;
  params.repartition_legion_dim = this->repartition_dim;
  params.repartition_degree = this->repartition_degree;
  if (this->name != nullptr) {
    strcpy(params.name, this->name);
  }
  return params;
}

Repartition::Repartition(FFModel &model,
                         const ParallelTensor _input,
                         int _repartition_legion_dim,
                         int _repartition_degree,
                         char const *name)
    : ParallelOp(model, OP_REPARTITION, name, _input),
      repartition_dim(_repartition_legion_dim),
      repartition_degree(_repartition_degree) {
  int numdim = _input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = _input->dims[i];
  }
  dims[repartition_dim].degree *= repartition_degree;
  ParallelTensorBase::update_parallel_ids(numdim, dims);
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, inputs[0]->data_type, this);
  // inputs[0]->print("Repartition::input");
  // outputs[0]->print("Repartition::output");
}

Repartition::Repartition(FFModel &model,
                         RepartitionParams const &params,
                         ParallelTensor const input,
                         char const *name)
    : Repartition(model,
                  input,
                  params.repartition_legion_dim,
                  params.repartition_degree,
                  params.name) {}

OpMeta *Repartition::init_task(Task const *task,
                               std::vector<PhysicalRegion> const &regions,
                               Context ctx,
                               Runtime *runtime) {
  return nullptr;
}

void Repartition::init_inference(
    FFModel const &ff,
    std::vector<ParallelTensor> const &batch_inputs,
    std::vector<ParallelTensor> const &batch_outputs,
    MachineView const *mv) {
  ArgumentMap argmap;
  parallel_is = batch_outputs[0]->parallel_is;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  size_t machine_view_hash =
      mv ? mv->hash() : batch_outputs[0]->machine_view.hash();
  IndexLauncher launcher(REPARTITION_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  assert(inference_input_lps.find(batch_inputs[0]) !=
         inference_input_lps.end());
  launcher.add_region_requirement(
      RegionRequirement(inference_input_lps[batch_inputs[0]],
                        0 /*projection id*/,
                        READ_ONLY,
                        EXCLUSIVE,
                        batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
}

void Repartition::init(FFModel const &ff) {
  ArgumentMap argmap;
  parallel_is = outputs[0]->parallel_is;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexLauncher launcher(REPARTITION_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
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
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
}

void Repartition::create_input_partition(FFModel &ff) {
  assert(ff.config.computationMode == COMP_MODE_TRAINING);
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

void Repartition::create_input_partition_inference(
    FFModel &ff,
    std::vector<ParallelTensor> const &batch_inputs,
    std::vector<ParallelTensor> const &batch_outputs) {
  assert(ff.config.computationMode == COMP_MODE_INFERENCE);
  assert(batch_outputs[0]->part != LogicalPartition::NO_PART);
  assert(batch_inputs[0]->part != LogicalPartition::NO_PART);
  ff.create_disjoint_partition(batch_outputs[0]->num_dims,
                               batch_outputs[0]->dims,
                               batch_outputs[0]->parallel_is,
                               batch_inputs[0]->region,
                               inference_input_lps[batch_inputs[0]]);
}

FutureMap
    Repartition::inference(FFModel const &ff,
                           BatchConfigFuture const &bc,
                           std::vector<ParallelTensor> const &batch_inputs,
                           std::vector<ParallelTensor> const &batch_outputs,
                           MachineView const *mv) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  assert(batch_inputs[0]->data_type == batch_outputs[0]->data_type);
  DataType data_type = batch_inputs[0]->data_type;
  size_t machine_view_hash =
      mv ? mv->hash() : batch_outputs[0]->machine_view.hash();
  /* std::cout << "Partition op machine_view: " << *(MachineView const *)mv
            << std::endl; */
  IndexLauncher launcher(REPARTITION_FWD_TASK_ID,
                         batch_outputs[0]->parallel_is,
                         TaskArgument(&data_type, sizeof(DataType)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_region_requirement(
      RegionRequirement(inference_input_lps[batch_inputs[0]],
                        0 /*projection id*/,
                        READ_ONLY,
                        EXCLUSIVE,
                        batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  return runtime->execute_index_space(ctx, launcher);
}

void Repartition::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  assert(inputs[0]->data_type == outputs[0]->data_type);
  DataType data_type = inputs[0]->data_type;
  IndexLauncher launcher(REPARTITION_FWD_TASK_ID,
                         outputs[0]->parallel_is,
                         TaskArgument(&data_type, sizeof(DataType)),
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

void Repartition::backward(FFModel const &ff) {
  // skip backpropagation for input
  if (inputs[0]->owner_op != nullptr &&
      inputs[0]->owner_op->op_type == OP_INPUT) {
    return;
  }
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  assert(inputs[0]->data_type == outputs[0]->data_type);
  DataType data_type = inputs[0]->data_type;
  IndexLauncher launcher(REPARTITION_BWD_TASK_ID,
                         inputs[0]->parallel_is,
                         TaskArgument(&data_type, sizeof(DataType)),
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

bool Repartition::measure_operator_cost(Simulator *sim,
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

bool Repartition::get_int_parameter(PMParameter para, int *value) const {
  switch (para) {
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

bool Repartition::append_parallel_op_info(
    std::vector<ParallelOpInfo> &parallel_ops) const {
  ParallelOpInfo ret;
  ret.op_type = op_type;
  ret.parallel_dim = repartition_dim;
  ret.parallel_degree = repartition_degree;
  parallel_ops.push_back(ret);
  return true;
}

tl::optional<RecordFormatter> Repartition::as_dot() const {
  RecordFormatter rf;
  {
    std::ostringstream oss;
    oss << "dim(" << this->repartition_dim << ")";
    rf << oss.str();
  }
  {
    std::ostringstream oss;
    oss << "deg(" << this->repartition_degree << ")";
    rf << oss.str();
  }
  return rf;
}

/*static*/
void Repartition::forward_task(Task const *task,
                               std::vector<PhysicalRegion> const &regions,
                               Context ctx,
                               Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  DataType data_type = *((DataType *)task->args);
  if (data_type == DT_FLOAT) {
    forward_task_with_type<float>(task, regions, ctx, runtime);
  } else if (data_type == DT_DOUBLE) {
    forward_task_with_type<double>(task, regions, ctx, runtime);
  } else if (data_type == DT_INT32) {
    forward_task_with_type<int32_t>(task, regions, ctx, runtime);
  } else if (data_type == DT_INT64) {
    forward_task_with_type<int64_t>(task, regions, ctx, runtime);
  } else {
    assert(false && "Unsupported data type in Repartition forward");
  }
}

template <typename DT>
void Repartition::forward_task_with_type(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(output_domain == input_domain);

  const DT *input_ptr = helperGetTensorPointerRO<DT>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  DT *output_ptr = helperGetTensorPointerWO<DT>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);

  forward_kernel<DT>(input_ptr, output_ptr, output_domain.get_volume());
}

void Repartition::backward_task(Task const *task,
                                std::vector<PhysicalRegion> const &regions,
                                Context ctx,
                                Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  DataType data_type = *((DataType *)task->args);
  if (data_type == DT_FLOAT) {
    backward_task_with_type<float>(task, regions, ctx, runtime);
  } else if (data_type == DT_DOUBLE) {
    backward_task_with_type<double>(task, regions, ctx, runtime);
  } else if (data_type == DT_INT32) {
    backward_task_with_type<int32_t>(task, regions, ctx, runtime);
  } else if (data_type == DT_INT64) {
    backward_task_with_type<int64_t>(task, regions, ctx, runtime);
  } else {
    assert(false && "Unsupported data type in Embedding forward");
  }
}

template <typename DT>
void Repartition::backward_task_with_type(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  Domain output_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain input_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(output_grad_domain == input_grad_domain);

  const DT *output_grad_ptr = helperGetTensorPointerRO<DT>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  DT *input_grad_ptr = helperGetTensorPointerRW<DT>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);

  backward_kernel<DT>(
      output_grad_ptr, input_grad_ptr, output_grad_domain.get_volume());
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::RepartitionParams>::operator()(
    FlexFlow::RepartitionParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.repartition_legion_dim);
  hash_combine(key, params.repartition_degree);
  return key;
}
}; // namespace std

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

#include "flexflow/parallel_ops/combine.h"
#include "flexflow/model.h"
#include "flexflow/parallel_ops/kernels/combine_kernels.h"
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

using namespace FlexFlow::Kernels::Combine;

/* Params */
bool operator==(CombineParams const &lhs, CombineParams const &rhs) {
  return lhs.combine_legion_dim == rhs.combine_legion_dim &&
         lhs.combine_degree == rhs.combine_degree;
}

bool CombineParams::is_valid(ParallelTensorShape const &input) const {
  bool valid = input.is_valid();
  valid &=
      (input.dims[this->combine_legion_dim].degree % this->combine_degree == 0);
  return valid;
}

CombineParams Combine::get_params() const {
  CombineParams params;
  params.combine_legion_dim = this->combine_dim;
  params.combine_degree = this->combine_degree;
  if (this->name != nullptr) {
    strcpy(params.name, this->name);
  }
  return params;
}

Combine::Combine(FFModel &model,
                 CombineParams const &params,
                 ParallelTensor const input,
                 char const *name)
    : Combine(model,
              input,
              params.combine_legion_dim,
              params.combine_degree,
              params.name) {}

Combine::Combine(FFModel &model,
                 const ParallelTensor _input,
                 int _combine_legion_dim,
                 int _combine_degree,
                 char const *name)
    : ParallelOp(model, OP_COMBINE, name, _input),
      combine_dim(_combine_legion_dim), combine_degree(_combine_degree) {
  int numdim = _input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = _input->dims[i];
  }
  assert(combine_degree > 0 && "Must use combine_degree > 0");
  assert(dims[combine_dim].degree % combine_degree == 0);
  dims[combine_dim].degree /= combine_degree;
  ParallelTensorBase::update_parallel_ids(numdim, dims);
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, _input->data_type, this);
  // inputs[0]->print("Combine::input");
  // outputs[0]->print("Combine::output");
}

OpMeta *Combine::init_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  Combine *cmb = (Combine *)task->args;
  FFHandler handle = *((FFHandler *)task->local_args);
  CombineMeta *m = new CombineMeta(handle);
  m->input_type[0] = cmb->inputs[0]->data_type;
  m->output_type[0] = cmb->outputs[0]->data_type;
  assert(m->input_type[0] == m->output_type[0]);
  return m;
}

void Combine::init(FFModel const &ff) {
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(COMBINE_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Combine)),
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
  set_opmeta_from_futuremap(ff, fm);
}

void Combine::init_inference(FFModel const &ff,
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
  set_argumentmap_for_init_inference(ff, argmap, batch_outputs[0]);
  IndexLauncher launcher(COMBINE_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Combine)),
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
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

void Combine::create_input_partition(FFModel &ff) {
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

void Combine::create_input_partition_inference(
    FFModel &ff,
    std::vector<ParallelTensor> const &batch_inputs,
    std::vector<ParallelTensor> const &batch_outputs) {
  assert(ff.config.computationMode == COMP_MODE_INFERENCE);
  assert(batch_outputs[0]->part != LogicalPartition::NO_PART);
  assert(batch_inputs[0]->part != LogicalPartition::NO_PART);
  // input_lp is a disjoint partition
  ff.create_disjoint_partition(batch_outputs[0]->num_dims,
                               batch_outputs[0]->dims,
                               batch_outputs[0]->parallel_is,
                               batch_inputs[0]->region,
                               inference_input_lps[batch_inputs[0]]);
}

FutureMap Combine::inference(FFModel const &ff,
                             BatchConfigFuture const &bc,
                             std::vector<ParallelTensor> const &batch_inputs,
                             std::vector<ParallelTensor> const &batch_outputs,
                             MachineView const *mv) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  parallel_is = batch_outputs[0]->parallel_is;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  assert(batch_inputs[0]->data_type == batch_outputs[0]->data_type);
  DataType data_type = batch_inputs[0]->data_type;
  size_t machine_view_hash =
      mv ? mv->hash() : batch_outputs[0]->machine_view.hash();
  set_argumentmap_for_inference(ff, argmap, batch_outputs[0]);
  IndexLauncher launcher(COMBINE_FWD_TASK_ID,
                         batch_outputs[0]->parallel_is,
                         TaskArgument(nullptr, 0),
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

void Combine::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  assert(inputs[0]->data_type == outputs[0]->data_type);
  DataType data_type = inputs[0]->data_type;
  IndexLauncher launcher(COMBINE_FWD_TASK_ID,
                         outputs[0]->parallel_is,
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
  runtime->execute_index_space(ctx, launcher);
}

void Combine::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  assert(inputs[0]->data_type == outputs[0]->data_type);
  DataType data_type = inputs[0]->data_type;
  IndexLauncher launcher(COMBINE_BWD_TASK_ID,
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

bool Combine::measure_operator_cost(Simulator *sim,
                                    MachineView const &mv,
                                    CostMetrics &cost_metrics) const {
  // TODO: to be implemented
  cost_metrics = CostMetrics();
  cost_metrics.forward_time = 0.05f;
  cost_metrics.backward_time = 0.05f;
  return true;
}

bool Combine::get_int_parameter(PMParameter para, int *value) const {
  switch (para) {
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

bool Combine::append_parallel_op_info(
    std::vector<ParallelOpInfo> &parallel_ops) const {
  ParallelOpInfo ret;
  ret.op_type = op_type;
  ret.parallel_dim = combine_dim;
  ret.parallel_degree = combine_degree;
  parallel_ops.push_back(ret);
  return true;
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
void Combine::forward_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  CombineMeta const *m = *((CombineMeta **)task->local_args);
  DataType data_type = m->input_type[0];
  if (data_type == DT_HALF) {
    forward_task_with_type<half>(task, regions, ctx, runtime);
  } else if (data_type == DT_FLOAT) {
    forward_task_with_type<float>(task, regions, ctx, runtime);
  } else if (data_type == DT_DOUBLE) {
    forward_task_with_type<double>(task, regions, ctx, runtime);
  } else if (data_type == DT_INT32) {
    forward_task_with_type<int32_t>(task, regions, ctx, runtime);
  } else if (data_type == DT_INT64) {
    forward_task_with_type<int64_t>(task, regions, ctx, runtime);
  } else {
    assert(false && "Unsupported data type in Combine forward");
  }
}

template <typename DT>
void Combine::forward_task_with_type(Task const *task,
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

void Combine::backward_task(Task const *task,
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
    assert(false && "Unsupported data type in Combine backward");
  }
}

template <typename DT>
void Combine::backward_task_with_type(
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
size_t hash<FlexFlow::CombineParams>::operator()(
    FlexFlow::CombineParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.combine_legion_dim);
  hash_combine(key, params.combine_degree);
  return key;
}
}; // namespace std

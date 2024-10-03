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

#include "flexflow/parallel_ops/allreduce.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/model.h"
#include "flexflow/parallel_ops/kernels/allreduce_kernels.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::Future;
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

using namespace FlexFlow::Kernels::AllReduce;

/* Params */
bool operator==(AllReduceParams const &lhs, AllReduceParams const &rhs) {
  return lhs.allreduce_legion_dim == rhs.allreduce_legion_dim &&
         std::strcmp(lhs.name, rhs.name) == 0;
}

bool AllReduceParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

AllReduceParams AllReduce::get_params() const {
  AllReduceParams params;
  params.allreduce_legion_dim = this->allreduce_dim;
  if (strlen(this->name) < MAX_OPNAME) {
    strcpy(params.name, this->name);
  }
  return params;
}

AllReduce::AllReduce(FFModel &model,
                     const ParallelTensor _input,
                     int _allreduce_legion_dim,
                     char const *name)
    : ParallelOp(model, OP_ALLREDUCE, name, _input),
      allreduce_dim(_allreduce_legion_dim) {
  int numdim = _input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = _input->dims[i];
  }
  assert(dims[allreduce_dim].degree > 1);
  // ParallelTensorBase::update_parallel_ids(numdim, dims);
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, _input->data_type, this);
}

AllReduce::AllReduce(FFModel &model,
                     AllReduceParams const &params,
                     ParallelTensor const input,
                     char const *name)
    : AllReduce(model, input, params.allreduce_legion_dim, params.name) {}

void AllReduce::create_input_partition(FFModel &ff) {
  // Do nothing
  return;
}

void AllReduce::create_input_partition_inference(
    FFModel &ff,
    std::vector<ParallelTensor> const &batch_inputs,
    std::vector<ParallelTensor> const &batch_outputs) {
  assert(ff.config.computationMode == COMP_MODE_INFERENCE);
  assert(batch_outputs[0]->part != LogicalPartition::NO_PART);
  assert(batch_inputs[0]->part != LogicalPartition::NO_PART);
  // Do nothing
  return;
}

OpMeta *AllReduce::init_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  AllReduce *ar = (AllReduce *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  AllReduceMeta *meta = new AllReduceMeta(handle, ar);
  meta->input_type[0] = ar->inputs[0]->data_type;
  meta->output_type[0] = ar->outputs[0]->data_type;
  assert(meta->input_type[0] == meta->output_type[0]);
  std::strcpy(meta->op_name, ar->name);
  return meta;
}

void AllReduce::init(FFModel const &ff) {
  ArgumentMap argmap;
  parallel_is = outputs[0]->parallel_is;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(ALLREDUCE_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(AllReduce)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
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

void AllReduce::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  parallel_is = outputs[0]->parallel_is;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(ALLREDUCE_FWD_TASK_ID,
                         outputs[0]->parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.concurrent = true;
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

/*static*/
void AllReduce::forward_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);

  AllReduceMeta const *m = *((AllReduceMeta **)task->local_args);

  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);

  assert(input.data_type == output.data_type);
  forward_kernel_wrapper(m, input, output);
}

void AllReduce::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  assert(numOutputs == 1);
  assert(numInputs == 1);
  IndexLauncher launcher(ALLREDUCE_BWD_TASK_ID,
                         inputs[0]->parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         inputs[0]->machine_view.hash());
  launcher.concurrent = true;
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void AllReduce::backward_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  AllReduceMeta const *m = *((AllReduceMeta **)task->local_args);

  GenericTensorAccessorW input_grad = helperGetGenericTensorAccessorRW(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR output_grad = helperGetGenericTensorAccessorRO(
      m->output_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);

  assert(input_grad.data_type == output_grad.data_type);
  backward_kernel_wrapper(m, input_grad, output_grad);
}

void AllReduce::init_inference(FFModel const &ff,
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
  IndexLauncher launcher(ALLREDUCE_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(AllReduce)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
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

FutureMap AllReduce::inference(FFModel const &ff,
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
  IndexLauncher launcher(ALLREDUCE_INF_TASK_ID,
                         batch_outputs[0]->parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
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

/*static*/
void AllReduce::inference_task(Task const *task,
                               std::vector<PhysicalRegion> const &regions,
                               Context ctx,
                               Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);

  AllReduceMeta *m = *((AllReduceMeta **)task->local_args);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_active_tokens() == 0) {
    return;
  }

  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);

  assert(input.data_type == output.data_type);
  inference_kernel_wrapper(m, bc, input, output);
  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    AllReduce::save_inference_tensors_to_file(
        m, shard_id, bc, {input}, {}, {output});
  }
}

FutureMap AllReduce::peft_bwd(FFModel const &ff,
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
  IndexLauncher launcher(ALLREDUCE_PEFT_BWD_TASK_ID,
                         batch_outputs[0]->parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  launcher.add_region_requirement(
      RegionRequirement(batch_inputs[0]->part_grad,
                        0 /*projection id*/,
                        WRITE_ONLY,
                        EXCLUSIVE,
                        batch_inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(batch_outputs[0]->part_grad,
                        0 /*projection id*/,
                        READ_WRITE,
                        EXCLUSIVE,
                        batch_outputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  return runtime->execute_index_space(ctx, launcher);
}

/*static*/
void AllReduce::peft_bwd_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);

  AllReduceMeta *m = *((AllReduceMeta **)task->local_args);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_active_peft_tokens() == 0) {
    return;
  }
  GenericTensorAccessorW input_grad = helperGetGenericTensorAccessorRW(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR output_grad = helperGetGenericTensorAccessorRO(
      m->output_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);

  assert(input_grad.data_type == output_grad.data_type);
  peft_bwd_kernel_wrapper(m, bc, input_grad, output_grad);
  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    AllReduce::save_inference_tensors_to_file(
        m, shard_id, bc, {input_grad}, {}, {output_grad}, false);
  }
}

bool AllReduce::measure_operator_cost(Simulator *sim,
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

bool AllReduce::get_int_parameter(PMParameter para, int *value) const {
  switch (para) {
    case PM_ALLREDUCE_DIM:
      *value = allreduce_dim;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

bool AllReduce::append_parallel_op_info(
    std::vector<ParallelOpInfo> &parallel_ops) const {
  ParallelOpInfo ret;
  ret.op_type = op_type;
  ret.parallel_dim = allreduce_dim;
  ret.parallel_degree = -1; // AllReduce does not affect parallel degree
  parallel_ops.push_back(ret);
  return true;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::AllReduceParams>::operator()(
    FlexFlow::AllReduceParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.allreduce_legion_dim);
  return key;
}

} // namespace std

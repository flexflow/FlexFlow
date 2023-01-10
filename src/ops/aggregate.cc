/* Copyright 2019 Stanford
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

#include "flexflow/ops/aggregate.h"

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

Tensor FFModel::aggregate(
    Tensor const
        *inputs, /* gate_preds, gate_assign, full_gate_pred, n * exp_pred */
    int n,
    float lambda_bal,
    char const *name) {
#ifdef DEADCODE
  Aggregate *aggr = new Aggregate(*this, inputs, n, lambda_bal, name);
  layers.push_back(aggr);
  return aggr->outputs[0];
#endif
  return nullptr;
}

Aggregate::Aggregate(FFModel &model,
                     ParallelTensor const *_inputs,
                     int _n,
                     float _lambda_bal,
                     char const *name)
    : Op(model,
         OP_AGGREGATE,
         DT_FLOAT,
         name,
         _n + 4 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         _inputs),
      n(_n), lambda_bal(_lambda_bal) {
  // FIXME: For now, set upper limits Better: Do as follows, but memory is
  // assigned per block, so requires to check that
  // https://stackoverflow.com/questions/5531247/allocating-shared-memory/5531640#5531640
  assert(n <= AGGREGATE_MAX_N && "Increase AGGREGATE_MAX_N in #define");
  assert(inputs[0]->dims[0].size <= AGGREGATE_MAX_K &&
         "Increase AGGREGATE_MAX_K in #define");
  assert(inputs[0]->dims[1].size <= AGGREGATE_MAX_BATCH_SIZE &&
         "Increase AGGREGATE_MAX_BATCH_SIZE in #define");

  assert(n + 4 == numInputs);
  assert(n > 0);
  assert(inputs[0]->num_dims == 2);
  assert(inputs[1]->num_dims == 2);
  assert(inputs[2]->num_dims == 2);
  assert(inputs[3]->num_dims == 2);

  for (int i = 0; i < inputs[0]->num_dims; i++) {
    assert(inputs[0]->dims[i] == inputs[1]->dims[i]);
    assert(inputs[0]->dims[i] == inputs[2]->dims[i]);
  }
  assert(inputs[0]->dims[1] == inputs[3]->dims[1]);
  assert(inputs[3]->dims[0].size == n);

  // expert inputs
  int num_dim = inputs[4]->num_dims;
  int out_dim = inputs[4]->dims[0].size;
  for (int i = 1; i < n; i++) {
    assert(inputs[i + 4]->num_dims == num_dim);
    assert(inputs[i + 4]->dims[0].size == out_dim);
  }
  // Set output shape
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < num_dim - 1; i++) {
    dims[i] = inputs[4]->dims[i];
  }
  dims[num_dim - 1] = inputs[0]->dims[num_dim - 1];
  numOutputs = 1;
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      num_dim, dims, DT_FLOAT, this);

  numWeights = 0;
}

void Aggregate::init(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  parallel_is = outputs[0]->parallel_is;
  IndexLauncher launcher(AGGREGATE_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Aggregate)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *Aggregate::init_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  Aggregate *agg = (Aggregate *)task->args;
  FFHandler handle = *((FFHandler *)task->local_args);
  AggregateMeta *m = new AggregateMeta(handle, agg->n);
  m->profiling = agg->profiling;
  return m;
}

void Aggregate::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  parallel_is = outputs[0]->parallel_is;
  IndexLauncher launcher(AGGREGATE_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Aggregate)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // gate_preds
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // gate_assign
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // exp_preds
  for (int i = 0; i < n; i++) {
    launcher.add_region_requirement(RegionRequirement(inputs[i + 4]->part,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      inputs[i + 4]->region));
    launcher.add_field(i + 2, FID_DATA);
  }
  // output
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(n + 2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Aggregate::forward_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  int n = ((Aggregate *)task->args)->n;

  assert((int)regions.size() == n + 3);
  assert((int)task->regions.size() == n + 3);

  AggregateMeta const *m = *((AggregateMeta **)task->local_args);

  // get gate_pred, gate_assign, output
  AccessorRO<float, 2> const acc_gate_pred(regions[0], FID_DATA);
  AccessorRO<int, 2> const acc_gate_assign(regions[1], FID_DATA);
  AccessorWO<float, 2> const acc_output(regions[n + 2], FID_DATA);

  Rect<2> rect_gate_pred = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_gate_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_output = runtime->get_index_space_domain(
      ctx, task->regions[n + 2].region.get_index_space());

  coord_t batch_size = rect_gate_pred.hi[1] - rect_gate_pred.lo[1] + 1;
  assert(batch_size == rect_gate_assign.hi[1] - rect_gate_assign.lo[1] + 1);
  assert(rect_gate_pred.hi[0] - rect_gate_pred.lo[0] ==
         rect_gate_assign.hi[0] - rect_gate_assign.lo[0]);
  assert(batch_size == rect_output.hi[1] - rect_output.lo[1] + 1);
  coord_t out_dim = rect_output.hi[0] - rect_output.lo[0] + 1;

  // get exp_preds
  float *exp_preds[n];
  // get first exp_pred and row and out_dim
  Domain exp_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  exp_preds[0] = helperGetTensorPointerWO<float>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  coord_t rows = exp_domain.hi()[1] - exp_domain.lo()[1] + 1;
  assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);

  for (int i = 1; i < n; i++) {
    exp_domain = runtime->get_index_space_domain(
        ctx, task->regions[i + 2].region.get_index_space());
    exp_preds[i] = helperGetTensorPointerWO<float>(
        regions[i + 2], task->regions[i + 2], FID_DATA, ctx, runtime);

    assert(rows == exp_domain.hi()[1] - exp_domain.lo()[1] + 1);
    assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);
  }

  int k = (int)(rect_gate_assign.hi[0] - rect_gate_assign.lo[0] + 1);

  Aggregate::forward_kernel_wrapper(m,
                                    exp_preds,
                                    acc_gate_assign.ptr(rect_gate_assign),
                                    acc_gate_pred.ptr(rect_gate_pred),
                                    acc_output.ptr(rect_output),
                                    n,
                                    k,
                                    rows,
                                    batch_size,
                                    out_dim);
}

void Aggregate::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  parallel_is = outputs[0]->parallel_is;
  IndexLauncher launcher(AGGREGATE_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Aggregate)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // gate_preds
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // gate_assign
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // true gate_assign
  launcher.add_region_requirement(RegionRequirement(inputs[2]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[2]->region));
  launcher.add_field(2, FID_DATA);
  // full_gate gradients
  launcher.add_region_requirement(RegionRequirement(inputs[3]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[3]->region_grad));
  launcher.add_field(3, FID_DATA);
  // exp_preds
  for (int i = 0; i < n; i++) {
    launcher.add_region_requirement(RegionRequirement(inputs[i + 4]->part,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      inputs[i + 4]->region));
    launcher.add_field(i + 4, FID_DATA);
  }
  // exp_preds gradients
  for (int i = 0; i < n; i++) {
    launcher.add_region_requirement(
        RegionRequirement(inputs[i + 4]->part_grad,
                          0 /*projection id*/,
                          READ_WRITE,
                          EXCLUSIVE,
                          inputs[i + 4]->region_grad));
    launcher.add_field(i + n + 4, FID_DATA);
  }

  // output
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(2 * n + 4, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

void Aggregate::backward_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  AggregateMeta const *m = *((AggregateMeta **)task->local_args);
  int n = ((Aggregate *)task->args)->n;
  float lambda_bal = ((Aggregate *)task->args)->lambda_bal;

  assert((int)regions.size() == 2 * n + 5);
  assert((int)task->regions.size() == 2 * n + 5);

  // get gate_pred, gate_grad, gate_assign, output_grad
  AccessorRO<float, 2> const acc_gate_pred(regions[0], FID_DATA);
  AccessorRO<int, 2> const acc_gate_assign(regions[1], FID_DATA);
  AccessorRO<int, 2> const acc_true_gate_assign(regions[2], FID_DATA);
  AccessorWO<float, 2> const full_acc_gate_grad(regions[3], FID_DATA);
  AccessorRO<float, 2> const acc_output_grad(regions[2 * n + 4], FID_DATA);

  Rect<2> rect_gate_pred = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_gate_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_true_gate_assign = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  Rect<2> rect_full_gate_grad = runtime->get_index_space_domain(
      ctx, task->regions[3].region.get_index_space());
  Rect<2> rect_out_grad = runtime->get_index_space_domain(
      ctx, task->regions[2 * n + 4].region.get_index_space());

  coord_t batch_size = rect_gate_pred.hi[1] - rect_gate_pred.lo[1] + 1;
  assert(batch_size == rect_gate_assign.hi[1] - rect_gate_assign.lo[1] + 1);
  assert(rect_gate_assign == rect_true_gate_assign);
  assert(batch_size == rect_out_grad.hi[1] - rect_out_grad.lo[1] + 1);
  assert(batch_size ==
         rect_full_gate_grad.hi[1] - rect_full_gate_grad.lo[1] + 1);
  coord_t k = rect_gate_assign.hi[0] - rect_gate_assign.lo[0] + 1;
  assert(rect_gate_pred.hi[0] - rect_gate_pred.lo[0] + 1 == k);
  coord_t out_dim = rect_out_grad.hi[0] - rect_out_grad.lo[0] + 1;
  assert(n == rect_full_gate_grad.hi[0] - rect_full_gate_grad.lo[0] + 1);

  // get exp_preds
  float *exp_preds[n];
  // get first exp_pred and row
  Domain exp_domain = runtime->get_index_space_domain(
      ctx, task->regions[4].region.get_index_space());
  exp_preds[0] = helperGetTensorPointerRW<float>(
      regions[4], task->regions[4], FID_DATA, ctx, runtime);
  coord_t rows = exp_domain.hi()[1] - exp_domain.lo()[1] + 1;
  assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);

  for (int i = 1; i < n; i++) {
    exp_domain = runtime->get_index_space_domain(
        ctx, task->regions[i + 4].region.get_index_space());
    exp_preds[i] = helperGetTensorPointerRW<float>(
        regions[i + 4], task->regions[i + 4], FID_DATA, ctx, runtime);
    assert(rows == exp_domain.hi()[1] - exp_domain.lo()[1] + 1);
    assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);
  }

  // get chosen_exp_grads
  float *exp_grads[n];
  for (int i = 0; i < n; i++) {
    exp_domain = runtime->get_index_space_domain(
        ctx, task->regions[n + i + 4].region.get_index_space());
    exp_grads[i] = helperGetTensorPointerRW<float>(
        regions[n + i + 4], task->regions[n + i + 4], FID_DATA, ctx, runtime);
    assert(rows == exp_domain.hi()[1] - exp_domain.lo()[1] + 1);
    assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);
  }

  Aggregate::backward_kernel_wrapper(
      m,
      exp_preds,
      exp_grads,
      acc_gate_assign.ptr(rect_gate_assign),
      acc_true_gate_assign.ptr(rect_true_gate_assign),
      acc_gate_pred.ptr(rect_gate_pred),
      full_acc_gate_grad.ptr(rect_full_gate_grad),
      acc_output_grad.ptr(rect_out_grad),
      n,
      k,
      rows,
      lambda_bal,
      batch_size,
      out_dim);
}

bool Aggregate::measure_operator_cost(Simulator *sim,
                                      MachineView const &mv,
                                      CostMetrics &cost_metrics) const {
  assert(numInputs <= MAX_NUM_INPUTS);
  ParallelTensorBase sub_inputs[MAX_NUM_INPUTS], sub_output;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }

  for (int i=0; i<numInputs, ++i) {
    if (!inputs[i]->get_sub_tensor(mv, sub_inputs[i])) {
      return false;
    }
  }

  AggregateMeta *m = new AggregateMeta(sim->handler, n);

  // inputs memory
  sim->free_all();
  float *input_ptrs[MAX_NUM_INPUTS];
  bool out_of_memory = false;
  for (int i = 0; i < numInputs; i++) {
    input_ptrs[i] =
        (float *)sim->allocate(sub_inputs[i].get_volume(), DT_FLOAT);
    out_of_memory = out_of_memory || (input_ptrs[i] == NULL);
  }
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  // outputs memory
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
  out_of_memory = out_of_memory || (output_ptr == NULL);

  if (out_of_memory) {
    cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    return true;
  }
  assert(m->profiling == false);

  // time
  std::function<void()> forward, backward;
  // forward

  /*forward = [&] {
    forward_kernel_wrapper(m,);
  };*/
  if (sim->computationMode == COMP_MODE_TRAINING) {
    // backward


    /*backward = [&] {
      backward_kernel_wrapper(m,);
    };*/
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);
  if (sim->computationMode == COMP_MODE_TRAINING) {
    log_measure.debug(
      "[Measure Aggregate] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
      name,
      cost_metrics.forward_time,
      cost_metrics.backward_time);
  } else {
    log_measure.debug(
      "[Measure Aggregate] name(%s) forward_time(%.4lf)\n",
      name,
      cost_metrics.forward_time);
  }

  delete m;
  return true;
}

}; // namespace FlexFlow

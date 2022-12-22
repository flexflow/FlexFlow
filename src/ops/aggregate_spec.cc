/* Copyright 2021 Stanford, Facebook
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

#include "flexflow/ops/aggregate_spec.h"

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

Tensor FFModel::aggregate_spec(
    Tensor const
        *inputs, /* gate_preds, gate_assign, full_gate_pred, n * exp_pred */
    int n,
    float lambda_bal,
    char const *name) {
  assert(false);
}

AggregateSpec::AggregateSpec(FFModel &model,
                             ParallelTensor const *_inputs,
                             int _n,
                             float _lambda_bal,
                             char const *name)
    : Op(model,
         OP_AGG_SPEC,
         DT_FLOAT,
         name,
         _n + 4 /*numInputs*/,
         0 /*numWeights*/,
         1 /*numOutputs*/,
         _inputs),
      n(_n), lambda_bal(_lambda_bal) {
  // FIXME: For now, set upper limits Better: Do as follows, but memory is
  // assigned per block, so requires to check that
  // https://stackoverflow.com/questions/5531247/allocating-shared-memory/5531640#5531640
  assert(n <= AGGREGATE_SPEC_MAX_N &&
         "Increase AGGREGATE_SPEC_MAX_N in #define");
  assert(inputs[0]->dims[0].size <= AGGREGATE_SPEC_MAX_K &&
         "Increase AGGREGATE_SPEC_MAX_K in #define");
  assert(inputs[0]->dims[1].size <= AGGREGATE_SPEC_MAX_BATCH_SIZE &&
         "Increase AGGREGATE_SPEC_MAX_BATCH_SIZE in #define");

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

void AggregateSpec::init(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  parallel_is = outputs[0]->parallel_is;
  IndexLauncher launcher(AGG_SPEC_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(AggregateSpec)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *AggregateSpec::init_task(Task const *task,
                                 std::vector<PhysicalRegion> const &regions,
                                 Context ctx,
                                 Runtime *runtime) {
  AggregateSpec *agg = (AggregateSpec *)task->args;
  FFHandler handle = *((FFHandler *)task->local_args);
  AggregateSpecMeta *m = new AggregateSpecMeta(handle, agg->n);
  m->profiling = agg->profiling;
  return m;
}

void AggregateSpec::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  parallel_is = outputs[0]->parallel_is;
  IndexLauncher launcher(AGG_SPEC_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(AggregateSpec)),
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

void AggregateSpec::forward_task(Task const *task,
                                 std::vector<PhysicalRegion> const &regions,
                                 Context ctx,
                                 Runtime *runtime) {
  int n = ((AggregateSpec *)task->args)->n;

  assert((int)regions.size() == n + 3);
  assert((int)task->regions.size() == n + 3);

  AggregateSpecMeta const *m = *((AggregateSpecMeta **)task->local_args);

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
  coord_t k = rect_gate_pred.hi[0] - rect_gate_pred.lo[0] + 1;
  assert(k == rect_gate_assign.hi[0] - rect_gate_assign.lo[0] + 1);
  assert(k * batch_size == rect_output.hi[1] - rect_output.lo[1] + 1);
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

  AggregateSpec::forward_kernel_wrapper(m,
                                        exp_preds,
                                        acc_gate_assign.ptr(rect_gate_assign),
                                        acc_output.ptr(rect_output),
                                        n,
                                        k,
                                        rows,
                                        batch_size,
                                        out_dim);
}

void AggregateSpec::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  parallel_is = outputs[0]->parallel_is;
  IndexLauncher launcher(AGG_SPEC_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(AggregateSpec)),
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

  // gate gradients full
  launcher.add_region_requirement(RegionRequirement(inputs[3]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[3]->region_grad));
  launcher.add_field(3, FID_DATA);

  // exp gradients
  for (int i = 0; i < n; i++) {
    launcher.add_region_requirement(
        RegionRequirement(inputs[i + 4]->part_grad,
                          0 /*projection id*/,
                          READ_WRITE,
                          EXCLUSIVE,
                          inputs[i + 4]->region_grad));
    launcher.add_field(i + 4, FID_DATA);
  }

  // output
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(n + 4, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

void AggregateSpec::backward_task(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime) {
  AggregateSpecMeta const *m = *((AggregateSpecMeta **)task->local_args);
  int n = ((AggregateSpec *)task->args)->n;
  float lambda_bal = ((AggregateSpec *)task->args)->lambda_bal;

  assert((int)regions.size() == n + 5);
  assert((int)task->regions.size() == n + 5);

  // get gate_pred, gate_assin, full_gate_grad, output_grad
  AccessorRO<float, 2> const acc_gate_pred(regions[0], FID_DATA);
  AccessorRO<int, 2> const acc_gate_assign(regions[1], FID_DATA);
  AccessorRO<int, 2> const acc_true_gate_assign(regions[2], FID_DATA);
  AccessorWO<float, 2> const acc_full_gate_grad(regions[3], FID_DATA);
  AccessorRO<float, 2> const acc_output_grad(regions[n + 4], FID_DATA);

  Rect<2> rect_gate_pred = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_gate_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_true_gate_assign = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  Rect<2> rect_full_gate_grad = runtime->get_index_space_domain(
      ctx, task->regions[3].region.get_index_space());
  Rect<2> rect_out_grad = runtime->get_index_space_domain(
      ctx, task->regions[n + 4].region.get_index_space());

  coord_t batch_size = rect_gate_pred.hi[1] - rect_gate_pred.lo[1] + 1;
  assert(batch_size == rect_gate_assign.hi[1] - rect_gate_assign.lo[1] + 1);
  assert(rect_gate_assign == rect_true_gate_assign);
  assert(batch_size ==
         rect_full_gate_grad.hi[1] - rect_full_gate_grad.lo[1] + 1);
  coord_t k = rect_gate_assign.hi[0] - rect_gate_assign.lo[0] + 1;
  assert(k * batch_size == rect_out_grad.hi[1] - rect_out_grad.lo[1] + 1);
  assert(rect_gate_pred.hi[0] - rect_gate_pred.lo[0] + 1 == k);
  coord_t out_dim = rect_out_grad.hi[0] - rect_out_grad.lo[0] + 1;
  assert(n == rect_full_gate_grad.hi[0] - rect_full_gate_grad.lo[0] + 1);

  // get exp_preds
  float *exp_grads[n];
  // get first exp_pred and row
  Domain exp_domain = runtime->get_index_space_domain(
      ctx, task->regions[4].region.get_index_space());
  exp_grads[0] = helperGetTensorPointerRW<float>(
      regions[4], task->regions[4], FID_DATA, ctx, runtime);
  coord_t rows = exp_domain.hi()[1] - exp_domain.lo()[1] + 1;
  assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);

  for (int i = 1; i < n; i++) {
    exp_domain = runtime->get_index_space_domain(
        ctx, task->regions[i + 4].region.get_index_space());
    exp_grads[i] = helperGetTensorPointerRW<float>(
        regions[i + 4], task->regions[i + 4], FID_DATA, ctx, runtime);
    assert(rows == exp_domain.hi()[1] - exp_domain.lo()[1] + 1);
    assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);
  }

  AggregateSpec::backward_kernel_wrapper(
      m,
      exp_grads,
      acc_gate_assign.ptr(rect_gate_assign),
      acc_true_gate_assign.ptr(rect_true_gate_assign),
      acc_gate_pred.ptr(rect_gate_pred),
      acc_full_gate_grad.ptr(rect_full_gate_grad),
      acc_output_grad.ptr(rect_out_grad),
      n,
      k,
      rows,
      lambda_bal,
      batch_size,
      out_dim);
}

bool AggregateSpec::measure_operator_cost(Simulator *sim,
                                          MachineView const &mv,
                                          CostMetrics &cost_metrics) const {
  // TODO: implement
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;
  cost_metrics.inputs_memory = 0;
  cost_metrics.outputs_memory = 0;
  cost_metrics.weights_memory = 0;
  return false;
}

}; // namespace FlexFlow

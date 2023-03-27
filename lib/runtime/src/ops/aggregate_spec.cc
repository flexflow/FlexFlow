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

#include "aggregate_spec.h"
#include "kernels/aggregate_spec_kernels.h"
#include "task_spec.h"

namespace FlexFlow {

enum Slots {
  GATE_PREDS,
  GATE_ASSIGN,
  EXP_PREDS,
  OUTPUT,
  TRUE_GATE_ASSIGN,
  GATE_GRADIENTS_FULL,
  ATTRS,
  PROFILING
};

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

using namespace FlexFlow::Kernels::AggregateSpec;

Tensor FFModel::aggregate_spec(
    Tensor const *inputs, /* gate_preds, gate_assign, gate assign TopK,
                             full_gate_pred, exp_pred_1, ... , exp_pred_n */
    int n,
    float lambda_bal,
    char const *name) {
  Layer *li = new Layer(this,
                        OP_AGG_SPEC,
                        DT_FLOAT,
                        name,
                        n + 4 /*inputs*/,
                        0 /*weights*/,
                        1 /*outputs*/,
                        inputs);
  {
    int num_dim = inputs[4]->num_dims;
    // Set output shape
    int dims[MAX_TENSOR_DIM];
    for (int i = 0; i < num_dim - 1; i++) {
      dims[i] = inputs[4]->dims[i];
    }
    dims[num_dim - 1] = inputs[0]->dims[num_dim - 1];
    li->outputs[0] = create_tensor_legion_ordering(
        num_dim, dims, DT_FLOAT, li, 0, true /*create_grad*/);
  }
  li->add_int_property("n", n);
  li->add_float_property("lambda_bal", lambda_bal);
  layers.push_back(li);
  return li->outputs[0];
}

Op *AggregateSpec::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value1;
  layer->get_int_property("n", value1);
  int n = value1;
  float value2;
  layer->get_float_property("lambda_bal", value2);
  float lambda_bal = value2;
  return new AggregateSpec(model, inputs.data(), n, lambda_bal, layer->name);
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
      attrs(_n, _lambda_bal) {
  // FIXME: For now, set upper limits Better: Do as follows, but memory is
  // assigned per block, so requires to check that
  // https://stackoverflow.com/questions/5531247/allocating-shared-memory/5531640#5531640
  assert(attrs.n <= AGGREGATE_SPEC_MAX_N &&
         "Increase AGGREGATE_SPEC_MAX_N in #define");
  assert(inputs[0]->dims[0].size <= AGGREGATE_SPEC_MAX_K &&
         "Increase AGGREGATE_SPEC_MAX_K in #define");
  assert(inputs[0]->dims[1].size <= AGGREGATE_SPEC_MAX_BATCH_SIZE &&
         "Increase AGGREGATE_SPEC_MAX_BATCH_SIZE in #define");

  assert(attrs.n + 4 == numInputs);
  assert(attrs.n > 0);
  assert(inputs[0]->num_dims == 2);
  assert(inputs[1]->num_dims == 2);
  assert(inputs[2]->num_dims == 2);
  assert(inputs[3]->num_dims == 2);

  for (int i = 0; i < inputs[0]->num_dims; i++) {
    assert(inputs[0]->dims[i] == inputs[1]->dims[i]);
    assert(inputs[0]->dims[i] == inputs[2]->dims[i]);
  }
  assert(inputs[0]->dims[1] == inputs[3]->dims[1]);
  assert(inputs[3]->dims[0].size == attrs.n);

  // expert inputs
  int num_dim = inputs[4]->num_dims;
  int out_dim = inputs[4]->dims[0].size;
  for (int i = 1; i < attrs.n; i++) {
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
static OpTaskSignature get_init_task_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_arg_slot<AggregateSpecAttrs>(ATTRS);
  init.add_arg_slot<bool>(PROFILING);

  return init;
}

static OpTaskSignature get_fwd_task_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<AggregateSpecAttrs>(ATTRS);

  fwd.add_input_slot(GATE_PREDS, READ_WRITE);
  fwd.add_input_slot(GATE_ASSIGN, READ_WRITE);
  fwd.add_input_slot(EXP_PREDS, SlotType::VARIADIC, READ_WRITE);
  fwd.add_output_slot(OUTPUT);

  return fwd;
}

static OpTaskSignature get_bwd_task_signature() {
  OpTaskSignature bwd(OpTaskType::BWD);

  bwd.add_arg_slot<AggregateSpecAttrs>(ATTRS);

  bwd.add_input_slot(GATE_PREDS, READ_WRITE);
  bwd.add_input_slot(GATE_ASSIGN);
  bwd.add_input_slot(TRUE_GATE_ASSIGN);
  bwd.add_input_grad_slot(GATE_GRADIENTS_FULL);
  bwd.add_input_grad_slot(EXP_PREDS, SlotType::VARIADIC);
  bwd.add_output_grad_slot(OUTPUT);

  return bwd;
}

OpTaskBinding AggregateSpec::get_init_task_binding() const {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, this->attrs);
  binding.bind_arg(PROFILING, this->profiling);

  return binding;
}

OpTaskBinding AggregateSpec::get_fwd_task_binding() const {
  OpTaskBinding binding;

  binding.bind(GATE_PREDS, input_tensor(0));
  binding.bind(GATE_ASSIGN, input_tensor(1));

  for (int i = 0; i < this->attrs.n; i++) {
    binding.bind(EXP_PREDS, input_tensor(i+4));
  }

  binding.bind(OUTPUT, output_tensor(0));

  binding.bind_arg(ATTRS, this->attrs);

  return binding;
}

OpTaskBinding AggregateSpec::get_bwd_task_binding() const {
  OpTaskBinding binding;

  binding.bind(GATE_PREDS, input_tensor(0));
  binding.bind(GATE_ASSIGN, input_tensor(1));
  binding.bind(TRUE_GATE_ASSIGN, input_tensor(2));
  binding.bind_grad(GATE_GRADIENTS_FULL, input_tensor(3).grad());

  for (int i = 0; i < attrs.n; i++) {
    binding.bind_grad(EXP_PREDS, input_tensor(i+4).grad());
  }

  binding.bind_grad(OUTPUT, output_tensor(0).grad());
  
  binding.bind_arg(ATTRS, this->attrs);

  return binding;
}

void AggregateSpec::init(FFModel const &ff) {
  this->execute_task(ff, AGG_SPEC_INIT_TASK_ID, get_init_task_signature());

  // assert(check_output_input_weight_same_parallel_is());
  // parallel_is = outputs[0]->parallel_is;
  // ArgumentMap argmap;
  // Context ctx = ff.config.lg_ctx;
  // Runtime *runtime = ff.config.lg_hlr;
  // set_argumentmap_for_init(ff, argmap);
  // IndexLauncher launcher(AGG_SPEC_INIT_TASK_ID,
  //                        parallel_is,
  //                        TaskArgument(this, sizeof(AggregateSpec)),
  //                        argmap,
  //                        Predicate::TRUE_PRED,
  //                        false /*must*/,
  //                        0 /*mapper_id*/,
  //                        outputs[0]->machine_view.hash());
  // FutureMap fm = runtime->execute_index_space(ctx, launcher);
  // fm.wait_all_results();
  // set_opmeta_from_futuremap(ff, fm);
}

PerDeviceOpState *AggregateSpec::init_task(Task const *task,
                                 std::vector<PhysicalRegion> const &regions,
                                 Context ctx,
                                 Runtime *runtime) {
  OpTaskArgumentAccessor acc(task, regions, ctx, runtime);
  auto const &attrs = acc.get_argument<AggregateSpecAttrs>(ATTRS);
  bool profiling = acc.get_argument<bool>(PROFILING);

  FFHandler handle = *((FFHandler *)task->local_args);
  AggregateSpecPerDeviceState *m = new AggregateSpecPerDeviceState(handle, attrs.n);
  m->profiling = profiling;
  return m;
}


void AggregateSpec::forward(FFModel const &ff) {
  this->execute_task(
    ff, AGG_SPEC_FWD_TASK_ID, get_fwd_task_signature()
  );
  
  //ArgumentMap argmap;
  //Context ctx = ff.config.lg_ctx;
  //Runtime *runtime = ff.config.lg_hlr;
  //set_argumentmap_for_forward(ff, argmap);
  //IndexLauncher launcher(AGG_SPEC_FWD_TASK_ID,
  //                       parallel_is,
  //                       TaskArgument(this, sizeof(AggregateSpec)),
  //                       argmap,
  //                       Predicate::TRUE_PRED,
  //                       false /*must*/,
  //                       0 /*mapper_id*/,
  //                       outputs[0]->machine_view.hash());
  //// gate_preds
  //launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
  //                                                  0 /*projection id*/,
  //                                                  READ_WRITE,
  //                                                  EXCLUSIVE,
  //                                                  inputs[0]->region));
  //launcher.add_field(0, FID_DATA);
  //// gate_assign
  //launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
  //                                                  0 /*projection id*/,
  //                                                  READ_WRITE,
  //                                                  EXCLUSIVE,
  //                                                  inputs[1]->region));
  //launcher.add_field(1, FID_DATA);
  //// exp_preds
  //for (int i = 0; i < n; i++) {
  //  launcher.add_region_requirement(RegionRequirement(inputs[i + 4]->part,
  //                                                    0 /*projection id*/,
  //                                                    READ_WRITE,
  //                                                    EXCLUSIVE,
  //                                                    inputs[i + 4]->region));
  //  launcher.add_field(i + 2, FID_DATA);
  //}
  //// output
  //launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
  //                                                  0 /*projection id*/,
  //                                                  WRITE_ONLY,
  //                                                  EXCLUSIVE,
  //                                                  outputs[0]->region));
  //launcher.add_field(n + 2, FID_DATA);
  //runtime->execute_index_space(ctx, launcher);
}

void AggregateSpec::forward_task(Task const *task,
                                 std::vector<PhysicalRegion> const &regions,
                                 Context ctx,
                                 Runtime *runtime) {
  OpTaskArgumentAccessor acc(task, regions, ctx, runtime);

  int n = acc.get_argument<AggregateSpecAttrs>(ATTRS).n;

  assert((int)regions.size() == n + 3);
  assert((int)task->regions.size() == n + 3);

  AggregateSpecPerDeviceState const *m = *((AggregateSpecPerDeviceState **)task->local_args);

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
  std::vector<float *> exp_preds(n);
  assert (exp_preds.size() == n);
  
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

  profile(
    forward_kernel,
    m->profiling,
    "[AggregateSpec] forward_time = %.2lfms\n",
    m,
    exp_preds.data(),
    acc_gate_assign.ptr(rect_gate_assign),
    acc_output.ptr(rect_output),
    n,
    k,
    rows,
    batch_size,
    out_dim);
}

void AggregateSpec::backward(FFModel const &ff) {
  this->execute_task(ff, AGG_SPEC_BWD_TASK_ID, get_bwd_task_signature());
  
  //ArgumentMap argmap;
  //Context ctx = ff.config.lg_ctx;
  //Runtime *runtime = ff.config.lg_hlr;
  //set_argumentmap_for_backward(ff, argmap);
  //IndexLauncher launcher(AGG_SPEC_BWD_TASK_ID,
  //                       parallel_is,
  //                       TaskArgument(this, sizeof(AggregateSpec)),
  //                       argmap,
  //                       Predicate::TRUE_PRED,
  //                       false /*must*/,
  //                       0 /*mapper_id*/,
  //                       outputs[0]->machine_view.hash());

  //// gate_preds
  //launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
  //                                                  0 /*projection id*/,
  //                                                  READ_WRITE,
  //                                                  EXCLUSIVE,
  //                                                  inputs[0]->region));
  //launcher.add_field(0, FID_DATA);

  //// gate_assign
  //launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
  //                                                  0 /*projection id*/,
  //                                                  READ_ONLY,
  //                                                  EXCLUSIVE,
  //                                                  inputs[1]->region));
  //launcher.add_field(1, FID_DATA);

  //// true gate_assign
  //launcher.add_region_requirement(RegionRequirement(inputs[2]->part,
  //                                                  0 /*projection id*/,
  //                                                  READ_ONLY,
  //                                                  EXCLUSIVE,
  //                                                  inputs[2]->region));
  //launcher.add_field(2, FID_DATA);

  //// gate gradients full
  //launcher.add_region_requirement(RegionRequirement(inputs[3]->part_grad,
  //                                                  0 /*projection id*/,
  //                                                  READ_WRITE,
  //                                                  EXCLUSIVE,
  //                                                  inputs[3]->region_grad));
  //launcher.add_field(3, FID_DATA);

  //// exp gradients
  //for (int i = 0; i < n; i++) {
  //  launcher.add_region_requirement(
  //      RegionRequirement(inputs[i + 4]->part_grad,
  //                        0 /*projection id*/,
  //                        READ_WRITE,
  //                        EXCLUSIVE,
  //                        inputs[i + 4]->region_grad));
  //  launcher.add_field(i + 4, FID_DATA);
  //}

  //// output
  //launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
  //                                                  0 /*projection id*/,
  //                                                  READ_WRITE,
  //                                                  EXCLUSIVE,
  //                                                  outputs[0]->region_grad));
  //launcher.add_field(n + 4, FID_DATA);

  //runtime->execute_index_space(ctx, launcher);
}

void AggregateSpec::backward_task(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime) {
  OpTaskArgumentAccessor acc(task, regions, ctx, runtime);
  auto const &attrs = acc.get_argument<AggregateSpecAttrs>(ATTRS);
  
  AggregateSpecPerDeviceState const *m = *((AggregateSpecPerDeviceState **)task->local_args);

  int n = attrs.n;
  float lambda_bal = attrs.lambda_bal;

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
  std::vector<float *> exp_grads(n);
  assert (exp_grads.size() == n);
  
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

  profile(
    backward_kernel,
    m->profiling,
    "[AggregateSpec] backward_time = %.2lfms\n",
    m,
    exp_grads.data(),
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
  assert(numInputs <= MAX_NUM_INPUTS);
  ParallelTensorBase sub_inputs[MAX_NUM_INPUTS], sub_assign, sub_output;
  for (int i = 0; i < numInputs; ++i) {
    if (!inputs[i + 4]->get_sub_tensor(mv, sub_inputs[i])) {
      return false;
    }
  }
  if (!inputs[1]->get_sub_tensor(mv, sub_assign)) {
    return false;
  }

  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }

  AggregateSpecPerDeviceState *m = new AggregateSpecPerDeviceState(sim->handler, this->attrs.n);

  // allocate
  sim->free_all();
  float *input_ptrs[MAX_NUM_INPUTS];
  bool out_of_memory = false;
  for (int i = 0; i < numInputs; ++i) {
    input_ptrs[i] =
        (float *)sim->allocate(sub_inputs[i].get_volume(), DT_FLOAT);
    out_of_memory = out_of_memory || (input_ptrs[i] == NULL);
  }
  int *assign_ptr = (int *)sim->allocate(sub_assign.get_volume(), DT_INT32);
  out_of_memory = out_of_memory || (assign_ptr == NULL);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
  out_of_memory = out_of_memory || (output_ptr == NULL);

  if (out_of_memory) {
    cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    return true;
  }

  assert(m->profiling == false);

  // compute
  std::function<void(ffStream_t)> forward, backward;
  Domain assign_domain = sub_assign.get_domain();
  Domain exp_domain = sub_inputs[0].get_domain();

  int k = assign_domain.hi()[0] - assign_domain.lo()[0] + 1;
  int batch_size = assign_domain.hi()[1] - assign_domain.lo()[1] + 1;
  int rows = exp_domain.hi()[1] - exp_domain.lo()[1] + 1;
  int out_dim = exp_domain.hi()[0] - exp_domain.lo()[0] + 1;

  forward = [&](ffStream_t stream) {
    forward_kernel(stream, m, input_ptrs, assign_ptr, output_ptr, this->attrs.n, k, rows, batch_size, out_dim);
  };

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);
  log_measure.debug("[Measure Agg Spec] name(%s) forward_time(%.4lf)\n",
                    name,
                    cost_metrics.forward_time);

  cost_metrics.backward_time = 0.0f; // not implemented for backward
  delete m;
  return true;
}

}

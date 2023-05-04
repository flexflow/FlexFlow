/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
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

#include "aggregate.h"
#include "kernels/aggregate_kernels.h"
#include "tasks.h"
#include "kernels/profiling.h"
#include "get_data_dependencies.h"

namespace FlexFlow {

enum Slots {
  GATE_PREDS,
  GATE_ASSIGN,
  TRUE_GATE_ASSIGN,
  FULL_GATE_GRADIENTS,
  EXP_PREDS,
  OUTPUT,
  ATTRS,
  PROFILING,
  PER_DEVICE_STATE,
};

using namespace FlexFlow::Kernels::Aggregate;

DataDependencies get_data_dependencies(AggregateAttrs const &attrs, TaskSignature const &sig) {
  DataDependencies deps;
  return pointwise_data_dependence({GATE_PREDS, GATE_ASSIGN, TRUE_GATE_ASSIGN, FULL_GATE_GRADIENTS, EXP_PREDS},
                                   {},
                                   {OUTPUT});
}

CostMetrics measure_operator_cost(Simulator const &sim,
                                  AggregateAttrs const &attrs,
                                  ParallelTensorShape const &gate_preds_shape,
                                  ParallelTensorShape const &gate_assign_shape,
                                  ParallelTensorShape const &true_gate_assign_shape,
                                  ParallelTensorShape const &full_gate_gradients_shape,
                                  std::vector<ParallelTensorShape> const &exp_preds_shapes,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv) 
{
  auto env = sim.new_environment();

  auto gate_preds = allocate_input(env, gate_preds_shape);
  auto gate_assign = allocate_input(env, gate_assign_shape);
  auto true_gate_assign = allocate_input(env, true_gate_assign_shape);
  auto full_gate_gradients = allocate_input(env, full_gate_gradients_shape);
  auto exp_preds = allocate(env, exp_preds_shapes);
  auto exp_grads = allocate(env, exp_preds_shapes);
  ParallelTensorShape output_shape = get_output_shape(attrs, gate_preds_shape, gate_assign_shape, true_gate_assign_shape, full_gate_gradients_shape, exp_preds_shapes);
  auto output = allocate(env, output_shape);
  auto output_grad = allocate(env, output_shape);

  int k = gate_assign.shape[legion_dim_t(0)];
  int rows = exp_preds[0].shape[legion_dim_t(1)];
  int batch_size = gate_preds.shape[legion_dim_t(1)];
  int out_dim = output.shape[legion_dim_t(1)];

  float forward_time = profiling_wrapper(
    forward_kernel,
    settings,
    get_float_ptrs(exp_preds),
    get_int32_ptr(gate_assign),
    get_float_ptr(gate_preds),
    get_float_ptr(output),
    attrs.n,
    k,
    rows,
    batch_size,
    out_dim
  ).value();

  float backward_time = profiling_wrapper(
    backward_kernel,
    settings,
    get_float_ptrs(exp_preds),
    get_float_ptrs(exp_grads),
    get_int32_ptr(gate_assign),
    get_int32_ptr(true_gate_assign),
    get_float_ptr(gate_preds),
    get_float_ptr(full_gate_gradients),
    get_float_ptr(output_grad),
    attrs.n,
    k,
    rows,
    attrs.lambda_bal,
    batch_size,
    out_dim
  ).value();

  float sync_time = default_estimate_sync_time(env);

  return make_metrics(forward_time, backward_time, sync_time, env);
}


OpTaskInvocation init(AggregateAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, attrs);

  return { AGGREGATE_INIT_TASK_ID, binding };
}

OpTaskInvocation foward(AggregateAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(GATE_PREDS, input_tensor(0));
  binding.bind(GATE_ASSIGN, input_tensor(1));
  
  for (int i = 0; i < attrs.n; i++) {
    binding.bind(EXP_PREDS, input_tensor(i+4));
  }

  binding.bind(OUTPUT, output_tensor(0));

  return { AGGREGATE_FWD_TASK_ID, binding };
}

OpTaskInvocation backward(AggregateAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(GATE_PREDS, input_tensor(0));
  binding.bind(GATE_ASSIGN, input_tensor(1));
  binding.bind(TRUE_GATE_ASSIGN, input_tensor(2));
  binding.bind_grad(FULL_GATE_GRADIENTS, input_tensor(3).grad());
  
  for (int i = 0; i < attrs.n; i++) {
    binding.bind(EXP_PREDS, input_tensor(i+4));
    binding.bind_grad(EXP_PREDS, input_tensor(i+4).grad());
  }

  binding.bind_grad(OUTPUT, output_tensor(0).grad());

  return { AGGREGATE_BWD_TASK_ID, binding };
}

static PerDeviceOpState *init_task(Legion::Task const *task,
                                   std::vector<Legion::PhysicalRegion> const &regions,
                                   Legion::Context ctx,
                                   Legion::Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);

  AggregateAttrs const &attrs = acc.get_argument<AggregateAttrs>(ATTRS);
  bool profiling = acc.get_argument<bool>(PROFILING);

  FFHandler handle = *((FFHandler *)task->local_args);
  AggregatePerDeviceState *m = new AggregatePerDeviceState(handle, attrs.n);

  m->profiling = profiling;
  return m;
}

static void forward_task(Legion::Task const *task,
                         std::vector<Legion::PhysicalRegion> const &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);

  AggregateAttrs const &attrs = acc.get_argument<AggregateAttrs>(ATTRS);
  int n = attrs.n;

  assert((int)regions.size() == n + 3);
  assert((int)task->regions.size() == n + 3);

  AggregatePerDeviceState const *m = *((AggregatePerDeviceState **)task->local_args);

  // get gate_pred, gate_assign, output
  auto gate_pred = acc.get_tensor<READ_WRITE>(GATE_PREDS);
  auto gate_assign = acc.get_tensor<READ_WRITE>(GATE_ASSIGN);
  auto output = acc.get_tensor<WRITE_ONLY>(OUTPUT);

  coord_t batch_size = gate_pred.shape[1];
  assert(batch_size == gate_assign.shape[1]);
  assert(gate_pred.shape[0] == gate_assign.shape[0]);
  assert(batch_size == output.shape[1]);
  coord_t out_dim = output.shape[0];

  // get exp_preds
  auto acc_exp_preds = acc.get_variadic_tensor<READ_WRITE>(EXP_PREDS);
  coord_t rows = acc_exp_preds[0].shape[1];
  assert (all_of(acc_exp_preds, [&](GenericTensorAccessorW const &a) { return a.shape[1] == rows; }));
  assert (all_of(acc_exp_preds, [&](GenericTensorAccessorW const &a) { return a.shape[0] == out_dim; }));
  
  std::vector<float *> exp_preds = vector_transform([](GenericTensorAccessorW const &a) { return a.get_float_ptr(); }, acc_exp_preds);
  assert (exp_preds.size() == n);

  int k = (int)(gate_assign.shape[0]);

  profile(
    forward_kernel, 
    m->profiling, 
    "[Aggregate] forward_time = %.2lfms\n",
    m,
    exp_preds.data(),
    gate_assign.get_float_ptr(),
    gate_pred.get_float_ptr(),
    output.get_float_ptr(),
    n,
    k,
    rows,
    batch_size,
    out_dim
  );
}

static void backward_task(Legion::Task const *task,
                          std::vector<Legion::PhysicalRegion> const &regions,
                          Legion::Context ctx,
                          Legion::Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);

  auto attrs = acc.get_argument<AggregateAttrs>(ATTRS);
  auto per_device_state = acc.get_argument<AggregatePerDeviceState>(PER_DEVICE_STATE);

  int n = attrs.n;
  float lambda_bal = attrs.lambda_bal;

  assert((int)regions.size() == 2 * n + 5);
  assert((int)task->regions.size() == 2 * n + 5);

  // get gate_pred, gate_grad, gate_assign, output_grad
  auto gate_pred = acc.get_tensor<READ_ONLY>(GATE_PREDS);
  auto gate_assign = acc.get_tensor<READ_ONLY>(GATE_ASSIGN);
  auto true_gate_assign = acc.get_tensor<READ_ONLY>(TRUE_GATE_ASSIGN);
  auto full_gate_grad = acc.get_tensor_grad<READ_WRITE>(GATE_GRADIENTS_FULL);
  auto output_grad = acc.get_tensor_grad<READ_ONLY>(OUTPUT);

  coord_t batch_size = gate_pred.shape[1];
  assert(batch_size == gate_assign.shape[1]);
  assert(gate_assign.shape == true_gate_assign.shape);
  assert(batch_size == full_gate_grad.shape[1]);
  coord_t k = gate_assign.shape[0];
  assert(k * batch_size == output_grad.shape[1]);
  assert(gate_pred.shape[0] == k);
  coord_t out_dim = output_grad.shape[0];
  assert(n == full_gate_grad.shape[0]);

  // get exp_preds
  auto acc_exp_preds = acc.get_variadic_tensor<READ_WRITE>(EXP_PREDS);
  coord_t rows = acc_exp_preds[0].shape[1];
  assert (all_of(acc_exp_preds, [&](GenericTensorAccessorW const &a) { return a.shape[1] == rows; }));
  assert (all_of(acc_exp_preds, [&](GenericTensorAccessorW const &a) { return a.shape[0] == out_dim; }));
  
  std::vector<float *> exp_preds = vector_transform([](GenericTensorAccessorW const &a) { return a.get_float_ptr(); }, acc_exp_preds);
  assert (exp_preds.size() == n);

  // get chosen_exp_grads
  auto acc_exp_grads = acc.get_variadic_tensor_grad<READ_WRITE>(EXP_PREDS);
  
  size_t rows = acc_exp_grads[0].shape[1];
  assert (all_of(acc_exp_grads, [&](GenericTensorAccessorW const &a) { return a.shape[1] == rows; }));
  assert (all_of(acc_exp_grads, [&](GenericTensorAccessorW const &a) { return a.shape[0] == out_dim; }));

  std::vector<float *> exp_grads = vector_transform([](GenericTensorAccessorW const &a) { return a.get_float_ptr(); }, acc_exp_grads);
  assert (exp_grads.size() == n);

  profile(
    backward_kernel,
    per_device_state.profiling,
    "[Aggregate] backward_time = %.2lfms\n",
    m,
    exp_preds.data(), 
    exp_grads.data(),
    gate_assign.get_float_ptr(),
    true_gate_assign.get_float_ptr(),
    gate_pred.get_float_ptr(),
    full_gate_grad.get_float_ptr(),
    output_grad.get_float_ptr(),
    n,
    k,
    rows,
    lambda_bal,
    batch_size,
    out_dim
  );
}

  // ArgumentMap argmap;
  // Context ctx = ff.config.lg_ctx;
  // Runtime *runtime = ff.config.lg_hlr;
  // set_argumentmap_for_backward(ff, argmap);
  // IndexLauncher launcher(AGGREGATE_BWD_TASK_ID,
  //                        parallel_is,
  //                        TaskArgument(this, sizeof(Aggregate)),
  //                        argmap,
  //                        Predicate::TRUE_PRED,
  //                        false /*must*/,
  //                        0 /*mapper_id*/,
  //                        get_std_hash(outputs[0]->machine_view));
  // // gate_preds
  // launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
  //                                                   0 /*projection id*/,
  //                                                   READ_WRITE,
  //                                                   EXCLUSIVE,
  //                                                   inputs[0]->region));
  // launcher.add_field(0, FID_DATA);
  // // gate_assign
  // launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
  //                                                   0 /*projection id*/,
  //                                                   READ_ONLY,
  //                                                   EXCLUSIVE,
  //                                                   inputs[1]->region));
  // launcher.add_field(1, FID_DATA);
  // // true gate_assign
  // launcher.add_region_requirement(RegionRequirement(inputs[2]->part,
  //                                                   0 /*projection id*/,
  //                                                   READ_ONLY,
  //                                                   EXCLUSIVE,
  //                                                   inputs[2]->region));
  // launcher.add_field(2, FID_DATA);
  // // full_gate gradients
  // launcher.add_region_requirement(RegionRequirement(inputs[3]->part_grad,
  //                                                   0 /*projection id*/,
  //                                                   READ_WRITE,
  //                                                   EXCLUSIVE,
  //                                                   inputs[3]->region_grad));
  // launcher.add_field(3, FID_DATA);
  // // exp_preds
  // for (int i = 0; i < n; i++) {
  //   launcher.add_region_requirement(RegionRequirement(inputs[i + 4]->part,
  //                                                     0 /*projection id*/,
  //                                                     READ_WRITE,
  //                                                     EXCLUSIVE,
  //                                                     inputs[i + 4]->region));
  //   launcher.add_field(i + 4, FID_DATA);
  // }
  // // exp_preds gradients
  // for (int i = 0; i < n; i++) {
  //   launcher.add_region_requirement(
  //       RegionRequirement(inputs[i + 4]->part_grad,
  //                         0 /*projection id*/,
  //                         READ_WRITE,
  //                         EXCLUSIVE,
  //                         inputs[i + 4]->region_grad));
  //   launcher.add_field(i + n + 4, FID_DATA);
  // }

  // // output
  // launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
  //                                                   0 /*projection id*/,
  //                                                   READ_WRITE,
  //                                                   EXCLUSIVE,
  //                                                   outputs[0]->region_grad));
  // launcher.add_field(2 * n + 4, FID_DATA);

  // runtime->execute_index_space(ctx, launcher);
}



/* void Aggregate::serialize(Legion::Serializer &sez) const { */
/*   sez.serialize(this->n); */
/*   sez.serialize(this->lambda_bal); */
/* } */

// bool Aggregate::measure_operator_cost(Simulator *sim,
//                                       MachineView const &mv,
//                                       CostMetrics &cost_metrics) const {
//   assert(numInputs <= MAX_NUM_INPUTS);
//   ParallelTensorBase sub_inputs[MAX_NUM_INPUTS], sub_pred, sub_assign,
//       sub_output;
// 
//   for (int i = 0; i < numInputs; ++i) {
//     if (!inputs[i + 4]->get_sub_tensor(mv, sub_inputs[i])) {
//       return false;
//     }
//   }
//   if (!inputs[0]->get_sub_tensor(mv, sub_pred)) {
//     return false;
//   }
//   if (!inputs[1]->get_sub_tensor(mv, sub_assign)) {
//     return false;
//   }
// 
//   if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
//     return false;
//   }
// 
//   AggregatePerDeviceState *m = new AggregatePerDeviceState(sim->handler, attrs.n);
// 
//   // allocate
//   sim->free_all();
//   float *input_ptrs[MAX_NUM_INPUTS];
//   bool out_of_memory = false;
//   for (int i = 0; i < numInputs; ++i) {
//     input_ptrs[i] =
//         (float *)sim->allocate(sub_inputs[i].get_volume(), DT_FLOAT);
//     out_of_memory = out_of_memory || (input_ptrs[i] == NULL);
//   }
//   int *assign_ptr = (int *)sim->allocate(sub_assign.get_volume(), DT_INT32);
//   out_of_memory = out_of_memory || (assign_ptr == NULL);
//   float *pred_ptr = (float *)sim->allocate(sub_pred.get_volume(), DT_FLOAT);
//   out_of_memory = out_of_memory || (pred_ptr == NULL);
//   cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
// 
//   float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
//   cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
//   out_of_memory = out_of_memory || (output_ptr == NULL);
// 
//   if (out_of_memory) {
//     cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
//     cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
//     return true;
//   }
// 
//   assert(m->profiling == false);
// 
//   // compute
//   std::function<void(ffStream_t)> forward, backward;
//   Domain assign_domain = sub_assign.get_domain();
//   Domain exp_domain = sub_inputs[0].get_domain();
// 
//   int k = assign_domain.hi()[0] - assign_domain.lo()[0] + 1;
//   int batch_size = assign_domain.hi()[1] - assign_domain.lo()[1] + 1;
//   int rows = exp_domain.hi()[1] - exp_domain.lo()[1] + 1;
//   int out_dim = exp_domain.hi()[0] - exp_domain.lo()[0] + 1;
// 
//   forward = [&](ffStream_t stream) {
//     forward_kernel(stream, m,
//                            input_ptrs,
//                            assign_ptr,
//                            pred_ptr,
//                            output_ptr,
//                            attrs.n,
//                            k,
//                            rows,
//                            batch_size,
//                            out_dim);
//   };
// 
//   inner_measure_operator_cost(sim, forward, backward, cost_metrics);
//   log_measure.debug("[Measure Aggregate] name(%s) forward_time(%.4lf)\n",
//                     name,
//                     cost_metrics.forward_time);
// 
//   cost_metrics.backward_time = 0.0f; // not implemented for backward
//   delete m;
//   return true;
// }

template <>
void register_task<AGGREGATE_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_arg_slot<AggregateAttrs>(ATTRS);
  init.add_arg_slot<bool>(PROFILING);

  register_task(AGGREGATE_INIT_TASK_ID, "Aggregate Init", init, aggregate_init_task);
}

template <>
void register_task<AGGREGATE_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(GATE_PREDS);
  fwd.add_input_slot(GATE_ASSIGN);
  fwd.add_input_slot(EXP_PREDS, SlotType::VARIADIC);
  fwd.add_output_slot(OUTPUT);

  register_task(AGGREGATE_FWD_TASK_ID, "Aggregate Fwd", fwd, aggregate_forward_task);
}

template <>
void register_task<AGGREGATE_BWD_TASK_ID>() {
  OpTaskSignature bwd(OpTaskType::BWD);

  bwd.add_input_slot(GATE_PREDS);
  bwd.add_input_slot(GATE_ASSIGN);
  bwd.add_input_slot(TRUE_GATE_ASSIGN);
  bwd.add_input_slot(FULL_GATE_GRADIENTS);
  bwd.add_input_slot(EXP_PREDS, SlotType::VARIADIC);
  bwd.add_output_slot(OUTPUT);

  bwd.add_arg_slot<AggregateAttrs>(ATTRS);

  register_task(AGGREGATE_BWD_TASK_ID, "Aggregate Bwd", bwd, aggregate_backward_task);
}

}

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
#include "get_data_dependencies.h"
#include "kernels/accessor.h"
#include "kernels/aggregate_kernels.h"
#include "kernels/profiling.h"
#include "task_spec/task_argument_accessor.h"
#include "tasks.h"

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
};

using namespace FlexFlow::Kernels::Aggregate;

/* DataDependencies get_data_dependencies(AggregateAttrs const &attrs, */
/*                                        TaskSignature const &sig) { */
/*   DataDependencies deps; */
/*   return pointwise_data_dependence({GATE_PREDS, */
/*                                     GATE_ASSIGN, */
/*                                     TRUE_GATE_ASSIGN, */
/*                                     FULL_GATE_GRADIENTS, */
/*                                     EXP_PREDS}, */
/*                                    {}, */
/*                                    {OUTPUT}); */
/* } */

// OpTaskInvocation init(AggregateAttrs const &attrs) {
//   OpTaskBinding binding;

//   binding.bind_arg(ATTRS, attrs);

//   return { AGGREGATE_INIT_TASK_ID, binding };
// }

OpTaskInvocation forward(AggregateAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(GATE_PREDS, input_tensor(0));
  binding.bind(GATE_ASSIGN, input_tensor(1));

  for (int i = 0; i < attrs.n; i++) {
    binding.bind(EXP_PREDS, input_tensor(i + 4));
  }

  binding.bind(OUTPUT, output_tensor(0));

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(ATTRS, attrs);

  return {AGGREGATE_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(AggregateAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(GATE_PREDS, input_tensor(0));
  binding.bind(GATE_ASSIGN, input_tensor(1));
  binding.bind(TRUE_GATE_ASSIGN, input_tensor(2));
  binding.bind_grad(FULL_GATE_GRADIENTS, input_tensor(3));

  for (int i = 0; i < attrs.n; i++) {
    binding.bind(EXP_PREDS, input_tensor(i + 4));
    binding.bind_grad(EXP_PREDS, input_tensor(i + 4));
  }

  binding.bind_grad(OUTPUT, output_tensor(0));

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(ATTRS, attrs);

  return {AGGREGATE_BWD_TASK_ID, binding};
}

/* static AggregatePerDeviceState init_task(TaskArgumentAccessor const &acc) {
 */
/*   AggregateAttrs const &attrs = acc.get_argument<AggregateAttrs>(ATTRS); */
/*   bool profiling = acc.get_argument<bool>(PROFILING); */

/*   FFHandler handle = *((FFHandler *)task->local_args); */
/*   AggregatePerDeviceState *m = new AggregatePerDeviceState(handle, attrs.n);
 */

/*   return m; */
/* } */

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<AggregateAttrs>(ATTRS);
  auto profiling_settings = acc.get_argument<ProfilingSettings>(PROFILING);

  int n = attrs.n;

  // get gate_pred, gate_assign, output
  auto gate_pred = acc.get_tensor<Permissions::RW>(GATE_PREDS);
  auto gate_assign = acc.get_tensor<Permissions::RW>(GATE_ASSIGN);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  size_t batch_size = gate_pred.shape[legion_dim_t(1)];
  assert(batch_size == gate_assign.shape[legion_dim_t(1)]);
  assert(gate_pred.shape[legion_dim_t(0)] ==
         gate_assign.shape[legion_dim_t(0)]);
  assert(batch_size == output.shape[legion_dim_t(1)]);
  size_t out_dim = output.shape[legion_dim_t(0)];

  // get exp_preds
  auto acc_exp_preds = acc.get_variadic_tensor<Permissions::RW>(EXP_PREDS);
  size_t rows = acc_exp_preds[0].shape[legion_dim_t(1)];
  assert(all_of(acc_exp_preds, [&](GenericTensorAccessorW const &a) {
    return a.shape[legion_dim_t(1)] == rows;
  }));
  assert(all_of(acc_exp_preds, [&](GenericTensorAccessorW const &a) {
    return a.shape[legion_dim_t(0)] == out_dim;
  }));

  std::vector<float *> exp_preds = vector_transform(
      [](GenericTensorAccessorW const &a) { return a.get_float_ptr(); },
      acc_exp_preds);
  assert(exp_preds.size() == n);

  int k = (int)(gate_assign.shape[legion_dim_t(0)]);

  return profile(forward_kernel,
                 profiling_settings,
                 "[Aggregate] forward_time = %.2lfms\n",
                 exp_preds.data(),
                 get_int32_ptr(gate_assign),
                 get_float_ptr(gate_pred),
                 get_float_ptr(output),
                 n,
                 k,
                 rows,
                 batch_size,
                 out_dim);
}

static void forward_task(Legion::Task const *task,
                         std::vector<Legion::PhysicalRegion> const &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<AggregateAttrs>(ATTRS);
  auto profiling_settings = acc.get_argument<ProfilingSettings>(PROFILING);

  int n = attrs.n;
  float lambda_bal = attrs.lambda_bal;

  // get gate_pred, gate_grad, gate_assign, output_grad
  auto gate_pred = acc.get_tensor<Permissions::RO>(GATE_PREDS);
  auto gate_assign = acc.get_tensor<Permissions::RO>(GATE_ASSIGN);
  auto true_gate_assign = acc.get_tensor<Permissions::RO>(TRUE_GATE_ASSIGN);
  auto full_gate_grad =
      acc.get_tensor_grad<Permissions::RW>(FULL_GATE_GRADIENTS);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  size_t batch_size = gate_pred.shape[legion_dim_t(1)];
  assert(batch_size == gate_assign.shape[legion_dim_t(1)]);
  assert(gate_assign.shape == true_gate_assign.shape);
  assert(batch_size == full_gate_grad.shape[legion_dim_t(1)]);
  size_t k = gate_assign.shape[legion_dim_t(0)];
  assert(k * batch_size == output_grad.shape[legion_dim_t(1)]);
  assert(gate_pred.shape[legion_dim_t(0)] == k);
  size_t out_dim = output_grad.shape[legion_dim_t(0)];
  assert(n == full_gate_grad.shape[legion_dim_t(0)]);

  // get exp_preds
  auto acc_exp_preds = acc.get_variadic_tensor<Permissions::RW>(EXP_PREDS);
  size_t rows = acc_exp_preds[0].shape[legion_dim_t(1)];
  assert(all_of(acc_exp_preds, [&](GenericTensorAccessorW const &a) {
    return a.shape[legion_dim_t(1)] == rows;
  }));
  assert(all_of(acc_exp_preds, [&](GenericTensorAccessorW const &a) {
    return a.shape[legion_dim_t(0)] == out_dim;
  }));

  std::vector<float *> exp_preds = vector_transform(
      [](GenericTensorAccessorW const &a) { return a.get_float_ptr(); },
      acc_exp_preds);
  assert(exp_preds.size() == n);

  // get chosen_exp_grads
  auto acc_exp_grads = acc.get_variadic_tensor_grad<Permissions::RW>(EXP_PREDS);

  assert(all_of(acc_exp_grads, [&](GenericTensorAccessorW const &a) {
    return a.shape[legion_dim_t(1)] == rows;
  }));
  assert(all_of(acc_exp_grads, [&](GenericTensorAccessorW const &a) {
    return a.shape[legion_dim_t(0)] == out_dim;
  }));

  std::vector<float *> exp_grads = vector_transform(
      [](GenericTensorAccessorW const &a) { return a.get_float_ptr(); },
      acc_exp_grads);
  assert(exp_grads.size() == n);

  return profile(backward_kernel,
                 profiling_settings,
                 "[Aggregate] backward_time = %.2lfms\n",
                 exp_preds.data(),
                 exp_grads.data(),
                 get_int32_ptr(gate_assign),
                 get_int32_ptr(true_gate_assign),
                 gate_pred.get_float_ptr(),
                 full_gate_grad.get_float_ptr(),
                 output_grad.get_float_ptr(),
                 n,
                 k,
                 rows,
                 lambda_bal,
                 batch_size,
                 out_dim);
}

static void backward_task(Legion::Task const *task,
                          std::vector<Legion::PhysicalRegion> const &regions,
                          Legion::Context ctx,
                          Legion::Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
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
// }

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
//   AggregatePerDeviceState *m = new AggregatePerDeviceState(sim->handler,
//   attrs.n);
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
//   cost_metrics.inputs_memory +=
//   cost_metrics.total_mem_diff_from(sim->offset);
//
//   float *output_ptr = (float *)sim->allocate(sub_output.get_volume(),
//   DT_FLOAT); cost_metrics.outputs_memory +=
//   cost_metrics.total_mem_diff_from(sim->offset); out_of_memory =
//   out_of_memory || (output_ptr == NULL);
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
void register_task<AGGREGATE_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_untrainable_input_slot(GATE_PREDS);
  fwd.add_untrainable_input_slot(GATE_ASSIGN);
  fwd.add_input_slot(EXP_PREDS, SlotType::VARIADIC);
  fwd.add_output_slot(OUTPUT);

  fwd.add_arg_slot<AggregateAttrs>(ATTRS);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);

  register_task(AGGREGATE_FWD_TASK_ID, "Aggregate Fwd", fwd, forward_task);
}

template <>
void register_task<AGGREGATE_BWD_TASK_ID>() {
  OpTaskSignature bwd(OpTaskType::BWD);

  /* OpTaskSignature bwd = */
  /*     infer_bwd_signature(get_signature(AGGREGATE_FWD_TASK_ID)); */
  bwd.add_input_slot(TRUE_GATE_ASSIGN);
  bwd.add_input_slot(FULL_GATE_GRADIENTS);

  bwd.add_arg_slot<AggregateAttrs>(ATTRS);
  bwd.add_arg_slot<ProfilingSettings>(PROFILING);

  register_task(AGGREGATE_BWD_TASK_ID, "Aggregate Bwd", bwd, backward_task);
}

CostMetrics
    measure_operator_cost(SimEnvFactory const &sim,
                          AggregateAttrs const &attrs,
                          InputParallelTensorDesc const &gate_preds,
                          InputParallelTensorDesc const &gate_assign,
                          InputParallelTensorDesc const &true_gate_assign,
                          InputParallelTensorDesc const &full_gate_gradients,
                          InputVariadicParallelTensorDesc const &exp_preds,
                          ProfilingSettings const &settings,
                          MachineView const &mv) {
  auto env = sim.new_environment();

  SimTaskBinding fwd_binding;
  fwd_binding.bind(GATE_PREDS, gate_preds);
  fwd_binding.bind(GATE_ASSIGN, gate_assign);
  fwd_binding.bind(EXP_PREDS, exp_preds);

  ParallelTensorShape output_shape = get_output_shape(attrs,
                                                      gate_preds.shape,
                                                      gate_assign.shape,
                                                      true_gate_assign.shape,
                                                      full_gate_gradients.shape,
                                                      exp_preds.shapes);
  fwd_binding.bind(OUTPUT, output_shape);

  fwd_binding.bind_arg(PROFILING, settings);

  auto fwd_accessor = env.get_fwd_accessor(AGGREGATE_FWD_TASK_ID, fwd_binding);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);
  bwd_binding.bind(FULL_GATE_GRADIENTS, full_gate_gradients);
  bwd_binding.bind(TRUE_GATE_ASSIGN, true_gate_assign);

  auto bwd_accessor = env.get_bwd_accessor(AGGREGATE_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);

  return make_metrics(forward_time, backward_time, sync_time, env);
}

} // namespace FlexFlow

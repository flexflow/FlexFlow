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
#include "task_spec/device_specific.h"

namespace FlexFlow {

enum Slots {
  GATE_PREDS,
  GATE_ASSIGN,
  EXP_PREDS,
  OUTPUT,
  TRUE_GATE_ASSIGN,
  FULL_GATE_GRADIENTS,
  ATTRS,
  PER_DEVICE_STATE,
  PROFILING
};

using namespace FlexFlow::Kernels::AggregateSpec;

// Tensor FFModel::aggregate_spec(
//     Tensor const *inputs, /* gate_preds, gate_assign, gate assign TopK,
//                              full_gate_pred, exp_pred_1, ... , exp_pred_n */
//     int n,
//     float lambda_bal,
//     char const *name) {
//   Layer *li = new Layer(this,
//                         OP_AGG_SPEC,
//                         DT_FLOAT,
//                         name,
//                         n + 4 /*inputs*/,
//                         0 /*weights*/,
//                         1 /*outputs*/,
//                         inputs);
//   {
//     int num_dim = inputs[4]->num_dims;
//     // Set output shape
//     int dims[MAX_TENSOR_DIM];
//     for (int i = 0; i < num_dim - 1; i++) {
//       dims[i] = inputs[4]->dims[i];
//     }
//     dims[num_dim - 1] = inputs[0]->dims[num_dim - 1];
//     li->outputs[0] = create_tensor_legion_ordering(
//         num_dim, dims, DT_FLOAT, li, 0, true /*create_grad*/);
//   }
//   li->add_int_property("n", n);
//   li->add_float_property("lambda_bal", lambda_bal);
//   layers.push_back(li);
//   return li->outputs[0];
// }
//
// Op *AggregateSpec::create_operator_from_layer(
//     FFModel &model,
//     Layer const *layer,
//     std::vector<ParallelTensor> const &inputs) {
//   long long value1;
//   layer->get_int_property("n", value1);
//   int n = value1;
//   float value2;
//   layer->get_float_property("lambda_bal", value2);
//   float lambda_bal = value2;
//   return new AggregateSpec(model, inputs.data(), n, lambda_bal, layer->name);
// }
//
// AggregateSpec::AggregateSpec(FFModel &model,
//                              ParallelTensor const *_inputs,
//                              int _n,
//                              float _lambda_bal,
//                              char onst *name)
//     : Op(model,
//          OP_AGG_SPEC,
//          DT_FLOAT,
//          name,
//          _n + 4 /*numInputs*/,
//          0 /*numWeights*/,
//          1 /*numOutputs*/,
//          _inputs),
//       attrs(_n, _lambda_bal) {
//   // FIXME: For now, set upper limits Better: Do as follows, but memory is
//   // assigned per block, so requires to check that
//   //
//   https://stackoverflow.com/questions/5531247/allocating-shared-memory/5531640#5531640
//   assert(attrs.n <= AGGREGATE_SPEC_MAX_N &&
//          "Increase AGGREGATE_SPEC_MAX_N in #define");
//   assert(inputs[0]->dims[0].size <= AGGREGATE_SPEC_MAX_K &&
//          "Increase AGGREGATE_SPEC_MAX_K in #define");
//   assert(inputs[0]->dims[1].size <= AGGREGATE_SPEC_MAX_BATCH_SIZE &&
//          "Increase AGGREGATE_SPEC_MAX_BATCH_SIZE in #define");
//
//   assert(attrs.n + 4 == numInputs);
//   assert(attrs.n > 0);
//   assert(inputs[0]->num_dims == 2);
//   assert(inputs[1]->num_dims == 2);
//   assert(inputs[2]->num_dims == 2);
//   assert(inputs[3]->num_dims == 2);
//
//   for (int i = 0; i < inputs[0]->num_dims; i++) {
//     assert(inputs[0]->dims[i] == inputs[1]->dims[i]);
//     assert(inputs[0]->dims[i] == inputs[2]->dims[i]);
//   }
//   assert(inputs[0]->dims[1] == inputs[3]->dims[1]);
//   assert(inputs[3]->dims[0].size == attrs.n);
//
//   // expert inputs
//   int num_dim = inputs[4]->num_dims;
//   int out_dim = inputs[4]->dims[0].size;
//   for (int i = 1; i < attrs.n; i++) {
//     assert(inputs[i + 4]->num_dims == num_dim);
//     assert(inputs[i + 4]->dims[0].size == out_dim);
//   }
//   // Set output shape
//   ParallelDim dims[MAX_TENSOR_DIM];
//   for (int i = 0; i < num_dim - 1; i++) {
//     dims[i] = inputs[4]->dims[i];
//   }
//   dims[num_dim - 1] = inputs[0]->dims[num_dim - 1];
//   numOutputs = 1;
//   outputs[0] = model.create_parallel_tensor_legion_ordering(
//       num_dim, dims, DT_FLOAT, this);
//
//   numWeights = 0;
// }

// OpTaskInvocation init(AggregateSpecAttrs const &attrs) {
//   OpTaskBinding binding;

//   binding.bind_arg(ATTRS, attrs);
//   binding.bind_arg(FF_HANDLE, ff_handle());

//   return { AGG_SPEC_INIT_TASK_ID, binding };
// }

OpTaskInvocation forward(AggregateSpecAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(GATE_PREDS, input_tensor(0));
  binding.bind(GATE_ASSIGN, input_tensor(1));

  for (int i = 0; i < attrs.n; i++) {
    binding.bind(EXP_PREDS, input_tensor(i + 4));
  }

  binding.bind(OUTPUT, output_tensor(0));

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(ATTRS, attrs);
  binding.bind_device_specific_arg(
      PER_DEVICE_STATE, per_device_op_state<AggregateSpecPerDeviceState>());

  return {AGG_SPEC_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(AggregateSpecAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(GATE_PREDS, input_tensor(0));
  binding.bind(GATE_ASSIGN, input_tensor(1));
  binding.bind(TRUE_GATE_ASSIGN, input_tensor(2));
  binding.bind_grad(FULL_GATE_GRADIENTS, input_tensor(3));

  for (int i = 0; i < attrs.n; i++) {
    binding.bind_grad(EXP_PREDS, input_tensor(i + 4));
  }

  binding.bind_grad(OUTPUT, output_tensor(0));

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(ATTRS, attrs);
  binding.bind_device_specific_arg(
      PER_DEVICE_STATE, per_device_op_state<AggregateSpecPerDeviceState>());

  return {AGG_SPEC_BWD_TASK_ID, binding};
}

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

// static PerDeviceOpState *init_task(Legion::Task const *task,
//                                    std::vector<Legion::PhysicalRegion> const
//                                    &regions, Legion::Context ctx,
//                                    Legion::Runtime *runtime) {
//   TaskArgumentAccessor acc(task, regions, ctx, runtime);
//   auto const &attrs = acc.get_argument<AggregateSpecAttrs>(ATTRS);
//   auto handle = acc.get_argument<PerDeviceFFHandle>(FF_HANDLE);

//   AggregateSpecPerDeviceState *m = new AggregateSpecPerDeviceState(handle,
//   attrs.n); return m;
// }

// ArgumentMap argmap;
// Context ctx = ff.config.lg_ctx;
// Runtime *runtime = ff.config.lg_hlr;
// set_argumentmap_for_forward(ff, argmap);
// IndexLauncher launcher(AGG_SPEC_FWD_TASK_ID,
//                        parallel_is,
//                        TaskArgument(this, sizeof(AggregateSpec)),
//                        argmap,
//                        Predicate::TRUE_PRED,
//                        false /*must*/,
//                        0 /*mapper_id*/,
//                        outputs[0]->machine_view.hash());
//// gate_preds
// launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
//                                                   0 /*projection id*/,
//                                                   READ_WRITE,
//                                                   EXCLUSIVE,
//                                                   inputs[0]->region));
// launcher.add_field(0, FID_DATA);
//// gate_assign
// launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
//                                                   0 /*projection id*/,
//                                                   READ_WRITE,
//                                                   EXCLUSIVE,
//                                                   inputs[1]->region));
// launcher.add_field(1, FID_DATA);
//// exp_preds
// for (int i = 0; i < n; i++) {
//   launcher.add_region_requirement(RegionRequirement(inputs[i + 4]->part,
//                                                     0 /*projection id*/,
//                                                     READ_WRITE,
//                                                     EXCLUSIVE,
//                                                     inputs[i + 4]->region));
//   launcher.add_field(i + 2, FID_DATA);
// }
//// output
// launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
//                                                   0 /*projection id*/,
//                                                   WRITE_ONLY,
//                                                   EXCLUSIVE,
//                                                   outputs[0]->region));
// launcher.add_field(n + 2, FID_DATA);
// runtime->execute_index_space(ctx, launcher);

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<AggregateSpecAttrs>(ATTRS);
  auto per_device_state =
      acc.get_argument<AggregateSpecPerDeviceState>(PER_DEVICE_STATE);
  auto profiling_settings = acc.get_argument<ProfilingSettings>(PROFILING);

  int n = attrs.n;

  auto gate_pred = acc.get_tensor<Permissions::RW>(GATE_PREDS);
  auto gate_assign = acc.get_tensor<Permissions::RW>(GATE_ASSIGN);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  size_t batch_size = gate_pred.shape[legion_dim_t(1)];
  assert(batch_size == gate_assign.shape[legion_dim_t(1)]);
  size_t k = gate_pred.shape[legion_dim_t(0)];
  assert(k == gate_assign.shape[legion_dim_t(0)]);
  assert(k * batch_size == output.shape[legion_dim_t(1)]);
  size_t out_dim = output.shape[legion_dim_t(0)];

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

  return profile(forward_kernel,
                 profiling_settings,
                 "[AggregateSpec] forward_time = %.2lfms\n",
                 &per_device_state,
                 exp_preds.data(),
                 gate_assign.get_int32_ptr(),
                 output.get_float_ptr(),
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

// ArgumentMap argmap;
// Context ctx = ff.config.lg_ctx;
// Runtime *runtime = ff.config.lg_hlr;
// set_argumentmap_for_backward(ff, argmap);
// IndexLauncher launcher(AGG_SPEC_BWD_TASK_ID,
//                        parallel_is,
//                        TaskArgument(this, sizeof(AggregateSpec)),
//                        argmap,
//                        Predicate::TRUE_PRED,
//                        false /*must*/,
//                        0 /*mapper_id*/,
//                        outputs[0]->machine_view.hash());

//// gate_preds
// launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
//                                                   0 /*projection id*/,
//                                                   READ_WRITE,
//                                                   EXCLUSIVE,
//                                                   inputs[0]->region));
// launcher.add_field(0, FID_DATA);

//// gate_assign
// launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
//                                                   0 /*projection id*/,
//                                                   READ_ONLY,
//                                                   EXCLUSIVE,
//                                                   inputs[1]->region));
// launcher.add_field(1, FID_DATA);

//// true gate_assign
// launcher.add_region_requirement(RegionRequirement(inputs[2]->part,
//                                                   0 /*projection id*/,
//                                                   READ_ONLY,
//                                                   EXCLUSIVE,
//                                                   inputs[2]->region));
// launcher.add_field(2, FID_DATA);

//// gate gradients full
// launcher.add_region_requirement(RegionRequirement(inputs[3]->part_grad,
//                                                   0 /*projection id*/,
//                                                   READ_WRITE,
//                                                   EXCLUSIVE,
//                                                   inputs[3]->region_grad));
// launcher.add_field(3, FID_DATA);

//// exp gradients
// for (int i = 0; i < n; i++) {
//   launcher.add_region_requirement(
//       RegionRequirement(inputs[i + 4]->part_grad,
//                         0 /*projection id*/,
//                         READ_WRITE,
//                         EXCLUSIVE,
//                         inputs[i + 4]->region_grad));
//   launcher.add_field(i + 4, FID_DATA);
// }

//// output
// launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
//                                                   0 /*projection id*/,
//                                                   READ_WRITE,
//                                                   EXCLUSIVE,
//                                                   outputs[0]->region_grad));
// launcher.add_field(n + 4, FID_DATA);

// runtime->execute_index_space(ctx, launcher);

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<AggregateSpecAttrs>(ATTRS);
  auto per_device_state =
      acc.get_argument<AggregateSpecPerDeviceState>(PER_DEVICE_STATE);
  auto profiling_settings = acc.get_argument<ProfilingSettings>(PROFILING);

  int n = attrs.n;
  float lambda_bal = attrs.lambda_bal;

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

  auto acc_exp_grads = acc.get_variadic_tensor_grad<Permissions::RW>(EXP_PREDS);

  size_t rows = acc_exp_grads[0].shape[legion_dim_t(1)];
  assert(all_of(acc_exp_grads, [&](GenericTensorAccessorW const &a) {
    return a.shape[legion_dim_t(1)] == rows;
  }));
  assert(all_of(acc_exp_grads, [&](GenericTensorAccessorW const &a) {
    return a.shape[legion_dim_t(0)] == out_dim;
  }));

  assert(acc_exp_grads.size() == n);

  return profile(backward_kernel,
                 profiling_settings,
                 "[AggregateSpec] backward_time = %.2lfms\n",
                 &per_device_state,
                 get_float_ptrs(acc_exp_grads).data(),
                 get_int32_ptr(gate_assign),
                 get_int32_ptr(true_gate_assign),
                 get_float_ptr(gate_pred),
                 get_float_ptr(full_gate_grad),
                 get_float_ptr(output_grad),
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

CostMetrics
    measure_operator_cost(SimEnvFactory const &sim,
                          AggregateSpecAttrs const &attrs,
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

// template <>
// void register_task<AGG_SPEC_INIT_TASK_ID>() {
//   OpTaskSignature init(OpTaskType::INIT);

//   init.add_arg_slot<AggregateSpecAttrs>(ATTRS);
//   init.add_arg_slot<PerDeviceFFHandle>(FF_HANDLE);
//   init.add_return_value<AggregateSpecPerDeviceState>();

//   register_task(AGG_SPEC_INIT_TASK_ID, "AggregateSpec Init", init,
//   init_task);
// }

template <>
void register_task<AGG_SPEC_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<AggregateSpecAttrs>(ATTRS);
  fwd.add_unchecked_arg_slot<AggregateSpecPerDeviceState>(PER_DEVICE_STATE);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);

  fwd.add_input_slot(GATE_PREDS);
  fwd.add_input_slot(GATE_ASSIGN);
  fwd.add_input_slot(EXP_PREDS, SlotType::VARIADIC);
  fwd.add_output_slot(OUTPUT);

  register_task(AGG_SPEC_FWD_TASK_ID, "AggregateSpec Fwd", fwd, forward_task);
}

template <>
void register_task<AGG_SPEC_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(AGG_SPEC_FWD_TASK_ID));

  register_task(AGG_SPEC_BWD_TASK_ID, "AggregateSpec Bwd", bwd, backward_task);
}

} // namespace FlexFlow

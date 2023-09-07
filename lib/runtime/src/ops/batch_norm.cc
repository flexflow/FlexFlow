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

#include "batch_norm.h"
#include "kernels/batch_norm_kernels.h"
#include "legion/legion_utilities.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::BatchNorm;

using Legion::Context;
using Legion::PhysicalRegion;
using Legion::Runtime;
using Legion::Task;

enum Slots {
  INPUT,  // tensor
  SCALE,  // tensor
  BIAS,   // tensor
  OUTPUT, // tensor
  ATTRS,
  PROFILING,
  PER_DEVICE_STATE,
  RELU,
  HANDLE
};

OpTaskInvocation init(BatchNormAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_tensor(0));
  binding.bind(BIAS, input_tensor(2));
  binding.bind(OUTPUT, output_tensor(0));

  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(HANDLE, ff_handle());

  return {BATCHNORM_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(BatchNormAttrs const &attrs) {
  OpTaskBinding binding;
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<BatchNormPerDeviceState>());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(SCALE, input_tensor(1));
  binding.bind(BIAS, input_tensor(2));
  binding.bind(OUTPUT, output_tensor(0));

  return {BATCHNORM_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(BatchNormAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {BATCHNORM_BWD_TASK_ID, binding};
}

static DeviceSpecific<BatchNormPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  Allocator allocator = acc.get_allocator();
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto const &attrs = acc.get_argument<BatchNormAttrs>(ATTRS);

  int output_w = output.shape[legion_dim_t(0)];
  int output_h = output.shape[legion_dim_t(1)];
  int output_c = output.shape[legion_dim_t(2)];
  int output_n = output.shape[legion_dim_t(3)];

  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffTensorDescriptor_t biasTensor;
  ffActivationDescriptor_t actiDesc;
  ffBatchNormMode_t mode;

  size_t totalSize = sizeof(float) * output_c * 4;
  float *runningMean = (float *)allocator.allocate(totalSize);
  float *runningVar = (float *)runningMean + output_c;
  float *saveMean = (float *)runningVar + output_c;
  float *saveVar = (float *)saveMean + output_c;

  DeviceSpecific<BatchNormPerDeviceState> per_device_state =
      acc.create_device_specific<BatchNormPerDeviceState>(
          init_kernel(handle,
                      allocator,
                      inputTensor,
                      outputTensor,
                      biasTensor,
                      actiDesc,
                      mode,
                      runningMean,
                      runningVar,
                      saveMean,
                      saveVar,
                      output_n,
                      output_c,
                      output_h,
                      output_w,
                      profiling,
                      attrs.relu));

  return per_device_state;
}

static DeviceSpecific<BatchNormPerDeviceState>
    init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<BatchNormPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto scale = acc.get_tensor<Permissions::RO>(SCALE);
  auto bias = acc.get_tensor<Permissions::RO>(SCALE);

  return profile(forward_kernel,
                 profiling,
                 "[BatchNorm] forward_time = %.2lfms\n",
                 &per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr(),
                 scale.get_float_ptr(),
                 bias.get_float_ptr());
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<BatchNormPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RW>(OUTPUT);
  auto scale = acc.get_tensor<Permissions::RO>(SCALE);
  auto scale_grad = acc.get_tensor_grad<Permissions::RW>(SCALE);
  auto bias_grad = acc.get_tensor_grad<Permissions::RW>(BIAS);

  return profile(backward_kernel,
                 profiling,
                 "[BatchNorm] backward_time = %.2lfms\n",
                 &per_device_state,
                 input.get_float_ptr(),
                 output_grad.get_float_ptr(),
                 output.get_float_ptr(),
                 input_grad.get_float_ptr(),
                 scale.get_float_ptr(),
                 scale_grad.get_float_ptr(),
                 bias_grad.get_float_ptr(),
                 output.shape.get_volume());
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  BatchNormAttrs const &attrs,
                                  InputParallelTensorDesc const &input_shape,
                                  InputParallelTensorDesc const &scale_shape,
                                  InputParallelTensorDesc const &bias_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv) {

  // int output_w = sub_output.dims[0].size;
  // int output_h = sub_output.dims[1].size;
  // int output_c = sub_output.dims[2].size;
  // int output_n = sub_output.dims[3].size;
  // BatchNormPerDeviceState *m = new BatchNormPerDeviceState(
  //     sim->handler, this, sim->memory, output_n, output_c, output_h,
  //     output_w);

  // sim->free_all();
  // float *input_ptr = (float *)sim->allocate(sub_input.get_volume(),
  // DT_FLOAT); assert(input_ptr != NULL); cost_metrics.inputs_memory +=
  // cost_metrics.total_mem_diff_from(sim->offset);

  // float *output_ptr = (float *)sim->allocate(sub_output.get_volume(),
  // DT_FLOAT); assert(output_ptr != NULL); cost_metrics.outputs_memory +=
  // cost_metrics.total_mem_diff_from(sim->offset);

  // float *bias_ptr = (float *)sim->allocate(output_c, DT_FLOAT);
  // assert(bias_ptr != NULL);
  // float *scale_ptr = (float *)sim->allocate(output_c, DT_FLOAT);
  // assert(scale_ptr != NULL);
  // cost_metrics.weights_memory +=
  // cost_metrics.total_mem_diff_from(sim->offset);

  auto env = sim.new_environment();

  ParallelTensorShape output_shape = get_output_shape(attrs);

  SimTaskBinding init_binding;
  init_binding.bind(INPUT, input_shape);
  init_binding.bind(BIAS, bias_shape);
  init_binding.bind(OUTPUT, output_shape);

  init_binding.bind_arg(ATTRS, attrs);
  init_binding.bind_arg(PROFILING, settings);
  init_binding.bind_arg(HANDLE, ff_handle());

  auto init_accessor =
      env.get_init_accessor(ATTENTION_INIT_TASK_ID, init_binding);
  DeviceSpecific<BatchNormPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  fwd_binding.bind(INPUT, input_shape);
  fwd_binding.bind(SCALE, scale_shape);
  fwd_binding.bind(BIAS, bias_shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(ATTENTION_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(ATTENTION_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<BATCHNORM_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);
  init.add_input_slot(INPUT);
  init.add_input_slot(BIAS);
  init.add_output_slot(OUTPUT);
  init.add_arg_slot<BatchNormAttrs>(ATTRS);
  init.add_arg_slot<bool>(PROFILING);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  register_task(BATCHNORM_INIT_TASK_ID, "BatchNorm Init", init, init_task);
}

template <>
void register_task<BATCHNORM_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(INPUT);
  fwd.add_input_slot(SCALE);
  fwd.add_input_slot(BIAS);
  fwd.add_output_slot(OUTPUT);
  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_unchecked_arg_slot<BatchNormPerDeviceState>(PER_DEVICE_STATE);

  register_task(BATCHNORM_FWD_TASK_ID, "BatchNorm Fwd", fwd, forward_task);
}

template <>
void register_task<BATCHNORM_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(BATCHNORM_FWD_TASK_ID));

  register_task(BATCHNORM_BWD_TASK_ID, "BatchNorm Bwd", bwd, backward_task);
}

}; // namespace FlexFlow
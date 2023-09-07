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

#include "reverse.h"
#include "kernels/reverse_kernels.h"
#include "kernels/accessor.h"
#include "op-attrs/get_output_shapes.h"

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

using namespace FlexFlow::Kernels::Reverse;

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING };

OpTaskInvocation init(ReverseAttrs const & attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {REVERSE_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(ReverseAttrs const & attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PROFILING, profiling_settings());

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));
  return {REVERSE_FWD_TASK_ID, binding};
}
OpTaskInvocation backward(ReverseAttrs const & attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {REVERSE_BWD_TASK_ID, binding};  
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  Context ctx = acc.ctx;
  Runtime *runtime = acc.runtime;
  Task *task = acc.task;

  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  Reverse const *reverse = (Reverse const *)task->args;
  int axis = in_domain.get_dim() - reverse->axis - 1;
  coord_t in_blk_size = 1, reverse_dim_size = 1, num_out_blks = 1;
  for (int i = 0; i < out_domain.get_dim(); i++) {
    if (i < axis) {
      in_blk_size *= out_domain.hi()[i] - out_domain.lo()[i] + 1;
    } else if (i == axis) {
      reverse_dim_size = out_domain.hi()[i] - out_domain.lo()[i] + 1;
    } else {
      num_out_blks *= out_domain.hi()[i] - out_domain.lo()[i] + 1;
    }
  }
  int output_size = out_domain.get_volume();

  return profile(forward_kernel,
                 profiling,
                 "[reverse] forward_time = %.2lfms\n",
                 input.get_float_ptr(),
                 output.get_float_ptr(),
                 num_out_blks,
                 reverse_dim_size,
                 in_blk_size,
                 output_size);
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input_grad = acc.get_tensor_grad<Permissions::RO>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::WO>(OUTPUT);

  Context ctx = acc.ctx;
  Runtime *runtime = acc.runtime;
  Task *task = acc.task;

  Domain out_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain in_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(out_grad_domain == in_grad_domain);

  Reverse const *reverse = (Reverse const *)task->args;
  int axis = in_grad_domain.get_dim() - reverse->axis - 1;

  for (int i = 0; i < in_grad_domain.get_dim(); i++) {
    if (i < axis) {
      in_blk_size *= in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1;
    } else if (i == axis) {
      reverse_dim_size = in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1;
    } else {
      num_out_blks *= in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1;
    }
  }

  return profile(backward_kernel,
                 profiling,
                 "[reverse] backward_time = %.2lfms\n",
                 output_grad.get_float_ptr(),
                 input_grad.get_float_ptr(),
                 num_out_blks,
                 reverse_dim_size,
                 in_blk_size,
                 in_grad_domain.get_volume());
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ReverseAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
  auto env = sim.new_environment();

  SimTaskBinding fwd_binding;
  fwd_binding.bind(INPUT, input);

  ParallelTensorShape output_shape = get_output_shape(attrs, input.shape);

  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(PROFILING, settings);

  auto fwd_accessor = env.get_fwd_accessor(REVERSE_FWD_TASK_ID, fwd_binding);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);
  //TODO Note: what should bwd_bindinig  bind?
  //according to aggregate.cc, bwd_binding bind the full_gate_gradients and true_gate_assign
  auto bwd_accessor = env.get_bwd_accessor(REVERSE_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);

  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<REVERSE_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);
  init.add_output_slot(OUTPUT);
    // TODO: should we implement the init_task? how to do it? because reverse doesn't need ReversePerDeviceOpState like cast 
  // register_task(REVERSE_INIT_TASK_ID, "Reverse init", init , init_task);
}

template <>
void register_task<REVERSE_FWD_TASK_ID>()) {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  register_task(REVERSE_FWD_TASK_ID, "Reverse forward", fwd, forward_task);
}

template <>
void register_task<REVERSE_BWD_TASK_ID>() {
  OpTaskSignature bwd = infer_bwd_signature(get_op_signature(REVERSE_BWD_TASK_ID));
  register_task(REVERSE_BWD_TASK_ID, "Reverse backward", bwd, backward_task);
}


// Tensor FFModel::reverse(const Tensor input, int axis, char const *name) {
//   assert(false);
// #ifdef DEADCODE
//   Reverse *reverse = new Reverse(*this, input, axis, name);
//   layers.push_back(reverse);
//   return reverse->outputs[0];
// #endif
// }

// Reverse::Reverse(FFModel &model,
//                  const ParallelTensor input,
//                  int _axis,
//                  char const *name)
//     : Op(model,
//          OP_REVERSE,
//          input->data_type,
//          name,
//          1 /*inputs*/,
//          0 /*weights*/,
//          1 /*outputs*/,
//          input),
//       axis(_axis) {
//   numOutputs = 1;
//   int numdim = input->num_dims;
//   ParallelDim dims[MAX_TENSOR_DIM];
//   for (int i = 0; i < numdim; i++) {
//     dims[i] = input->dims[i];
//   }
//   outputs[0] = model.create_parallel_tensor_legion_ordering(
//       numdim, dims, input->data_type, this);
// }

// void Reverse::init(FFModel const &ff) {
//   assert(check_output_input_weight_same_parallel_is());
//   parallel_is = outputs[0]->parallel_is;
//   ArgumentMap argmap;
//   Context ctx = ff.config.lg_ctx;
//   Runtime *runtime = ff.config.lg_hlr;
//   IndexLauncher launcher(REVERSE_INIT_TASK_ID,
//                          parallel_is,
//                          TaskArgument(this, sizeof(Reverse)),
//                          argmap,
//                          Predicate::TRUE_PRED,
//                          false /*must*/,
//                          0 /*mapper_id*/,
//                          outputs[0]->machine_view.hash());
//   launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     inputs[0]->region));
//   launcher.add_field(0, FID_DATA);
//   launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
//                                                     0 /*projection id*/,
//                                                     WRITE_ONLY,
//                                                     EXCLUSIVE,
//                                                     outputs[0]->region));
//   launcher.add_field(1, FID_DATA);
//   runtime->execute_index_space(ctx, launcher);
// }

// PerDeviceOpState *Reverse::init_task(Task const *task,
//                                      std::vector<PhysicalRegion> const &regions,
//                                      Context ctx,
//                                      Runtime *runtime) {
//   return NULL;
// }

// void Reverse::forward(FFModel const &ff) {
//   ArgumentMap argmap;
//   Context ctx = ff.config.lg_ctx;
//   Runtime *runtime = ff.config.lg_hlr;
//   IndexLauncher launcher(REVERSE_FWD_TASK_ID,
//                          parallel_is,
//                          TaskArgument(this, sizeof(Reverse)),
//                          argmap,
//                          Predicate::TRUE_PRED,
//                          false /*must*/,
//                          0 /*mapper_id*/,
//                          outputs[0]->machine_view.hash());
//   launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     inputs[0]->region));
//   launcher.add_field(0, FID_DATA);
//   launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
//                                                     0 /*projection id*/,
//                                                     WRITE_ONLY,
//                                                     EXCLUSIVE,
//                                                     outputs[0]->region));
//   launcher.add_field(1, FID_DATA);
//   runtime->execute_index_space(ctx, launcher);
// }

// void Reverse::forward_task(Task const *task,
//                            std::vector<PhysicalRegion> const &regions,
//                            Context ctx,
//                            Runtime *runtime) {
//   assert(regions.size() == 2);
//   assert(task->regions.size() == 2);
//   Reverse const *reverse = (Reverse const *)task->args;
//   Domain in_domain = runtime->get_index_space_domain(
//       ctx, task->regions[0].region.get_index_space());
//   Domain out_domain = runtime->get_index_space_domain(
//       ctx, task->regions[1].region.get_index_space());
//   assert(out_domain == in_domain);
//   float const *in_ptr = helperGetTensorPointerRO<float>(
//       regions[0], task->regions[0], FID_DATA, ctx, runtime);
//   float *out_ptr = helperGetTensorPointerWO<float>(
//       regions[1], task->regions[1], FID_DATA, ctx, runtime);
//   int axis = in_domain.get_dim() - reverse->axis - 1;
//   coord_t in_blk_size = 1, reverse_dim_size = 1, num_out_blks = 1;
//   for (int i = 0; i < out_domain.get_dim(); i++) {
//     if (i < axis) {
//       in_blk_size *= out_domain.hi()[i] - out_domain.lo()[i] + 1;
//     } else if (i == axis) {
//       reverse_dim_size = out_domain.hi()[i] - out_domain.lo()[i] + 1;
//     } else {
//       num_out_blks *= out_domain.hi()[i] - out_domain.lo()[i] + 1;
//     }
//   }
//   int output_size = out_domain.get_volume();

//   forward_kernel_wrapper(in_ptr,
//                          out_ptr,
//                          num_out_blks,
//                          reverse_dim_size,
//                          in_blk_size,
//                          output_size);
// }

// void Reverse::backward(FFModel const &ff) {
//   ArgumentMap argmap;
//   Context ctx = ff.config.lg_ctx;
//   Runtime *runtime = ff.config.lg_hlr;
//   IndexLauncher launcher(REVERSE_BWD_TASK_ID,
//                          parallel_is,
//                          TaskArgument(this, sizeof(Reverse)),
//                          argmap,
//                          Predicate::TRUE_PRED,
//                          false /*must*/,
//                          0 /*mapper_id*/,
//                          outputs[0]->machine_view.hash());
//   // regions[0](I): output_grad
//   launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
//                                                     0 /*projection id*/,
//                                                     READ_ONLY,
//                                                     EXCLUSIVE,
//                                                     outputs[0]->region_grad));
//   launcher.add_field(0, FID_DATA);
//   // regions[1](I/O): input0_grad
//   launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
//                                                     0 /*projection id*/,
//                                                     READ_WRITE,
//                                                     EXCLUSIVE,
//                                                     inputs[0]->region_grad));
//   launcher.add_field(1, FID_DATA);
//   runtime->execute_index_space(ctx, launcher);
// }

// void Reverse::backward_task(Task const *task,
//                             std::vector<PhysicalRegion> const &regions,
//                             Context ctx,
//                             Runtime *runtime) {
//   assert(regions.size() == 2);
//   assert(task->regions.size() == 2);
//   Reverse const *reverse = (Reverse const *)task->args;
//   Domain out_grad_domain = runtime->get_index_space_domain(
//       ctx, task->regions[0].region.get_index_space());
//   Domain in_grad_domain = runtime->get_index_space_domain(
//       ctx, task->regions[1].region.get_index_space());
//   assert(out_grad_domain == in_grad_domain);
//   float const *out_grad_ptr = helperGetTensorPointerRO<float>(
//       regions[0], task->regions[0], FID_DATA, ctx, runtime);
//   float *in_grad_ptr = helperGetTensorPointerRW<float>(
//       regions[1], task->regions[1], FID_DATA, ctx, runtime);
//   // We reuse the forward kernel for backward tasks
//   int axis = in_grad_domain.get_dim() - reverse->axis - 1;
//   coord_t in_blk_size = 1, reverse_dim_size = 1, num_out_blks = 1;
//   for (int i = 0; i < in_grad_domain.get_dim(); i++) {
//     if (i < axis) {
//       in_blk_size *= in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1;
//     } else if (i == axis) {
//       reverse_dim_size = in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1;
//     } else {
//       num_out_blks *= in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1;
//     }
//   }

//   backward_kernel_wrapper(out_grad_ptr,
//                           in_grad_ptr,
//                           num_out_blks,
//                           reverse_dim_size,
//                           in_blk_size,
//                           in_grad_domain.get_volume());
// }

// bool Reverse::measure_operator_cost(Simulator *sim,
//                                     MachineView const &mv,
//                                     CostMetrics &cost_metrics) const {
//   ParallelTensorBase sub_input, sub_output;
//   if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
//     return false;
//   }
//   if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
//     return false;
//   }

//   sim->free_all();
//   float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
//   assert(input_ptr != NULL);
//   cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

//   float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
//   assert(output_ptr != NULL);
//   cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

//   coord_t in_blk_size = 1, reverse_dim_size = 1, num_out_blks = 1;
//   for (int i = 0; i < sub_output.num_dims; i++) {
//     if (i < axis) {
//       in_blk_size *= sub_output.dims[i].size;
//     } else if (i == axis) {
//       reverse_dim_size = sub_output.dims[i].size;
//     } else {
//       num_out_blks *= sub_output.dims[i].size;
//     }
//   }

//   std::function<void()> forward, backward;
//   forward = [&] {
//     forward_kernel_wrapper(input_ptr,
//                            output_ptr,
//                            num_out_blks,
//                            reverse_dim_size,
//                            in_blk_size,
//                            sub_output.get_volume());
//   };
//   if (sim->computationMode == COMP_MODE_TRAINING) {
//     float *input_grad_ptr =
//         (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
//     assert(input_grad_ptr != NULL);
//     cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

//     float *output_grad_ptr =
//         (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
//     assert(output_grad_ptr != NULL);
//     cost_metrics.outputs_memory +=
//         cost_metrics.total_mem_diff_from(sim->offset);

//     backward = [&] {
//       backward_kernel_wrapper(output_grad_ptr,
//                               input_grad_ptr,
//                               num_out_blks,
//                               reverse_dim_size,
//                               in_blk_size,
//                               sub_input.get_volume());
//     };
//   }

//   inner_measure_operator_cost(sim, forward, backward, cost_metrics);

//   if (sim->computationMode == COMP_MODE_TRAINING) {
//     printf(
//         "[Measure Reverse] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
//         name,
//         cost_metrics.forward_time,
//         cost_metrics.backward_time);
//   } else {
//     printf("[Measure Reverse] name(%s) forward_time(%.4lf)\n",
//            name,
//            cost_metrics.forward_time);
//   }

//   return true;
// }

}; // namespace FlexFlow

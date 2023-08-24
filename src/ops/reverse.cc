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

#include "flexflow/ops/reverse.h"

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

Tensor FFModel::reverse(const Tensor input, int axis, char const *name) {
  assert(false);
#ifdef DEADCODE
  Reverse *reverse = new Reverse(*this, input, axis, name);
  layers.push_back(reverse);
  return reverse->outputs[0];
#endif
}

Reverse::Reverse(FFModel &model,
                 const ParallelTensor input,
                 int _axis,
                 char const *name)
    : Op(model,
         OP_REVERSE,
         input->data_type,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         input),
      axis(_axis) {
  numOutputs = 1;
  int numdim = input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = input->dims[i];
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, input->data_type, this);
}

void Reverse::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  IndexLauncher launcher(REVERSE_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Reverse)),
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
  runtime->execute_index_space(ctx, launcher);
}

OpMeta *Reverse::init_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  return NULL;
}

void Reverse::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  IndexLauncher launcher(REVERSE_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Reverse)),
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
  runtime->execute_index_space(ctx, launcher);
}

void Reverse::forward_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  Reverse const *reverse = (Reverse const *)task->args;
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(out_domain == in_domain);
  float const *in_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float *out_ptr = helperGetTensorPointerWO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
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

  Reverse::forward_kernel_wrapper(in_ptr,
                                  out_ptr,
                                  num_out_blks,
                                  reverse_dim_size,
                                  in_blk_size,
                                  output_size);
}

void Reverse::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  IndexLauncher launcher(REVERSE_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Reverse)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0](I): output_grad
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  // regions[1](I/O): input0_grad
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Reverse::backward_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  Reverse const *reverse = (Reverse const *)task->args;
  Domain out_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain in_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  assert(out_grad_domain == in_grad_domain);
  float const *out_grad_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float *in_grad_ptr = helperGetTensorPointerRW<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  // We reuse the forward kernel for backward tasks
  int axis = in_grad_domain.get_dim() - reverse->axis - 1;
  coord_t in_blk_size = 1, reverse_dim_size = 1, num_out_blks = 1;
  for (int i = 0; i < in_grad_domain.get_dim(); i++) {
    if (i < axis) {
      in_blk_size *= in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1;
    } else if (i == axis) {
      reverse_dim_size = in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1;
    } else {
      num_out_blks *= in_grad_domain.hi()[i] - in_grad_domain.lo()[i] + 1;
    }
  }

  Reverse::backward_kernel_wrapper(out_grad_ptr,
                                   in_grad_ptr,
                                   num_out_blks,
                                   reverse_dim_size,
                                   in_blk_size,
                                   in_grad_domain.get_volume());
}

bool Reverse::measure_operator_cost(Simulator *sim,
                                    MachineView const &mv,
                                    CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_input, sub_output;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }

  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_ptr != NULL);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  coord_t in_blk_size = 1, reverse_dim_size = 1, num_out_blks = 1;
  for (int i = 0; i < sub_output.num_dims; i++) {
    if (i < axis) {
      in_blk_size *= sub_output.dims[i].size;
    } else if (i == axis) {
      reverse_dim_size = sub_output.dims[i].size;
    } else {
      num_out_blks *= sub_output.dims[i].size;
    }
  }

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel_wrapper(input_ptr,
                           output_ptr,
                           num_out_blks,
                           reverse_dim_size,
                           in_blk_size,
                           sub_output.get_volume());
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr =
        (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert(input_grad_ptr != NULL);
    cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

    float *output_grad_ptr =
        (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert(output_grad_ptr != NULL);
    cost_metrics.outputs_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);

    backward = [=] {
      backward_kernel_wrapper(output_grad_ptr,
                              input_grad_ptr,
                              num_out_blks,
                              reverse_dim_size,
                              in_blk_size,
                              sub_input.get_volume());
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf(
        "[Measure Reverse] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Reverse] name(%s) forward_time(%.4lf)\n",
           name,
           cost_metrics.forward_time);
  }

  return true;
}

}; // namespace FlexFlow

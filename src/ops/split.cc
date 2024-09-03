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

#include "flexflow/ops/split.h"
#include "flexflow/model.h"
#include "flexflow/ops/kernels/split_kernels.h"
#include "flexflow/utils/hash_utils.h"

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
using PCG::Node;

using namespace FlexFlow::Kernels::Split;

bool operator==(SplitParams const &lhs, SplitParams const &rhs) {
  return lhs.splits == rhs.splits && lhs.legion_axis == rhs.legion_axis;
}

bool SplitParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

SplitParams Split::get_params() const {
  SplitParams params;
  params.splits = this->splits;
  params.legion_axis = this->legion_axis;
  if (strlen(this->name) < MAX_OPNAME) {
    strcpy(params.name, this->name);
  }
  return params;
}

void FFModel::split(const Tensor input,
                    Tensor *outputs,
                    std::vector<int> const &splits,
                    int axis,
                    char const *name) {
  Layer *split = new Layer(this,
                           OP_SPLIT,
                           DT_FLOAT,
                           name,
                           1 /*inputs*/,
                           0 /*weights*/,
                           splits.size() /*outputs*/,
                           input);
  int numdim = input->num_dims;
  int dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = input->dims[i];
  }
  for (size_t i = 0; i < splits.size(); i++) {
    dims[numdim - axis - 1] = splits[i];
    split->outputs[i] = create_tensor_legion_ordering(
        numdim, dims, input->data_type, split, 0, true /*create_grad*/);
    outputs[i] = split->outputs[i];
  }
  split->add_int_property("legion_axis", numdim - axis - 1);
  layers.push_back(split);
}

Op *Split::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("legion_axis", value);
  int legion_axis = value;
  std::vector<int> splits;
  for (int i = 0; i < layer->numOutputs; i++) {
    splits.push_back(layer->outputs[i]->dims[legion_axis]);
  }
  assert(inputs.size() == 1);
  return new Split(model, inputs[0], splits, legion_axis, layer->name);
}

Split::Split(FFModel &model,
             const ParallelTensor input,
             std::vector<int> const &splits,
             int _legion_axis,
             char const *name)
    : Op(model,
         OP_SPLIT,
         input->data_type,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         splits.size() /*outputs*/,
         input),
      legion_axis(_legion_axis), splits(splits) {
  numOutputs = splits.size();
  // Note that we use the Legion dim ordering
  assert(legion_axis >= 0);
  numWeights = 0;
  int split_size = 0;
  for (int i = 0; i < numOutputs; i++) {
    split_size += splits[i];
    int numdim = input->num_dims;
    ParallelDim dims[MAX_TENSOR_DIM];
    for (int j = 0; j < numdim; j++) {
      dims[j] = input->dims[j];
    }
    dims[legion_axis].size = splits[i];
    // Assert the _axis dim cannot be parallelized
    assert(dims[legion_axis].degree == 1);
    assert(dims[legion_axis].parallel_idx == -1);
    outputs[i] = model.create_parallel_tensor_legion_ordering(
        numdim, dims, input->data_type, this /*owner_op*/, i /*owner_idx*/);
  }
  // Check split sizes
  assert(split_size == input->dims[legion_axis].size);
}

Split::Split(FFModel &model,
             SplitParams const &params,
             const ParallelTensor input,
             char const *name)
    : Split(model, input, params.splits, params.legion_axis, params.name) {}

void Split::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  IndexLauncher launcher(SPLIT_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Split)),
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
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(RegionRequirement(outputs[i]->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[i]->region));
    launcher.add_field(i + 1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void Split::init_inference(FFModel const &ff,
                           std::vector<ParallelTensor> const &batch_inputs,
                           std::vector<ParallelTensor> const &batch_outputs,
                           MachineView const *mv) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = batch_outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  size_t machine_view_hash = view->hash();
  set_argumentmap_for_init_inference(ff, argmap, batch_outputs[0]);

  IndexLauncher launcher(SPLIT_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Split)),
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
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(
        RegionRequirement(batch_outputs[i]->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_outputs[i]->region));
    launcher.add_field(i + 1, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

OpMeta *Split::init_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  return NULL;
}

void Split::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  IndexLauncher launcher(SPLIT_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Split)),
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
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(RegionRequirement(outputs[i]->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[i]->region));
    launcher.add_field(i + 1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}
FutureMap Split::inference(FFModel const &ff,
                           BatchConfigFuture const &bc,
                           std::vector<ParallelTensor> const &batch_inputs,
                           std::vector<ParallelTensor> const &batch_outputs,
                           MachineView const *mv) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  parallel_is = batch_outputs[0]->parallel_is;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  set_argumentmap_for_inference(ff, argmap, batch_outputs[0]);
  size_t machine_view_hash = view->hash();

  IndexLauncher launcher(SPLIT_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Split)),
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
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(
        RegionRequirement(batch_outputs[i]->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_outputs[i]->region));
    launcher.add_field(i + 1, FID_DATA);
  }
  return runtime->execute_index_space(ctx, launcher);
}

void calc_block_size(coord_t &num_blks,
                     coord_t &blk_size,
                     Domain const &domain,
                     int axis) {
  num_blks = 1;
  blk_size = 1;
  for (int d = 0; d < domain.get_dim(); d++) {
    if (d <= axis) {
      blk_size *= (domain.hi()[d] - domain.lo()[d] + 1);
    } else {
      num_blks *= (domain.hi()[d] - domain.lo()[d] + 1);
    }
  }
}

void Split::forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  Split const *split = (Split *)task->args;
  assert(regions.size() == split->numOutputs + 1);
  assert(task->regions.size() == split->numOutputs + 1);
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  float *out_ptr[MAX_NUM_OUTPUTS];
  size_t total_volume = 0;
  float const *in_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  coord_t num_blks, in_blk_size, out_blk_size[MAX_NUM_OUTPUTS];
  calc_block_size(num_blks, in_blk_size, in_domain, split->legion_axis);
  for (int i = 0; i < split->numOutputs; i++) {
    Domain out_domain = runtime->get_index_space_domain(
        ctx, task->regions[i + 1].region.get_index_space());
    out_ptr[i] = helperGetTensorPointerWO<float>(
        regions[i + 1], task->regions[i + 1], FID_DATA, ctx, runtime);
    coord_t out_num_blks;
    calc_block_size(
        out_num_blks, out_blk_size[i], out_domain, split->legion_axis);
    assert(out_num_blks == num_blks);
    for (int j = 0; j < out_domain.get_dim(); j++) {
      if (j != split->legion_axis) {
        assert(out_domain.hi()[j] == in_domain.hi()[j]);
        assert(out_domain.lo()[j] == in_domain.lo()[j]);
      }
    }
    total_volume += out_domain.get_volume();
  }
  assert(total_volume == in_domain.get_volume());

  forward_kernel_wrapper(
      out_ptr, in_ptr, out_blk_size, in_blk_size, num_blks, split->numOutputs);
}

void Split::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  IndexLauncher launcher(SPLIT_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Split)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(RegionRequirement(outputs[i]->part_grad,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[i]->region_grad));
    launcher.add_field(i + 1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void Split::backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  Split const *split = (Split *)task->args;
  assert(regions.size() == split->numOutputs + 1);
  assert(task->regions.size() == split->numOutputs + 1);
  Domain in_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  float const *out_grad_ptr[MAX_NUM_OUTPUTS];
  size_t total_volume = 0;
  float *in_grad_ptr = helperGetTensorPointerRW<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  coord_t num_blks, in_blk_size, out_blk_size[MAX_NUM_OUTPUTS];
  calc_block_size(num_blks, in_blk_size, in_grad_domain, split->legion_axis);
  for (int i = 0; i < split->numOutputs; i++) {
    Domain out_grad_domain = runtime->get_index_space_domain(
        ctx, task->regions[i + 1].region.get_index_space());
    out_grad_ptr[i] = helperGetTensorPointerRO<float>(
        regions[i + 1], task->regions[i + 1], FID_DATA, ctx, runtime);
    coord_t out_num_blks;
    calc_block_size(
        out_num_blks, out_blk_size[i], out_grad_domain, split->legion_axis);
    assert(out_num_blks == num_blks);
    for (int j = 0; j < out_grad_domain.get_dim(); j++) {
      if (j != split->legion_axis) {
        assert(out_grad_domain.hi()[j] == in_grad_domain.hi()[j]);
        assert(out_grad_domain.lo()[j] == in_grad_domain.lo()[j]);
      }
    }
    total_volume += out_grad_domain.get_volume();
  }
  assert(total_volume == in_grad_domain.get_volume());

  backward_kernel_wrapper(in_grad_ptr,
                          out_grad_ptr,
                          out_blk_size,
                          in_blk_size,
                          num_blks,
                          split->numOutputs);
}

bool Split::measure_operator_cost(Simulator *sim,
                                  MachineView const &mv,
                                  CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_output[MAX_NUM_OUTPUTS], sub_input;
  for (int i = 0; i < numOutputs; i++) {
    if (!outputs[i]->get_sub_tensor(mv, sub_output[i])) {
      return false;
    }
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }
  Domain in_domain = sub_input.get_domain();
  sim->free_all();
  float *output_ptr[MAX_NUM_OUTPUTS];
  size_t total_volume = 0;
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
  coord_t num_blks, in_blk_size, out_blk_size[MAX_NUM_OUTPUTS];
  calc_block_size(num_blks, in_blk_size, in_domain, legion_axis);
  for (int i = 0; i < numOutputs; i++) {
    Domain out_domain = sub_output[i].get_domain();
    output_ptr[i] =
        (float *)sim->allocate(sub_output[i].get_volume(), DT_FLOAT);
    coord_t out_num_blks;
    calc_block_size(out_num_blks, out_blk_size[i], out_domain, legion_axis);
    assert(out_num_blks == num_blks);
    for (int j = 0; j < out_domain.get_dim(); j++) {
      if (j != legion_axis) {
        assert(out_domain.hi()[j] == in_domain.hi()[j]);
        assert(out_domain.lo()[j] == in_domain.lo()[j]);
      }
    }
    total_volume += out_domain.get_volume();
  }
  assert(total_volume == in_domain.get_volume());
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel_wrapper(
        output_ptr, input_ptr, out_blk_size, in_blk_size, num_blks, numOutputs);
  };
  // Assume backward has the same cost as forward
  backward = forward;

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);
  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Split] name(%s) num_elements(%zu) forward_time(%.4lf) "
           "backward_time(%.4lf)\n",
           name,
           sub_input.get_volume(),
           cost_metrics.forward_time,
           cost_metrics.backward_time);
  } else {
    printf("[Measure Split] name(%s) num_elements(%zu) forward_time(%.4lf)\n",
           name,
           sub_input.get_volume(),
           cost_metrics.forward_time);
  }
  return true;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::SplitParams>::operator()(
    FlexFlow::SplitParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.splits.size());
  for (int n : params.splits) {
    hash_combine(key, n);
  }
  hash_combine(key, params.legion_axis);
  return key;
}
}; // namespace std

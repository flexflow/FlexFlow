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

#include "groupby.h"
#include <math.h>
#include <stdio.h>

namespace FlexFlow {
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::InlineLauncher;
using Legion::Machine;
using Legion::Memory;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;
using PCG::Node;

void FFModel::group_by(const Tensor input,
                       const Tensor assign,
                       Tensor *outputs,
                       int n,
                       float alpha,
                       char const *name) {
  Layer *li = new Layer(this,
                        OP_GROUP_BY,
                        DT_FLOAT,
                        name,
                        2 /*inputs*/,
                        0 /*weights*/,
                        n /*outputs*/,
                        input,
                        assign);
  {
    int k = assign->dims[0];
    int num_dims = input->num_dims;
    int dims[num_dims];
    for (int i = 0; i < num_dims - 1; i++) {
      dims[i] = input->dims[i];
    }
    // Batch dimension is replaced by max expert capacity
    dims[num_dims - 1] = (int)ceil(alpha * k / n * input->dims[num_dims - 1]);
    for (int i = 0; i < n; i++) {
      // Creating one tensor per expert, each with size (DATA_DIMS,
      // max_expert_capacity)
      li->outputs[i] = create_tensor_legion_ordering(
          num_dims, dims, input->data_type, li, 0, true /*create_grad*/);
    }
  }
  li->add_int_property("n", n);
  li->add_float_property("alpha", alpha);
  layers.push_back(li);
  for (int i = 0; i < n; i++) {
    assert(li->outputs[i] != nullptr);
    outputs[i] = li->outputs[i];
    assert(outputs[i] != nullptr);
  }
}

Op *Group_by::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value1;
  layer->get_int_property("n", value1);
  int n = value1;
  float value2;
  layer->get_float_property("alpha", value2);
  float alpha = value2;
  return new Group_by(model, inputs[0], inputs[1], n, alpha, layer->name);
}

Group_byParams Group_by::get_params() const {
  Group_byParams params;
  params.n = this->n;
  params.alpha = this->alpha;
  return params;
}

bool Group_byParams::is_valid(
    std::pair<ParallelTensorShape, ParallelTensorShape> const &) const {
  // Group_by is always valid
  return true;
}

bool operator==(Group_byParams const &lhs, Group_byParams const &rhs) {
  return lhs.n == rhs.n && lhs.alpha == rhs.alpha;
}

Group_by::Group_by(FFModel &model,
                   const ParallelTensor _input,
                   const ParallelTensor _assign,
                   int _n,
                   float _alpha,
                   char const *name)
    : Op(model,
         OP_GROUP_BY,
         _input->data_type,
         name,
         2 /*inputs*/,
         0 /*weights*/,
         _n /*outputs*/,
         _input,
         _assign),
      n(_n), alpha(_alpha) {
  assert(_input->dims[1] == _assign->dims[1]);
  assert(n > 0);
  assert(inputs[0] != nullptr);

  int k = _assign->dims[0].size;
  int num_dims = _input->num_dims;

  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < num_dims; i++) {
    dims[i] = inputs[0]->dims[i];
  }
  // replace batch size with max expert size
  dims[num_dims - 2].size = (int)ceil(alpha * k / n * inputs[0]->dims[1].size);

  for (int i = 0; i < n; i++) {
    outputs[i] = model.create_parallel_tensor_legion_ordering(
        num_dims, dims, DT_FLOAT, this, i /*owner_idx*/);
    assert(outputs[i] != nullptr);
  }

  numWeights = 0;
}

Group_by::Group_by(FFModel &model,
                   Group_by const &other,
                   const ParallelTensor input,
                   const ParallelTensor assign)
    : Group_by(model, input, assign, other.n, other.alpha, other.name) {}

Group_by::Group_by(FFModel &model,
                   Group_byParams const &params,
                   std::pair<ParallelTensor, ParallelTensor> const &inputs,
                   char const *name)
    : Group_by(
          model, inputs.first, inputs.second, params.n, params.alpha, name) {}

void Group_by::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(GROUP_BY_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Group_by)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // data
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // assign
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);

  // output
  for (int i = 0; i < n; i++) {
    launcher.add_region_requirement(RegionRequirement(outputs[i]->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[i]->region));
    launcher.add_field(i + 2, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

PerDeviceOpState *
    Group_by::init_task(Task const *task,
                        std::vector<PhysicalRegion> const &regions,
                        Context ctx,
                        Runtime *runtime) {
  Group_by *gb = (Group_by *)task->args;
  FFHandler handle = *((FFHandler *)task->local_args);
  GroupByMeta *m = new GroupByMeta(handle, gb->n);
  m->profiling = gb->profiling;
  return m;
}

void Group_by::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(GROUP_BY_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Group_by)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // data
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);

  // assign
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);

  // output
  for (int i = 0; i < n; i++) {
    launcher.add_region_requirement(RegionRequirement(outputs[i]->part,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[i]->region));
    launcher.add_field(i + 2, FID_DATA);
  }

  runtime->execute_index_space(ctx, launcher);
}

void Group_by::forward_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  // Get n, alpha
  Group_by const *gb = (Group_by *)task->args;
  int n = gb->n;
  float alpha = gb->alpha;

  assert((int)regions.size() == n + 2);
  assert((int)task->regions.size() == n + 2);

  GroupByMeta const *m = *((GroupByMeta **)task->local_args);

  // get input and assign regions. Each tensor has three dimensions:
  // (datapoint_dim, batch_size, replica_dim)
  AccessorRO<float, 3> const acc_input(regions[0], FID_DATA);
  AccessorRO<int, 3> const acc_assign(regions[1], FID_DATA);

  Rect<3> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<3> rect_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  coord_t input_rows = rect_input.hi[1] - rect_input.lo[1] + 1;
  coord_t input_cols = rect_input.hi[0] - rect_input.lo[0] + 1;
  assert(input_rows == rect_assign.hi[1] - rect_assign.lo[1] + 1);

  int k = rect_assign.hi[0] - rect_assign.lo[0] + 1;
  int batch_size = input_rows;
  int data_dim = input_cols;

  // Create a vector of n outputs, where n is the number of experts.
  // Each entry in the "outputs" vector points to the Legion tensor that will
  // contain the tockens dispatched to the corresponding expert
  float *outputs[n];
  int exp_output_rows = (int)ceil(alpha * k / n * batch_size);
  for (int i = 0; i < n; i++) {
    Domain out_domain = runtime->get_index_space_domain(
        ctx, task->regions[i + 2].region.get_index_space());
    outputs[i] = helperGetTensorPointerWO<float>(
        regions[i + 2], task->regions[i + 2], FID_DATA, ctx, runtime);

    coord_t output_rows = out_domain.hi()[1] - out_domain.lo()[1] + 1;
    coord_t output_cols = out_domain.hi()[0] - out_domain.lo()[0] + 1;
    assert((int)output_rows == exp_output_rows);
    assert(output_cols == input_cols);
  }

  Group_by::forward_kernel_wrapper(m,
                                   acc_input.ptr(rect_input),
                                   acc_assign.ptr(rect_assign),
                                   outputs,
                                   n,
                                   k,
                                   alpha,
                                   batch_size,
                                   data_dim);
}

void Group_by::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(GROUP_BY_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Group_by)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());

  // input_grad
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);

  // assign
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);

  // output grad
  for (int i = 0; i < n; i++) {
    launcher.add_region_requirement(RegionRequirement(outputs[i]->part_grad,
                                                      0 /*projection id*/,
                                                      WRITE_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[i]->region_grad));
    launcher.add_field(i + 2, FID_DATA);
  }

  runtime->execute_index_space(ctx, launcher);
}

void Group_by::backward_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  // Get n, alpha
  GroupByMeta const *m = *((GroupByMeta **)task->local_args);
  Group_by const *gb = (Group_by *)task->args;
  int n = gb->n;
  float alpha = gb->alpha;

  assert((int)regions.size() == n + 2);
  assert((int)task->regions.size() == n + 2);

  // get input and assign regions
  AccessorWO<float, 3> const acc_input_grad(regions[0], FID_DATA);
  AccessorRO<int, 3> const acc_assign(regions[1], FID_DATA);

  Rect<3> rect_input_grad = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<3> rect_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  coord_t input_rows = rect_input_grad.hi[1] - rect_input_grad.lo[1] + 1;
  coord_t input_cols = rect_input_grad.hi[0] - rect_input_grad.lo[0] + 1;
  assert(input_rows == rect_assign.hi[1] - rect_assign.lo[1] + 1);

  int k = rect_assign.hi[0] - rect_assign.lo[0] + 1;
  int batch_size = input_rows;
  int data_dim = input_cols;

  // get output
  float *output_grads[n];
  int exp_output_rows = (int)ceil(alpha * k / n * batch_size);
  for (int i = 0; i < n; i++) {
    Domain out_domain = runtime->get_index_space_domain(
        ctx, task->regions[i + 2].region.get_index_space());
    output_grads[i] = helperGetTensorPointerRW<float>(
        regions[i + 2], task->regions[i + 2], FID_DATA, ctx, runtime);

    coord_t output_rows = out_domain.hi()[1] - out_domain.lo()[1] + 1;
    coord_t output_cols = out_domain.hi()[0] - out_domain.lo()[0] + 1;
    assert((int)output_rows == exp_output_rows);
    assert(output_cols == input_cols);
  }

  Group_by::backward_kernel_wrapper(m,
                                    acc_input_grad.ptr(rect_input_grad),
                                    acc_assign.ptr(rect_assign),
                                    output_grads,
                                    n,
                                    k,
                                    alpha,
                                    batch_size,
                                    data_dim);
}

void Group_by::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->n);
  sez.serialize(this->alpha);
}

Node Group_by::deserialize(FFModel &ff,
                           Legion::Deserializer &dez,
                           ParallelTensor inputs[],
                           int num_inputs) {
  assert(num_inputs == 2);
  int n;
  float alpha;
  dez.deserialize(n);
  dez.deserialize(alpha);
  Group_byParams params;
  params.n = n;
  params.alpha = alpha;
  return ff.get_or_create_node<Group_by>(std::make_pair(inputs[0], inputs[1]),
                                         params);
}

Op *Group_by::materialize(FFModel &ff,
                          ParallelTensor inputs[],
                          int num_inputs) const {
  Group_byParams params = get_params();
  return new Group_by(ff, params, {inputs[0], inputs[1]}, this->name);
}

bool Group_by::measure_operator_cost(Simulator *sim,
                                     MachineView const &mv,
                                     CostMetrics &cost_metrics) const {
  assert(numOutputs <= MAX_NUM_OUTPUTS);
  ParallelTensorBase sub_input, sub_assign;
  ParallelTensorBase sub_outputs[MAX_NUM_OUTPUTS];
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_assign)) {
    return false;
  }
  for (int i = 0; i < numOutputs; ++i) {
    if (!outputs[i]->get_sub_tensor(mv, sub_outputs[i])) {
      return false;
    }
  }

  GroupByMeta *m = new GroupByMeta(sim->handler, n);

  // allocate
  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  int *assign_ptr = (int *)sim->allocate(sub_assign.get_volume(), DT_INT32);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *output_ptrs[MAX_NUM_OUTPUTS];
  bool out_of_memory = false;
  for (int i = 0; i < numOutputs; i++) {
    output_ptrs[i] =
        (float *)sim->allocate(sub_outputs[i].get_volume(), DT_FLOAT);
    out_of_memory = out_of_memory || (output_ptrs[i] == NULL);
  }
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  if (out_of_memory || !input_ptr || !assign_ptr) {
    cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    return true;
  }

  assert(m->profiling == false);

  // compute
  std::function<void()> forward, backward;

  Domain in_domain = sub_input.get_domain();
  int k = sub_assign.dims[0].size;
  int batch_size = in_domain.hi()[1] - in_domain.lo()[1] + 1;
  int data_dim = in_domain.hi()[0] - in_domain.lo()[0] + 1;

  forward = [&] {
    forward_kernel_wrapper(m,
                           input_ptr,
                           assign_ptr,
                           output_ptrs,
                           n,
                           k,
                           alpha,
                           batch_size,
                           data_dim);
  };

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);
  log_measure.debug("[Measure GroupBy] name(%s) forward_time(%.4lf)\n",
                    name,
                    cost_metrics.forward_time);

  cost_metrics.backward_time = 0.0f; // not implemented for MOE
  delete m;
  return true;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::Group_byParams>::operator()(
    FlexFlow::Group_byParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.n);
  hash_combine(key, params.alpha);
  return key;
}
}; // namespace std

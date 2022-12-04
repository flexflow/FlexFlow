/* Copyright 2019 Stanford
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

#include "flexflow/model.h"
#include "flexflow/ops/groupby.h"
#include "flexflow/utils/hash_utils.h"
#include "legion/legion_utilities.h"
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
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

void FFModel::group_by(const Tensor input,
                       const Tensor assign,
                       Tensor *outputs,
                       int n,
                       float alpha,
                       char const *name) {
//   assert(false);
// #ifdef DEADCODE
//   Group_by *group_by = new Group_by(*this, input, assign, n, alpha, name);
//   layers.push_back(group_by);
//   for (int i = 0; i < n; i++)
//     outputs[i] = group_by->outputs[i];
// #endif
  Layer *li = new Layer(this,
                        OP_GROUP_BY,
                        name,
                        2 /*inputs*/,
                        0 /*weights*/,
                        n /*outputs*/,
                        input,
                        assign);
  {
    int k = assign->dims[0];
    int num_dims = 2;
    int dims[num_dims];
    dims[0] = input->dims[0];
    dims[1] = (int)ceil(alpha * k / n * input->dims[1]);
    for (int i=0; i<n; i++) {
      li->outputs[i] = create_tensor_legion_ordering(
        num_dims, dims, input->data_type, li, 0, true /*create_grad*/);
    }

  }
  li->add_int_property("n", n);
  li->add_float_property("alpha", alpha);
  layers.push_back(li);
  for (int i=0; i<n; i++){
    outputs[i] = li->outputs[i];
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
  inputs[0]->print("inputs[0]");
  inputs[1]->print("inputs[1]");
  return new Group_by(model, inputs[0], inputs[1], n, alpha, layer->name);
}

Group_byParams Group_by::get_params() const {
  Group_byParams params;
  params.n = this->n;
  params.alpha = this->alpha;
  return params;
}

bool Group_byParams::is_valid(ParallelTensorShape const &) const {
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
  _input->print("_input");
  _assign->print("_assign");
  assert(_input->num_dims == 2+1); // NOTE: Is that a problem if you e.g. want to pass in images
  assert(_input->num_dims == 2+1);
  assert(_input->dims[1] == _assign->dims[1]);
  assert(n > 0);

  // List of outputs
  int k = _assign->dims[0].size;
  for (int i = 0; i < n; i++) {
    outputs[i]->num_dims = 2;
    outputs[i]->dims[0].size = inputs[0]->dims[0].size;
    outputs[i]->dims[1].size =
        (int)ceil(alpha * k / n * inputs[0]->dims[1].size);
  }

  numWeights = 0;
}

void Group_by::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
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
  runtime->execute_index_space(ctx, launcher);
}

OpMeta *Group_by::init_task(Task const *task,
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

  // get input and assign regions
  AccessorRO<float, 2> const acc_input(regions[0], FID_DATA);
  AccessorRO<int, 2> const acc_assign(regions[1], FID_DATA);

  Rect<2> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  coord_t input_rows = rect_input.hi[1] - rect_input.lo[1] + 1;
  coord_t input_cols = rect_input.hi[0] - rect_input.lo[0] + 1;
  assert(input_rows == rect_assign.hi[1] - rect_assign.lo[1] + 1);
  int k = rect_assign.hi[0] - rect_assign.lo[0] + 1;
  int batch_size = input_rows;
  int data_dim = input_cols;

  // get output
  float *outputs[n];
  // int exp_output_rows = (int)ceil(alpha*k/n*batch_size);
  for (int i = 0; i < n; i++) {
    Domain out_domain = runtime->get_index_space_domain(
        ctx, task->regions[i + 2].region.get_index_space());
    outputs[i] = helperGetTensorPointerWO<float>(
        regions[i + 2], task->regions[i + 2], FID_DATA, ctx, runtime);

    // coord_t output_rows = out_domain.hi()[1] - out_domain.lo()[1] + 1;
    coord_t output_cols = out_domain.hi()[0] - out_domain.lo()[0] + 1;
    // assert((int)output_rows == exp_output_rows);
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
  AccessorWO<float, 2> const acc_input_grad(regions[0], FID_DATA);
  AccessorRO<int, 2> const acc_assign(regions[1], FID_DATA);

  Rect<2> rect_input_grad = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  coord_t input_rows = rect_input_grad.hi[1] - rect_input_grad.lo[1] + 1;
  coord_t input_cols = rect_input_grad.hi[0] - rect_input_grad.lo[0] + 1;
  assert(input_rows == rect_assign.hi[1] - rect_assign.lo[1] + 1);
  int k = rect_assign.hi[0] - rect_assign.lo[0] + 1;
  int batch_size = input_rows;
  int data_dim = input_cols;

  // get output
  float *output_grads[n];
  // int exp_output_rows = (int)ceil(alpha*k/n*batch_size);
  for (int i = 0; i < n; i++) {
    Domain out_domain = runtime->get_index_space_domain(
        ctx, task->regions[i + 2].region.get_index_space());
    output_grads[i] = helperGetTensorPointerRW<float>(
        regions[i + 2], task->regions[i + 2], FID_DATA, ctx, runtime);

    // coord_t output_rows = out_domain.hi()[1] - out_domain.lo()[1] + 1;
    coord_t output_cols = out_domain.hi()[0] - out_domain.lo()[0] + 1;
    // assert((int)output_rows == exp_output_rows);
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

bool Group_by::measure_operator_cost(Simulator *sim,
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

namespace std {
size_t hash<FlexFlow::Group_byParams>::operator()(
    FlexFlow::Group_byParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.n);
  hash_combine(key, params.alpha);
  return key;
}
}; // namespace std

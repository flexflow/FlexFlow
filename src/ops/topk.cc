/* Copyright 2021 Facebook
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

#include "flexflow/ops/topk.h"

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

// For an input tensor, computes the top k entries in each row
// (resp. vector along the last dimension). Thus,
// values.shape = indices.shape = input.shape[:-1] + [k]
void FFModel::top_k(
    const Tensor input, Tensor *outputs, int k, bool sorted, char const *name) {
  assert(false);
#ifdef DEADCODE
  TopK *topk = new TopK(*this, input, k, sorted, name);
  layers.push_back(topk);
  assert(topk->numOutputs == 2);
  outputs[0] = topk->outputs[0];
  outputs[1] = topk->outputs[1];
#endif
}

TopK::TopK(FFModel &model,
           const ParallelTensor _input,
           int _k,
           bool _sorted,
           char const *name)
    : Op(model,
         OP_TOPK,
         _input->data_type,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         2 /*outputs*/,
         _input),
      k(_k), sorted(_sorted) {
  int numdim = inputs[0]->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = inputs[0]->dims[i];
  }
  dims[0].size = k;
  assert(inputs[0]->dims[0].degree == 1);
  assert(inputs[0]->dims[0].parallel_idx == -1);
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, _input->data_type, this, 0 /*owner_idx*/);
  outputs[1] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, DT_INT32, this, 1 /*owner_idx*/);
}

void TopK::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(TOPK_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(TopK)),
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
  launcher.add_region_requirement(RegionRequirement(outputs[1]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[1]->region));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *TopK::init_task(Task const *task,
                        std::vector<PhysicalRegion> const &regions,
                        Context ctx,
                        Runtime *runtime) {
  TopK *topk = (TopK *)task->args;
  FFHandler handle = *((FFHandler *)task->local_args);
  TopKMeta *m = new TopKMeta(handle);
  m->profiling = topk->profiling;
  m->sorted = topk->sorted;
  return m;
}

void TopK::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(TOPK_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
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
  launcher.add_region_requirement(RegionRequirement(outputs[1]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[1]->region));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void TopK::forward_task(Task const *task,
                        std::vector<PhysicalRegion> const &regions,
                        Context ctx,
                        Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  // const TopK* topk = (const TopK*) task->args;
  TopKMeta const *m = *((TopKMeta **)task->local_args);
  Domain in1_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain out1_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Domain out2_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());

  int in_cols = in1_domain.hi()[0] - in1_domain.lo()[0] + 1;
  int out1_cols = out1_domain.hi()[0] - out1_domain.lo()[0] + 1;
  int out2_cols = out2_domain.hi()[0] - out2_domain.lo()[0] + 1;

  assert(out1_domain == out2_domain);
  for (int i = 1; i < in1_domain.get_dim(); i++) {
    assert(in1_domain.lo()[i] == out1_domain.lo()[i]);
    assert(in1_domain.hi()[i] == out1_domain.hi()[i]);
  }
  float const *in_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float *value_ptr = helperGetTensorPointerWO<float>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  int *index_ptr = helperGetTensorPointerWO<int>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);

  int length = in1_domain.hi()[0] - in1_domain.lo()[0] + 1;
  int k =
      out1_domain.hi()[0] - out1_domain.lo()[0] + 1; /*TODO: This prints to 5*/
  size_t batch_size = in1_domain.get_volume() / length;

  TopK::forward_kernel_wrapper(
      m, in_ptr, value_ptr, index_ptr, batch_size, length, k, m->sorted);
}

void TopK::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(TOPK_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0](I): value_grad
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  // regions[1](I): indices
  launcher.add_region_requirement(RegionRequirement(outputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // regions[2](I/O): input_grad
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): out1_grad
  regions[1](I): out2
  regions[2](I/0): in_grad
*/
void TopK::backward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  // const TopK* topk = (const TopK*) task->args;
  TopKMeta const *m = *((TopKMeta **)task->local_args);
  assert(regions.size() == 3);
  Domain out1_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain out2_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  assert(out1_domain == out2_domain);
  for (int i = 1; i < in_domain.get_dim(); i++) {
    assert(in_domain.lo()[i] == out1_domain.lo()[i]);
    assert(in_domain.hi()[i] == out1_domain.hi()[i]);
  }
  float const *value_grad_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  int const *indices_ptr = helperGetTensorPointerRO<int>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  float *in_grad_ptr = helperGetTensorPointerRW<float>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);

  int length = in_domain.hi()[0] - in_domain.lo()[0] + 1;
  int k = out1_domain.hi()[0] - out1_domain.lo()[0] + 1;
  size_t batch_size = in_domain.get_volume() / length;
  TopK::backward_kernel_wrapper(
      m, value_grad_ptr, indices_ptr, in_grad_ptr, batch_size, length, k);
}

bool TopK::measure_operator_cost(Simulator *sim,
                                 MachineView const &mv,
                                 CostMetrics &cost_metrics) const {
  // To be implemented
  assert(false);
  return false;
}

}; // namespace FlexFlow

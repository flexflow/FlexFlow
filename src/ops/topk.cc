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

#include "flexflow/ops/topk.h"
#include "flexflow/model.h"
#include "flexflow/utils/hash_utils.h"
#include "legion/legion_utilities.h"

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

// For an input tensor, computes the top k entries in each row
// (resp. vector along the last dimension). Thus,
// values.shape = indices.shape = input.shape[:-1] + [k]
void FFModel::top_k(
    const Tensor input, Tensor *outputs, int k, bool sorted, char const *name) {
  Layer *li = new Layer(this,
                        OP_TOPK,
                        input->data_type,
                        name,
                        1 /*inputs*/,
                        0 /*weights*/,
                        2 /*outputs*/,
                        input);
  {
    int numdims = input->num_dims;
    int dims[MAX_TENSOR_DIM];
    for (int i = 0; i < numdims; i++) {
      dims[i] = input->dims[i];
    }
    dims[0] = k;
    li->outputs[0] = create_tensor_legion_ordering(
        numdims, dims, input->data_type, li, 0, true /*create_grad*/);
    li->outputs[1] = create_tensor_legion_ordering(
        numdims, dims, DT_INT32, li, 0, true /*create_grad*/);
  }
  li->add_int_property("k", k);
  li->add_int_property("sorted", sorted);
  layers.push_back(li);
  outputs[0] = li->outputs[0];
  outputs[1] = li->outputs[1];
}

Op *TopK::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("k", value);
  int k = value;
  layer->get_int_property("sorted", value);
  bool sorted = (bool)value;
  return new TopK(model, inputs[0], k, sorted, layer->name);
}

TopKParams TopK::get_params() const {
  TopKParams params;
  params.k = this->k;
  params.sorted = this->sorted;
  if (this->name != nullptr) {
    strcpy(params.name, this->name);
  }
  return params;
}

bool TopKParams::is_valid(ParallelTensorShape const &) const {
  // topk is always valid
  return true;
}

bool operator==(TopKParams const &lhs, TopKParams const &rhs) {
  return lhs.k == rhs.k && lhs.sorted == rhs.sorted;
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

TopK::TopK(FFModel &model, TopK const &other, const ParallelTensor input)
    : TopK(model, input, other.k, other.sorted, other.name) {}

TopK::TopK(FFModel &model,
           TopKParams const &params,
           const ParallelTensor input,
           char const *name)
    : TopK(model, input, params.k, params.sorted, params.name) {}

void TopK::init_inference(FFModel const &ff,
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
  IndexLauncher launcher(TOPK_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(TopK)),
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
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[1]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[1]->region));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
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
  m->inference_debugging = topk->inference_debugging;
  m->sorted = topk->sorted;
  std::strcpy(m->op_name, topk->name);
  m->layer_guid = topk->layer_guid;
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

FutureMap TopK::inference(FFModel const &ff,
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
  /* std::cout << "TopK op machine_view: " << *(MachineView const *)mv
            << std::endl; */
  IndexLauncher launcher(TOPK_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
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
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[1]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[1]->region));
  launcher.add_field(2, FID_DATA);
  return runtime->execute_index_space(ctx, launcher);
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

void TopK::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->k);
  sez.serialize(this->sorted);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

Node TopK::deserialize(FFModel &ff,
                       Legion::Deserializer &dez,
                       ParallelTensor inputs[],
                       int num_inputs) {
  assert(num_inputs == 1);
  int k;
  bool sorted;
  dez.deserialize(k);
  dez.deserialize(sorted);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);
  TopKParams params;
  params.k = k;
  params.sorted = sorted;
  strcpy(params.name, name);
  return ff.get_or_create_node<TopK>(inputs[0], params);
}

Op *TopK::materialize(FFModel &ff,
                      ParallelTensor inputs[],
                      int num_inputs) const {
  TopKParams params = get_params();
  return new TopK(ff, params, inputs[0], this->name);
}

bool TopK::measure_operator_cost(Simulator *sim,
                                 MachineView const &mv,
                                 CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_input, sub_output, sub_output_ind;
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!outputs[1]->get_sub_tensor(mv, sub_output_ind)) {
    return false;
  }

  TopKMeta *m = new TopKMeta(sim->handler);
  m->sorted = sorted;

  // allocate
  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  int *output_ind_ptr =
      (int *)sim->allocate(sub_output_ind.get_volume(), DT_INT32);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  if (!(input_ptr && output_ptr && output_ind_ptr)) {
    cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    return true;
  }

  assert(m->profiling == false);

  // compute
  std::function<void()> forward, backward;

  Domain in_domain = sub_input.get_domain();
  int length = in_domain.hi()[0] - in_domain.lo()[0] + 1;
  size_t batch_size = in_domain.get_volume() / length;

  forward = [&] {
    forward_kernel_wrapper(m,
                           input_ptr,
                           output_ptr,
                           output_ind_ptr,
                           batch_size,
                           length,
                           k,
                           sorted);
  };

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);
  log_measure.debug("[Measure TopK] name(%s) forward_time(%.4lf)\n",
                    name,
                    cost_metrics.forward_time);

  cost_metrics.backward_time = 0.0f; // not implemented for MOE
  delete m;
  return true;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::TopKParams>::operator()(
    FlexFlow::TopKParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.k);
  hash_combine(key, params.sorted);
  return key;
}
}; // namespace std

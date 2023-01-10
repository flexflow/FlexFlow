/* Copyright 2022 CMU
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

#include "flexflow/ops/experts.h"
#include "legion/legion_utilities.h"

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

bool operator==(ExpertsParams const &lhs, ExpertsParams const &rhs) {
  return lhs.num_experts == rhs.num_experts &&
         lhs.experts_start_idx == rhs.experts_start_idx &&
         lhs.experts_num_layers == rhs.experts_num_layers &&
         lhs.experts_output_dim_size == rhs.experts_output_dim_size &&
         lhs.experts_internal_dim_size == rhs.experts_internal_dim_size;
}

bool ExpertsParams::is_valid(
    std::pair<ParallelTensorShape, ParallelTensorShape> const &input) const {
  if (!input.first.is_valid())
    return false;
  if (!input.second.is_valid())
    return false;
  if (input.first.num_dims != input.second.num_dims + 1)
    return false;
  if (input.second.data_type != DT_INT32 && input.second.data_type != DT_INT64)
    return false;
  for (int i = 0; i < input.second.num_dims; i++)
    if (input.second.dims[i] != input.first.dims[i + 1])
      return false;
  return true;
}

ExpertsParams Experts::get_params() const {
  ExpertsParams params;
  params.num_experts = num_experts;
  params.experts_start_idx = experts_start_idx;
  params.experts_num_layers = experts_num_layers;
  params.experts_output_dim_size = experts_output_dim_size;
  params.experts_internal_dim_size = experts_internal_dim_size;
  return params;
}

Tensor FFModel::experts(const Tensor input,
                        const Tensor indices,
                        int num_experts,
                        int experts_start_idx,
                        int experts_num_layers,
                        int experts_output_dim_size,
                        int experts_internal_dim_size,
                        char const *name) {
  Layer *e = new Layer(this,
                       OP_EXPERTS,
                       DT_FLOAT,
                       name,
                       2 /*inputs*/,
                       1 /*weights*/,
                       1 /*outputs*/,
                       input,
                       indices);
  assert(input->num_dims == indices->num_dims + 1);
  for (int i = 0; i < indices->num_dims; i++)
    assert(input->dims[i + 1] == indices->dims[i]);
  assert(indices->data_type == DT_INT32 || indices->data_type == DT_INT64);
  int dims[MAX_TENSOR_DIM];
  int numdim = input->num_dims;
  for (int i = 1; i < input->num_dims; i++)
    dims[i] = input->dims[i];
  dims[0] = experts_output_dim_size;
  e->outputs[0] = create_tensor_legion_ordering(
      numdim, dims, input->data_type, e, 0, true /*create_grad*/);
  e->add_int_property("num_experts", num_experts);
  e->add_int_property("experts_start_idx", experts_start_idx);
  e->add_int_property("experts_num_layers", experts_num_layers);
  e->add_int_property("experts_output_dim_size", experts_output_dim_size);
  e->add_int_property("experts_internal_dim_size", experts_internal_dim_size);
  layers.push_back(e);
  return e->outputs[0];
}

Op *Experts::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("num_experts", value);
  int num_experts = value;
  layer->get_int_property("experts_start_idx", value);
  int experts_start_idx = value;
  layer->get_int_property("experts_num_layers", value);
  int experts_num_layers = value;
  layer->get_int_property("experts_output_dim_size", value);
  int experts_output_dim_size = value;
  layer->get_int_property("experts_internal_dim_size", value);
  int experts_internal_dim_size = value;
  return new Experts(model,
                     inputs[0],
                     inputs[1],
                     num_experts,
                     experts_start_idx,
                     experts_num_layers,
                     experts_output_dim_size,
                     experts_internal_dim_size,
                     layer->name);
}

Experts::Experts(FFModel &model,
                 ExpertsParams const &params,
                 std::pair<ParallelTensor, ParallelTensor> const &inputs,
                 char const *name)
    : Experts(model,
              inputs.first,
              inputs.second,
              params.num_experts,
              params.experts_start_idx,
              params.experts_num_layers,
              params.experts_output_dim_size,
              params.experts_internal_dim_size,
              name) {}

Experts::Experts(FFModel &model,
                 const ParallelTensor input,
                 const ParallelTensor indices,
                 int _num_experts,
                 int _experts_start_idx,
                 int _experts_num_layers,
                 int _experts_output_dim_size,
                 int _experts_internal_dim_size,
                 char const *name)
    : Op(model,
         OP_EXPERTS,
         DT_FLOAT,
         name,
         2 /*inputs*/,
         1 /*weights*/,
         1 /*outputs*/,
         input,
         indices),
      num_experts(_num_experts), experts_start_idx(_experts_start_idx),
      experts_num_layers(_experts_num_layers),
      experts_output_dim_size(_experts_output_dim_size),
      experts_internal_dim_size(_experts_internal_dim_size) {
  assert(input->num_dims == indices->num_dims + 1);
  assert(indices->data_type == DT_INT32 || indices->data_type == DT_INT64);
  for (int i = 0; i < indices->num_dims; i++)
    assert(input->dims[i + 1] == indices->dims[i]);
  // Assume that we don't parallelize the channel dim
  assert(input->dims[0].degree == 1);
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < input->num_dims; i++)
    dims[i] = input->dims[i];
  dims[0].size = experts_output_dim_size;
  numOutputs = 1;
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      input->num_dims, dims, input->data_type, this);
}

void Experts::serialize(Legion::Serializer &sez) const {
  ExpertsParams params = get_params();
  sez.serialize(params.num_experts);
  sez.serialize(params.experts_start_idx);
  sez.serialize(params.experts_num_layers);
  sez.serialize(params.experts_output_dim_size);
  sez.serialize(params.experts_internal_dim_size);
}

using PCG::Node;
Node Experts::deserialize(FFModel &ff,
                          Legion::Deserializer &dez,
                          ParallelTensor inputs[],
                          int num_inputs) {
  assert(num_inputs == 2);
  int num_experts, experts_start_idx, experts_num_layers,
      experts_output_dim_size, experts_internal_dim_size;
  dez.deserialize(num_experts);
  dez.deserialize(experts_start_idx);
  dez.deserialize(experts_num_layers);
  dez.deserialize(experts_output_dim_size);
  dez.deserialize(experts_internal_dim_size);

  ExpertsParams params;
  params.num_experts = num_experts;
  params.experts_start_idx = experts_start_idx;
  params.experts_num_layers = experts_num_layers;
  params.experts_output_dim_size = experts_output_dim_size;
  params.experts_internal_dim_size = experts_internal_dim_size;
  return ff.get_or_create_node<Experts>({inputs[0], inputs[1]}, params);
}

Op *Experts::materialize(FFModel &ff,
                         ParallelTensor inputs[],
                         int num_inputs) const {
  ExpertsParams params = get_params();
  return new Experts(ff, params, {inputs[0], inputs[1]}, this->name);
}

void Experts::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(EXPERTS_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Experts)),
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
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *Experts::init_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  Experts const *bmm = (Experts *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  ExpertsMeta *m = new ExpertsMeta(handle);
  return m;
}

void Experts::forward(FFModel const &ff) {
  assert(false && "Experts is designed for inference only");
}

void Experts::forward_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  assert(false && "Experts is designed for inference only");
}

void Experts::backward(FFModel const &ff) {
  assert(false && "Experts is designed for inference only");
}

void Experts::backward_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  assert(false && "Experts is designed for inference only");
}

FutureMap Experts::inference(FFModel const &ff,
                             std::vector<ParallelTensor> const &batch_inputs,
                             std::vector<ParallelTensor> const &batch_outputs,
                             MachineView const *mv) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  size_t machine_view_hash = mv ? mv->hash() : outputs[0]->machine_view.hash();
  IndexLauncher launcher(EXPERTS_INF_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
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
  launcher.add_region_requirement(RegionRequirement(batch_inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(2, FID_DATA);
  return runtime->execute_index_space(ctx, launcher);
}

void Experts::inference_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  // TODO: to be implemented
}

void Experts::print_layer(FFModel const &ff) {
  return;
}

bool Experts::measure_operator_cost(Simulator *sim,
                                    MachineView const &c,
                                    CostMetrics &cost_metrics) const {
  // This is an inference only operator
  assert(false);
  return false;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ExpertsParams>::operator()(
    FlexFlow::ExpertsParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.num_experts);
  hash_combine(key, params.experts_start_idx);
  hash_combine(key, params.experts_num_layers);
  hash_combine(key, params.experts_output_dim_size);
  hash_combine(key, params.experts_internal_dim_size);
  return key;
}
}; // namespace std

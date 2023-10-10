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

#include "flexflow/ops/rms_norm.h"
#include "flexflow/model.h"
#include "flexflow/ops/kernels/rms_norm_kernels.h"
#include "flexflow/utils/hash_utils.h"
#include "legion/legion_utilities.h"

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
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

using namespace FlexFlow::Kernels::RMSNorm;

bool operator==(RMSNormParams const &lhs, RMSNormParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid && lhs.eps == rhs.eps;
}

bool RMSNormParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

RMSNormParams RMSNorm::get_params() const {
  RMSNormParams params;
  params.layer_guid = this->layer_guid;
  params.eps = this->eps;
  params.dim = this->dim;
  return params;
}

Tensor FFModel::rms_norm(const Tensor input,
                         float eps,
                         int dim,
                         DataType data_type,
                         char const *name) {
  if (data_type == DT_NONE) {
    data_type = input->data_type;
  }
  Layer *rm = nullptr;
  if (data_type != input->data_type) {
    Tensor casted_input = cast(input, data_type, "type cast for rms_norm");
    rm = new Layer(this,
                   OP_RMS_NORM,
                   data_type,
                   name,
                   1 /*inputs*/,
                   1 /*weights*/,
                   1 /*outputs*/,
                   casted_input);
  } else {
    rm = new Layer(this,
                   OP_RMS_NORM,
                   data_type,
                   name,
                   1 /*inputs*/,
                   1 /*weights*/,
                   1 /*outputs*/,
                   input);
  }
  rm->outputs[0] = create_tensor_legion_ordering(
      input->num_dims, input->dims, data_type, rm, 0, true /*create_grad*/);

  // weights
  int weight_dims[1] = {dim};
  rm->weights[0] = create_weight_legion_ordering(1,
                                                 weight_dims,
                                                 data_type,
                                                 rm,
                                                 true /*create_grad*/,
                                                 nullptr,
                                                 CHOSEN_SYNC_TYPE);

  rm->add_float_property("eps", eps);
  rm->add_int_property("dim", dim);
  layers.push_back(rm);
  return rm->outputs[0];
}

Op *RMSNorm::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  float eps;
  layer->get_float_property("eps", eps);
  long long value;
  layer->get_int_property("dim", value);
  int dim = value;

  return new RMSNorm(
      model, layer->layer_guid, inputs[0], eps, dim, false, layer->name);
}

RMSNorm::RMSNorm(FFModel &model,
                 RMSNormParams const &params,
                 ParallelTensor const input,
                 bool allocate_weights = false,
                 char const *name)
    : RMSNorm(model,
              params.layer_guid,
              input,
              params.eps,
              params.dim,
              allocate_weights,
              name) {}

RMSNorm::RMSNorm(FFModel &model,
                 RMSNorm const &other,
                 const ParallelTensor input,
                 bool allocate_weights)
    : RMSNorm(model,
              other.layer_guid,
              input,
              other.eps,
              other.dim,
              allocate_weights,
              other.name) {}
RMSNorm::RMSNorm(FFModel &model,
                 LayerID const &_layer_guid,
                 const ParallelTensor _input,
                 float _eps,
                 int dim,
                 bool allocate_weights,
                 char const *name)
    : Op(model,
         OP_RMS_NORM,
         _input->data_type,
         name,
         1 /*num of inputs tensor */,
         1 /*num of weights tensor */,
         1 /*onum of utputs tensor */,
         _input) {
  eps = _eps;
  inputs[0] = _input;
  layer_guid = _layer_guid;
  int num_dims = _input->num_dims;
  this->dim = dim;
  data_dim = _input->dims[0].size;
  effective_batch_size = 1;
  for (int i = 1; i <= num_dims - 2; i++) {
    effective_batch_size *= _input->dims[i].size;
  }
  // Currently assert that all non-replica dims are not parallelized
  // We only support parallelism along the replica dim now
  for (int i = 0; i < _input->num_dims - 1; i++) {
    assert(_input->dims[i].degree == 1);
  }
  // output has the same parallel dims as input
  ParallelDim output_dims[MAX_TENSOR_DIM];
  for (int i = 0; i < _input->num_dims; i++) {
    output_dims[i] = _input->dims[i];
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      _input->num_dims, output_dims, _input->data_type, this);
  if (allocate_weights) {
    // weights should have the shape of (data_dim, data_dim)
    ParallelDim new_weight_dims[MAX_TENSOR_DIM];

    new_weight_dims[0].size = dim;
    new_weight_dims[0].degree = 1;
    new_weight_dims[0].parallel_idx = -1;
    new_weight_dims[1] = _input->dims[_input->num_dims - 1]; // replica dim

    // weights
    Initializer *kernel_initializer = new GlorotUniform(std::rand() /*seed*/);
    weights[0] =
        model.create_parallel_weight_legion_ordering(2,
                                                     new_weight_dims,
                                                     _input->data_type,
                                                     nullptr /*owner_op*/,
                                                     false /*create_grad*/,
                                                     kernel_initializer,
                                                     CHOSEN_SYNC_TYPE);
  }
}

void RMSNorm::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(RMSNORM_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(RMSNorm)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(1, FID_DATA);

  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void RMSNorm::init_inference(FFModel const &ff,
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

  IndexLauncher launcher(RMSNORM_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(RMSNorm)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(1, FID_DATA);

  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

OpMeta *RMSNorm::init_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  RMSNorm *rn = (RMSNorm *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .only_kind(Memory::GPU_FB_MEM)
                       .best_affinity_to(task->target_proc)
                       .first();
  MemoryAllocator gpu_mem_allocator(gpu_mem);
  RMSNormMeta *meta = new RMSNormMeta(handle, rn, gpu_mem_allocator);
  std::strcpy(meta->op_name, rn->name);
  meta->layer_guid = rn->layer_guid;
  return meta;
}

void RMSNorm::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(RMSNORM_FWD_TASK_ID,
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
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

FutureMap RMSNorm::inference(FFModel const &ff,
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

  IndexLauncher launcher(RMSNORM_INF_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
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
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(2, FID_DATA);
  return runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input
  regions[1](O): output
  regions[2](I/O): weight
*/
void RMSNorm::forward_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  assert(task->regions.size() == 3);
  assert(regions.size() == 3);
  RMSNormMeta const *m = *((RMSNormMeta **)task->local_args);
  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorR weight = helperGetGenericTensorAccessorRO(
      m->weight_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  forward_kernel_wrapper(m, input, weight, output);
}

/*
  regions[0](I): input
  regions[1](O): output
  regions[2](I/O): weight
*/
void RMSNorm::inference_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  assert(task->regions.size() == 3);
  assert(regions.size() == 3);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_tokens == 0) {
    return;
  }
  RMSNormMeta *m = *((RMSNormMeta **)task->local_args);
  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorR weight = helperGetGenericTensorAccessorRO(
      m->weight_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  forward_kernel_wrapper(m, input, weight, output);
  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    RMSNorm::save_inference_tensors_to_file(
        m, shard_id, bc, {input}, {weight}, {output});
  }
}

void RMSNorm::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->layer_guid.model_id);
  sez.serialize(this->eps);
  sez.serialize(this->dim);
}

using PCG::Node;
/*static*/
Node RMSNorm::deserialize(FFModel &ff,
                          Legion::Deserializer &dez,
                          ParallelTensor inputs[],
                          int num_inputs) {
  assert(num_inputs == 1);
  float eps;
  size_t id, transformer_layer_id, deserialized_model_id;
  int dim;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  dez.deserialize(deserialized_model_id);

  LayerID layer_guid(id, transformer_layer_id, deserialized_model_id);
  dez.deserialize(eps);
  dez.deserialize(dim);
  RMSNormParams params;
  params.layer_guid = layer_guid;
  params.eps = eps;
  params.dim = dim;
  return ff.get_or_create_node<RMSNorm>(inputs[0], params);
}

Op *RMSNorm::materialize(FFModel &ff,
                         ParallelTensor inputs[],
                         int num_inputs) const {
  RMSNormParams params = get_params();
  return new RMSNorm(ff, params, inputs[0], true, this->name);
}

void RMSNorm::backward(FFModel const &ff) {}

bool RMSNorm::measure_operator_cost(Simulator *sim,
                                    MachineView const &mv,
                                    CostMetrics &cost_metrics) const {
  return false;
}

} // namespace FlexFlow
namespace std {
size_t hash<FlexFlow::RMSNormParams>::operator()(
    FlexFlow::RMSNormParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.eps);
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.dim);
  return key;
}
}; // namespace std

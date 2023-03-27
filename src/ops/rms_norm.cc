* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
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

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::Predicate;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::TaskArgument;
using Legion::TaskLauncher;

bool operator==(RMSNormParams const &lhs, RMSNormParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid && lhs.eps = rhs.eps;
}

bool RMSNormParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

RMSNormParams RMSNorm::get_params() const {
  RMSNormParams params;
  params.layer_guid = this->layer_guid;
  params.eps = this->eps;
  return params;
}

Tensor FFModel::rms_norm(const Tensor input, float eps, int dim, char const *name) {
  Layer *rm = new Layer(this,
                        OP_RMS_NORM,
                        DT_FLOAT,
                        name,
                        1 /*inputs*/,
                        1 /*weights*/,
                        1 /*outputs*/,
                        input);
  rm->outputs[0] = create_tensor_legion_ordering(
      input->num_dims, input->dims, DT_FLOAT, rm, 0, true /*create_grad*/);

  // weights
  // TODO weight dims check
  rm->weights[0] = create_weight_legion_ordering(1,
                                                 dim,
                                                 DT_FLOAT,
                                                 rm,
                                                 true /*create_grad*/,
                                                 nullptr,
                                                 CHOSEN_SYNC_TYPE);
  ln->add_float_property("eps", eps);
  ln->add_float_property("dim", dim);
  layers.push_back(bm);
  return rm->outputs[0];
}

Op *RMSNorm::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  float eps;
  layer->get_int_property("eps", eps);
  return new RMSNorm(model, layer->layer_guid, inputs[0], eps, layer->name);
}

RMSNorm::RMSNorm(FFModel &model,
                 LayerID const &_layer_guid,
                 const ParallelTensor _input,
                 float _eps,
                 char const *name)
    : Op(model,
         OP_RMS_NORM,
         input->data_type,
         name,
         1 /*inputs*/,
         1 /*weights*/,
         1 /*outputs*/,
         input),
{

  // output has the same parallel dims as input
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < input->num_dims; i++) {
    dims[i] = input->dims[i];
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      input->num_dims, dims, input->data_type, this);
  // weights
  Initializer *kernel_initializer = new GlorotUniform(std::rand() /*seed*/);
  weights[0] =
      model.create_parallel_weight_legion_ordering(input->num_dims,
                                                   dims,
                                                   input->data_type,
                                                   this /*owner_op*/,
                                                   true /*create_grad*/,
                                                   kernel_initializer,
                                                   CHOSEN_SYNC_TYPE);
}

void RMSNorm::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(RMSNROM_INIT_TASK_ID,
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
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void RMSNorm::init_inference(
    FFModel const &ff,
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
  set_argumentmap_for_init_inference(ff, argmap, view);

  IndexLauncher launcher(RMSNROM_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(RMSNorm)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
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
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, view);
}

OpMeta *RMSNorm::init_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  RMSNorm *rn = (RMSNorm *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  RMSNormMeta *meta = new RMSNormMeta(handle, rn);
  return meta;
}

void RMSNorm::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(RMSNROM_FWD_TASK_ID,
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

void RMSNorm::inference(FFModel const &ff,
                             BatchConfig const &bc,
                             std::vector<ParallelTensor> const &batch_inputs,
                             std::vector<ParallelTensor> const &batch_outputs,
                             MachineView const *mv) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  parallel_is = batch_outputs[0]->parallel_is;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  set_argumentmap_for_inference(ff, argmap, view);
  size_t machine_view_hash = view->hash();

  IndexLauncher launcher(RMSNROM_FWD_TASK_ID,
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
  GenericTensorAccessorR weight = helperGetGenericTensorAccessorRO(
      m->input_type[1], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  forward_kernel_wrapper(m, input, weight, output);
}


void RMSNorm::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->eps);
}

using PCG::Node;
/*static*/
Node RMSNorm::deserialize(FFModel &ff,
                          Legion::Deserializer &dez,
                          ParallelTensor inputs[],
                          int num_inputs) {
  assert(num_inputs == 1);
  float eps;
  size_t id;
  dez.deserialize(id);
  LayerID layer_guid(id);
  dez.deserialize(eps);

  RMSParams params;
  params.layer_guid = layer_guid;
  params.eps = eps;
  return ff.get_or_create_node<RMSNorm>(inputs[0], params);
}

Op *RMSNorm::materialize(FFModel &ff,
                           ParallelTensor inputs[],
                           int num_inputs) const {
  RMSParams params = get_params();
  return new RMSNorm(
      ff, params, inputs[0], this->name, true /*allocate_weights*/);
}

} // namespace FlexFlow
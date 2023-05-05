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

#include "flexflow/ops/place_holder.h"
#include "flexflow/model.h"
#include "flexflow/utils/hash_utils.h"
#include "legion/legion_utilities.h"
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include "flexflow/utils/cuda_helper.h"
#else
#include "flexflow/utils/hip_helper.h"
#endif

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
Tensor FFModel::place_holder(const Tensor input, char const *name) {
  Layer *li = new Layer(this,
                        OP_PLACE_HOLDER,
                        input->data_type,
                        name,
                        1 /*inputs*/,
                        0 /*weights*/,
                        1 /*outputs*/,
                        input);
  {
    int numdims = 1;
    int dims[MAX_TENSOR_DIM];
    dims[0] = 1;

    li->outputs[0] = create_tensor_legion_ordering(
        numdims, dims, DT_INT32, li, 0, false /*create_grad*/);
  }
  layers.push_back(li);
  return li->outputs[0];
}

Op *PlaceHolder::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  return new PlaceHolder(model, inputs[0], layer->layer_guid, layer->name);
}

PlaceHolderParams PlaceHolder::get_params() const {
  PlaceHolderParams params;
  params.layer_guid = this->layer_guid;
  return params;
}

bool PlaceHolderParams::is_valid(ParallelTensorShape const &) const {
  return true;
}

bool operator==(PlaceHolderParams const &lhs, PlaceHolderParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid;
}

PlaceHolder::PlaceHolder(FFModel &model,
                         const ParallelTensor _input,
                         LayerID const &_layer_guid,
                         char const *name)
    : Op(model,
         OP_PLACE_HOLDER,
         _input->data_type,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         _input) {
  layer_guid = _layer_guid;
  int numdim = 1;
  assert(inputs[0]->dims[0].degree == 1);
  assert(inputs[0]->dims[0].parallel_idx == -1);
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, inputs[0]->dims, DT_INT32, this, 0 /*owner_idx*/);
}

PlaceHolder::PlaceHolder(FFModel &model,
                         PlaceHolder const &other,
                         const ParallelTensor input)
    : PlaceHolder(model, input, other.layer_guid, other.name) {}

PlaceHolder::PlaceHolder(FFModel &model,
                         PlaceHolderParams const &params,
                         const ParallelTensor input,
                         char const *name)
    : PlaceHolder(model, input, params.layer_guid, name) {}

void PlaceHolder::init_inference(
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
  set_argumentmap_for_init_inference(ff, argmap, batch_outputs[0]);
  IndexLauncher launcher(PLACE_HOLDER_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(PlaceHolder)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

void PlaceHolder::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(PLACE_HOLDER_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(PlaceHolder)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *PlaceHolder::init_task(Task const *task,
                               std::vector<PhysicalRegion> const &regions,
                               Context ctx,
                               Runtime *runtime) {
  FFHandler handle = *((FFHandler *)task->local_args);
  PlaceHolderMeta *m = new PlaceHolderMeta(handle);
  return m;
}

void PlaceHolder::forward(FFModel const &ff) {
  assert(false);
}

FutureMap
    PlaceHolder::inference(FFModel const &ff,
                           BatchConfig const &bc,
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

  IndexLauncher launcher(
      PLACE_HOLDER_INF_TASK_ID,
      parallel_is,
      TaskArgument(
          &bc, std::max(sizeof(BatchConfig), sizeof(BeamSearchBatchConfig))),
      argmap,
      Predicate::TRUE_PRED,
      false /*must*/,
      0 /*mapper_id*/,
      machine_view_hash);
  return runtime->execute_index_space(ctx, launcher);
}

void PlaceHolder::inference_task(Task const *task,
                                 std::vector<PhysicalRegion> const &regions,
                                 Context ctx,
                                 Runtime *runtime) {}

void PlaceHolder::backward(FFModel const &ff) {
  assert(false);
}

void PlaceHolder::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
}

Node PlaceHolder::deserialize(FFModel &ff,
                              Legion::Deserializer &dez,
                              ParallelTensor inputs[],
                              int num_inputs) {
  assert(num_inputs == 1);
  size_t id;
  dez.deserialize(id);
  LayerID layer_guid(id);
  PlaceHolderParams params;
  params.layer_guid = layer_guid;
  return ff.get_or_create_node<PlaceHolder>(inputs[0], params);
}

Op *PlaceHolder::materialize(FFModel &ff,
                             ParallelTensor inputs[],
                             int num_inputs) const {
  PlaceHolderParams params = get_params();
  return new PlaceHolder(ff, params, inputs[0], this->name);
}

bool PlaceHolder::measure_operator_cost(Simulator *sim,
                                        MachineView const &mv,
                                        CostMetrics &cost_metrics) const {
  return false;
}

PlaceHolderMeta::PlaceHolderMeta(FFHandler _handle) : OpMeta(_handle) {}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::PlaceHolderParams>::operator()(
    FlexFlow::PlaceHolderParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  return key;
}
}; // namespace std
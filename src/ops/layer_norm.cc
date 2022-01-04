/* Copyright 2021 CMU, Facebook
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

#include "flexflow/ops/layer_norm.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::PhysicalRegion;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;
using Legion::InlineLauncher;

Tensor FFModel::layer_norm(const Tensor input,
                           const std::vector<int>& axes,
                           bool elementwise_affine,
                           float eps,
                           const char* name)
{
  // axes must be the last axes.size() dimensions
  for (int i = 0; i < axes.size(); i++) {
    bool found = false;
    for (int j = 0; j < axes.size(); j++) 
      if (axes[j] == input->num_dims - 1 - i)
        found = true;
    if (!found) {
      assert(false && "axes must be the last axes.size() dimensions");
    }
  }
  int num_weights = elementwise_affine ? 2 : 0;
  Layer *ln = new Layer(this, OP_LAYERNORM, name, 1/*inputs*/, num_weights, 1/*outputs*/, input);
  ln->outputs[0] = create_tensor_legion_ordering(
      input->num_dims, input->dims, input->data_type, ln, 0, true/*create_grad*/);
  if (num_weights == 2) {
    int M = 1;
    for (int i = 0; i < axes.size(); i++)
      M *= input->dims[input->num_dims-1-axes[i]];
    int dims[1] = {M};
    ln->weights[0] = create_weight_legion_ordering(1, dims, input->data_type,
        ln, true/*create_grad*/, nullptr, CHOSEN_SYNC_TYPE);
    ln->weights[1] = create_weight_legion_ordering(1, dims, input->data_type,
        ln, true/*create_grad*/, nullptr, CHOSEN_SYNC_TYPE);
  }
  ln->add_int_property("elementwise_affine", elementwise_affine);
  ln->add_int_vector_property("axes", axes);
  ln->add_float_property("eps", eps);
  layers.push_back(ln);
  return ln->outputs[0];
}

Op* LayerNorm::create_operator_from_layer(
    FFModel& model,
    const Layer* layer,
    const std::vector<ParallelTensor>& inputs) {
  long long value;
  layer->get_int_property("elementwise_affine", value);
  bool elementwise_affine = (bool)value;
  std::vector<int> axes;
  layer->get_int_vector_property("axes", axes);
  float eps;
  layer->get_float_property("eps", eps);
  return new LayerNorm(
      model,
      layer->layer_guid,
      inputs[0],
      axes,
      elementwise_affine,
      eps,
      false, // allocate_weights
      layer->name);
}

LayerNorm::LayerNorm(FFModel& model,
                     const LayerID& _layer_guid,
                     const ParallelTensor _input,
                     const std::vector<int>& axes,
                     bool _elementwise_affine,
                     float _eps,
                     bool allocate_weights,
                     const char *name)
: Op(model, OP_LAYERNORM, name, 1/*inputs*/, _elementwise_affine ? 2 : 0/*weights*/,
     1/*outputs*/, _input),
  elementwise_affine(_elementwise_affine),
  eps(_eps)
{
  // overwrite layer_guid
  layer_guid = _layer_guid;
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      _input->num_dims, _input->dims, _input->data_type, this);
  assert(check_output_input_weight_parallel_dims(allocate_weights));
  ParallelDim output_dims[MAX_TENSOR_DIM];
  int M = 1;
  for (int i = 0; i < axes.size(); i++)
    M *= inputs[0]->dims[inputs[0]->num_dims-1-axes[i]].size;
  effective_num_elements = M;
  effective_batch_size = inputs[0]->get_volume() / M;
  if (numWeights > 0 && allocate_weights) {
    int kernel_dims = 2;
    assert(false);
    //weights[0] = model.create_parallel_weight_legion_ordering(
    //    kernel_dims, 
  } else {
    // do nothing
  }
  return;
}

void LayerNorm::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(LAYERNORM_INIT_TASK_ID, parallel_is,
      TaskArgument(this, sizeof(LayerNorm)), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void LayerNorm::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(
      LAYERNORM_FWD_TASK_ID, parallel_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  if (elementwise_affine) {
    launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, weights[0]->region));
    launcher.add_field(2, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(weights[1]->part, 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, weights[1]->region));
    launcher.add_field(3, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void LayerNorm::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(
      LAYERNORM_BWD_TASK_ID, parallel_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
  // regions[0](I): output_grad
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  // regions[1](I): input
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(1, FID_DATA);
  // regions[2](I/O): input_grad
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(2, FID_DATA);
  if (elementwise_affine) {
    // regions[3](I): gamma
    launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, weights[0]->region));
    launcher.add_field(3, FID_DATA);
    // regions[4](I/O): gamma_grad
    launcher.add_region_requirement(
      RegionRequirement(weights[0]->part_grad, 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, weights[0]->region_grad));
    launcher.add_field(4, FID_DATA);
    // regions[5](I/O): beta_grad
    launcher.add_region_requirement(
      RegionRequirement(weights[1]->part_grad, 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, weights[1]->region_grad));
    launcher.add_field(5, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

}; // namespace FlexFlow

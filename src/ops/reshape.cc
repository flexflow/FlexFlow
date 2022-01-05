/* Copyright 2022 CMU, Stanford, Facebook
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

#include "flexflow/ops/reshape.h"
#include "flexflow/utils/hash_utils.h"
#include "legion/legion_utilities.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::PhysicalRegion;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;
ReshapeParams::ReshapeParams(const std::vector<int>& _shape)
: shape(_shape) {}

Tensor FFModel::reshape(const Tensor input,
                        const std::vector<int>& shape,
                        const char* name)
{
  Layer* reshape = new Layer(this, OP_RESHAPE, name, 1/*inputs*/,
                             0/*weights*/, 1/*outputs*/, input);
  int dims[MAX_TENSOR_DIM];
  int numdim = shape.size();
  for (int i = 1; i < numdim; i++) {
    dims[i] = shape[i];
  }
  reshape->outputs[0] = create_tensor(
      numdim, dims, input->data_type, reshape, 0, true/*create_grad*/);
  reshape->add_int_vector_property("shape", shape);
  layers.push_back(reshape);
  return reshape->outputs[0];
}

Op* Reshape::create_operator_from_layer(
    FFModel& model,
    const Layer* layer,
    const std::vector<ParallelTensor>& inputs)
{
  std::vector<int> shape;
  layer->get_int_vector_property("shape", shape);
  return new Reshape(model, inputs[0], shape, layer->name);
}

Reshape::Reshape(FFModel& model,
                 const ParallelTensor input,
                 const std::vector<int>& _shape,
                 const char* name)
: Op(model, OP_RESHAPE, name, 1/*inputs*/, 0/*weights*/, 1/*outputs*/, input)
{
  shape_length = _shape.size();
  assert(shape_length <= MAX_TENSOR_DIM);
  for (int i = 0; i < shape_length; i++)
    shape_array[i] = _shape[i];
  numOutputs = 1;
  numWeights = 0;
  int num_replica_dims = 0;
  for (int i = 0; i < input->num_dims; i++)
    if (input->dims[i].is_replica_dim)
      num_replica_dims ++;
  // assert that all replica dims are leading dims
  for (int i = 0; i < num_replica_dims; i++)
    assert(input->dims[input->num_dims-1-i].is_replica_dim);
  int numdim = (int) _shape.size();
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i].size = _shape[numdim-1-i];
    dims[i].degree = 1;
    dims[i].parallel_idx = -1;
    dims[i].is_replica_dim = false;
  }
  // copy all replica dims
  for (int i = 0; i < num_replica_dims; i++)
    dims[i+numdim] = input->dims[input->num_dims-1-i];
  numdim += num_replica_dims;
  for (int i = num_replica_dims; i < numdim && i < input->num_dims; i++) {
    if (dims[numdim-1-i].size != input->dims[input->num_dims-1-i].size)
      break;
    dims[numdim-1-i] = input->dims[input->num_dims-1-i];
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      numdim, dims, input->data_type, this);
  assert(outputs[0]->get_volume() == inputs[0]->get_volume());
}

void Reshape::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(RESHAPE_INIT_TASK_ID, parallel_is,
      TaskArgument(this, sizeof(Reshape)), argmap,
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
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void Reshape::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(RESHAPE_FWD_TASK_ID, parallel_is,
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
  runtime->execute_index_space(ctx, launcher);
}

void Reshape::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(RESHAPE_BWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0](I): output_grad
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  // regions[3](I/O): input0_grad
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
                      READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

ReshapeParams Reshape::get_params() const {
  std::vector<int> shape_vec;
  for (size_t i = 0; i < shape_length; i++)
    shape_vec.push_back(shape_array[i]);
  ReshapeParams params(shape_vec);
  return params;
}

size_t ReshapeParams::get_hash(const ParallelTensor input) const {
  size_t hash = input->get_owner_independent_hash();
  hash_combine(hash, shape.size());
  for (size_t i = 0; i < shape.size(); i++)
    hash_combine(hash, shape[i]);
  return hash;
}

size_t Reshape::get_params_hash() const {
  return this->get_params().get_hash(this->inputs[0]);
}

void Reshape::serialize(Legion::Serializer & sez) const {
  sez.serialize(this->shape_length);
  for (size_t i = 0; i < this->shape_length; i++)
    sez.serialize(this->shape_array[i]);
}

using PCG::Node;

Node Reshape::deserialize(FFModel& ff,
                          Legion::Deserializer& dez,
                          ParallelTensor inputs[],
                          int num_inputs) {
  assert(num_inputs == 1);
  size_t shape_length;
  std::vector<int> shape;
  dez.deserialize(shape_length);
  for (size_t i = 0; i < shape_length; i++) {
    int value;
    dez.deserialize(value);
    shape.push_back(value);
  }
  return ff.get_or_create_reshape_node(
      inputs[0],
      shape);
}


Op* Reshape::materialize(FFModel& ff, ParallelTensor inputs[], int num_inputs) const {
  assert(num_inputs == 1);
  std::vector<int> shape;
  for (size_t i = 0; i < this->shape_length; i++)
    shape.push_back(shape_array[i]);
  return new Reshape(ff, inputs[0], shape, this->name);
}

Node FFModel::get_or_create_reshape_node(const ParallelTensor input,
                                         const ReshapeParams& params) {
  size_t hash = params.get_hash(input);
  Reshape *reshape = nullptr;

  const auto& it = this->cached_reshape_ops.find(hash);
  if (it != cached_reshape_ops.end()) {
    reshape = it->second;
  } else {
    reshape = new Reshape(*this, input, params.shape, nullptr);
    cached_reshape_ops[hash] = reshape;
  }

  return this->new_node(reshape);
}

Node FFModel::get_or_create_reshape_node(const ParallelTensor input,
                                         const std::vector<int>& shape) {
  ReshapeParams params(shape);
  return this->get_or_create_reshape_node(input, params);
}

}; // namespace FlexFlow

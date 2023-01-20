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

static constexpr int KERNEL_IDX = 0;
static constexpr int BIAS_IDX = 1;

bool operator==(ExpertsParams const &lhs, ExpertsParams const &rhs) {
  return lhs.num_experts == rhs.num_experts &&
         lhs.experts_start_idx == rhs.experts_start_idx &&
         lhs.experts_num_layers == rhs.experts_num_layers &&
         lhs.experts_output_dim_size == rhs.experts_output_dim_size &&
         lhs.experts_internal_dim_size == rhs.experts_internal_dim_size &&
         lhs.use_bias == rhs.use_bias &&
         lhs.activation == rhs.activation;
}

bool ExpertsParams::is_valid(std::vector<ParallelTensorShape> const &inputs) const {
  for (std::vector<ParallelTensorShape>::iterator it = inputs.begin() ; it != inputs.end(); ++it) {
    if (! it->is_valid()) {
      printf("the is_valid() method from one of the inputs returned false\n");
      return false;
    }
    if (it->num_dims != inputs[0].num_dims) {
      printf("inputs numb_dims mismatch\n");
      return false;
    }
    for (int i=0; i < it->num_dims; i++) {
      if (it->dims[i] != inputs[0].dims[i]) {
        printf("dimension %i is %i != %i\n", i, it->dims[i], inputs[0].dims[i]);
        return false;
      }
    }
  }
  // if (!input.first.is_valid()) {
  //   return false;
  // }
  // if (!input.second.is_valid()) {
  //   return false;
  // }
  // if (input.first.num_dims != input.second.num_dims) {
  //   return false;
  // }
  // if (input.second.data_type != DT_INT32 &&
  //     input.second.data_type != DT_INT64) {
  //   return false;
  // }
  // for (int i = 1; i < input.second.num_dims; i++) {
  //   if (input.second.dims[i] != input.first.dims[i]) {
  //     return false;
  //   }
  // }
  return true;
}

ExpertsParams Experts::get_params() const {
  ExpertsParams params;
  params.num_experts = num_experts;
  params.experts_start_idx = experts_start_idx;
  params.experts_num_layers = experts_num_layers;
  params.experts_output_dim_size = experts_output_dim_size;
  params.experts_internal_dim_size = experts_internal_dim_size;
  params.use_bias = use_bias;
  params.activation = activation;
  return params;
}

// For now, we use one input and one output per expert
void FFModel::experts(Tensor const *inputs,
                        Tensor *outputs,
                        int num_experts,
                        int experts_start_idx,
                        int experts_num_layers,
                        int experts_output_dim_size,
                        int experts_internal_dim_size,
                        // linear layer params
                        ActiMode activation,
                        bool use_bias,
                        DataType data_type,
                        Layer const *shared_op,
                        Initializer *kernel_initializer,
                        Initializer *bias_initializer,
                        // end of linear layer params
                        char const *name) {
  Layer *e = new Layer(this,
                       OP_EXPERTS,
                       data_type,
                       name,
                       num_experts /*inputs*/,
                       num_experts*(use_bias ? 2 : 1) /*weights*/,
                       num_experts /*outputs*/,
                       inputs);
  for (int i=0; i<num_experts; i++) {
    assert(inputs[i] != nullptr);
    assert(inputs[i]->num_dims == inputs[0]->num_dims);
    assert(inputs[i]->data_type == data_type);
    for (int j=0; j<inputs[i]->num_dims; j++) {
      assert(inputs[i]->dims[j] == inputs[0]->dims[j]);
    }
  }
  //assert(input->num_dims == indices->num_dims);
  //for (int i = 1; i < indices->num_dims; i++) {
  //  assert(input->dims[i] == indices->dims[i]);
  //}
  //assert(indices->data_type == DT_INT32 || indices->data_type == DT_INT64);
  {
    int dims[MAX_TENSOR_DIM];
    int numdim = inputs[0]->num_dims;
    for (int i = 1; i < numdim; i++) {
      dims[i] = inputs[0]->dims[i];
    }
    dims[0] = experts_output_dim_size;
    for (int i=0; i < num_experts; i++) {
      e->outputs[i] = create_tensor_legion_ordering(numdim, dims, data_type, e, 0, true /*create_grad*/);
      assert(e->outputs[i] != nullptr);
      outputs[i] = e->outputs[i];
      assert(outputs[i] != nullptr);
    }
  }
  {
    int dims[2] = {inputs[0]->dims[0], experts_output_dim_size};
    int weights_per_expert = use_bias ? 2 : 1;
    for (int i=0; i<num_experts; i++) {
      e->weights[weights_per_expert * i + KERNEL_IDX] =
        create_weight_legion_ordering(2,
                                      dims,
                                      data_type,
                                      e,
                                      true /*create_grad*/,
                                      kernel_initializer,
                                      CHOSEN_SYNC_TYPE);
    }
  }
  if (use_bias) {
    assert(weights_per_expert == 2);
    int dims[1] = {experts_output_dim_size};
    e->weights[weights_per_expert * i + BIAS_IDX] = create_weight_legion_ordering(1,
                                                          dims,
                                                          data_type,
                                                          e,
                                                          true /*create_grad*/,
                                                          bias_initializer,
                                                          CHOSEN_SYNC_TYPE);
  }

  e->add_int_property("num_experts", num_experts);
  e->add_int_property("experts_start_idx", experts_start_idx);
  e->add_int_property("experts_num_layers", experts_num_layers);
  e->add_int_property("experts_output_dim_size", experts_output_dim_size);
  e->add_int_property("experts_internal_dim_size", experts_internal_dim_size);
  e->add_int_property("use_bias", use_bias);
  e->add_int_property("activation", activation);
  layers.push_back(e);
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
  layer->get_int_property("use_bias", value);
  bool use_bias = (bool)value;
  layer->get_int_property("activation", value);
  ActiMode activation = (ActiMode)value;
  return new Experts(model,
                     inputs.data(),
                     num_experts,
                     experts_start_idx,
                     experts_num_layers,
                     experts_output_dim_size,
                     experts_internal_dim_size,
                     use_bias,
                     activation,
                     e->data_type,
                     false /*allocate_weights*/,
                     layer->name);
}

Experts::Experts(FFModel &model,
                 ExpertsParams const &params,
                 //std::pair<ParallelTensor, ParallelTensor> const &inputs,
                 std::vector<ParallelTensor> const &inputs,
                 char const *name,
                 bool allocate_weights)
    : Experts(model,
              inputs.data(),
              params.num_experts,
              params.experts_start_idx,
              params.experts_num_layers,
              params.experts_output_dim_size,
              params.experts_internal_dim_size,
              params.use_bias,
              params.activation,
              params.data_type,
              allocate_weights,
              name) {}

Experts::Experts(FFModel &model,
                 ParallelTensor const *inputs,
                 int _num_experts,
                 int _experts_start_idx,
                 int _experts_num_layers,
                 int _experts_output_dim_size,
                 int _experts_internal_dim_size,
                 bool _use_bias,
                 ActiMode _activation,
                 DataType _data_type,
                 bool allocate_weights,
                 char const *name)
    : Op(model,
         OP_EXPERTS,
         _data_type,
         name,
         _num_experts /*inputs*/,
         _num_experts*(_use_bias ? 2 : 1) /*weights*/,
         allocate_weights,
         _num_experts /*outputs*/,
         inputs),
      num_experts(_num_experts), experts_start_idx(_experts_start_idx),
      experts_num_layers(_experts_num_layers),
      experts_output_dim_size(_experts_output_dim_size),
      experts_internal_dim_size(_experts_internal_dim_size),
      use_bias(_use_bias),
      activation(_activation) {
  
  assert(num_experts > 0);
  assert(numInputs == num_experts);
  assert(numOutputs == num_experts);
  
  int num_dim = inputs[0]->num_dims;
  int out_dim = experts_output_dim_size;
  for (int i = 0; i < num_experts; i++) {
    assert(inputs[i]->num_dims == num_dim);
    for (int j=0; j<num_dim; j++) {
      assert(inputs[i]->dims[j] == inputs[0]->dims[j]);
    }
    assert(inputs[i]->dims[0].size == out_dim);
  }

  // assert(input->num_dims == indices->num_dims);
  // assert(indices->data_type == DT_INT32 || indices->data_type == DT_INT64);
  // for (int i = 1; i < indices->num_dims; i++) {
  //   assert(input->dims[i] == indices->dims[i]);
  // }

  // Assume that we don't parallelize the channel dim of input
  // nor the expert_assigned dim of indices
  // assert(input->dims[0].degree == 1);
  // assert(indices->dims[0].degree == 1);
  
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < num_dim; i++) {
    dims[i] = inputs[0]->dims[i];
  }
  dims[0].size = experts_output_dim_size;
  // numOutputs = num_experts;
  // numWeights = 0;
  for (int i=0; i < num_experts; i++) {
    outputs[i] = model.create_parallel_tensor_legion_ordering(
      num_dim, dims, inputs[0]->data_type, this);
    assert(outputs[i] != nullptr);
  }
}

void Experts::serialize(Legion::Serializer &sez) const {
  ExpertsParams params = get_params();
  sez.serialize(params.num_experts);
  sez.serialize(params.experts_start_idx);
  sez.serialize(params.experts_num_layers);
  sez.serialize(params.experts_output_dim_size);
  sez.serialize(params.experts_internal_dim_size);
  sez.serialize(params.use_bias);
  sez.serialize(params.activation);
}

using PCG::Node;
Node Experts::deserialize(FFModel &ff,
                          Legion::Deserializer &dez,
                          ParallelTensor inputs[],
                          int num_inputs) {
  int num_experts, experts_start_idx, experts_num_layers,
      experts_output_dim_size, experts_internal_dim_size;
  bool use_bias;
  ActiMode activation;
  dez.deserialize(num_experts);
  dez.deserialize(experts_start_idx);
  dez.deserialize(experts_num_layers);
  dez.deserialize(experts_output_dim_size);
  dez.deserialize(experts_internal_dim_size);
  dez.deserialize(use_bias);
  dez.deserialize(activation);

  assert(num_inputs == num_experts);

  ExpertsParams params;
  params.num_experts = num_experts;
  params.experts_start_idx = experts_start_idx;
  params.experts_num_layers = experts_num_layers;
  params.experts_output_dim_size = experts_output_dim_size;
  params.experts_internal_dim_size = experts_internal_dim_size;
  params.use_bias = use_bias;
  params.activation = activation;
  return ff.get_or_create_node<Experts>({std::begin(inputs), std::begin(inputs) + num_inputs}, params);
}

Op *Experts::materialize(FFModel &ff,
                         ParallelTensor inputs[],
                         int num_inputs) const {
  ExpertsParams params = get_params();
  return new Experts(ff, params, inputs, this->name);
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

void Experts::inference(FFModel const &ff,
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
  runtime->execute_index_space(ctx, launcher);
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
  hash_combine(key, params.use_bias);
  hash_combine(key, params.activation);
  return key;
}
}; // namespace std

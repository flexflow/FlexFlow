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

// For now, we use one input and one output per expert
Tensor FFModel::experts(Tensor const *inputs,
                        int num_experts,
                        int experts_start_idx,
                        int experts_output_dim_size,
                        float alpha,
                        int experts_num_layers,
                        int experts_internal_dim_size,
                        char const *name) {

  // Check that there are three inputs: the input tensor, the indices and the
  // topk_gate_preds
  assert(inputs[0] != nullptr);
  int num_dims = inputs[0]->num_dims;
  assert(inputs[1]->num_dims == num_dims);
  assert(inputs[2]->num_dims == num_dims);
  int topk = inputs[1]->dims[0];
  assert(inputs[2]->dims[0] == topk);
  for (int i = 1; i < num_dims; i++) {
    assert(inputs[0]->dims[i] == inputs[1]->dims[i]);
    assert(inputs[1]->dims[i] == inputs[2]->dims[i]);
  }

  assert(inputs[1]->data_type == DT_INT32 || inputs[1]->data_type == DT_INT64);

  assert(experts_num_layers == 1 && "Multi-layer experts not implemented yet.");
  assert(experts_num_layers == 1 || experts_internal_dim_size > 0);

  // parameters for the FFN implementing the experts. We can make these
  // FFModel::experts(...) function parameters if needed.
  bool use_bias = false;
  ActiMode activation = AC_MODE_RELU;

  Layer *e = new Layer(this,
                       OP_EXPERTS,
                       DT_FLOAT,
                       name,
                       3 /*inputs*/,
                       num_experts * (1 + use_bias) /*weights*/,
                       1 /*outputs*/,
                       inputs);
  {
    int dims[MAX_TENSOR_DIM];
    for (int i = 1; i < num_dims; i++) {
      dims[i] = inputs[0]->dims[i];
    }
    dims[0] = experts_output_dim_size;
    e->outputs[0] = create_tensor_legion_ordering(
        num_dims, dims, DT_FLOAT, e, 0, true /*create_grad*/);
    assert(e->outputs[0] != nullptr);
  }
  for (int i = 0; i < num_experts; i++) {
    {
      int dims[2] = {inputs[0]->dims[0], experts_output_dim_size};
      e->weights[i * (1 + use_bias)] =
          create_weight_legion_ordering(2,
                                        dims,
                                        DT_FLOAT,
                                        e,
                                        true /*create_grad*/,
                                        nullptr,
                                        CHOSEN_SYNC_TYPE);
    }
    if (use_bias) {
      int dims[1] = {experts_output_dim_size};
      e->weights[i * (1 + use_bias) + use_bias] =
          create_weight_legion_ordering(1,
                                        dims,
                                        DT_FLOAT,
                                        e,
                                        true /*create_grad*/,
                                        nullptr,
                                        CHOSEN_SYNC_TYPE);
    }
  }

  e->add_int_property("num_experts", num_experts);
  e->add_int_property("experts_start_idx", experts_start_idx);
  e->add_int_property("experts_output_dim_size", experts_output_dim_size);
  e->add_float_property("alpha", alpha);
  e->add_int_property("experts_num_layers", experts_num_layers);
  e->add_int_property("experts_internal_dim_size", experts_internal_dim_size);
  e->add_int_property("use_bias", use_bias);
  e->add_int_property("activation", activation);
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
  layer->get_int_property("experts_output_dim_size", value);
  int experts_output_dim_size = value;
  float value2;
  layer->get_float_property("alpha", value2);
  float alpha = value2;
  layer->get_int_property("experts_num_layers", value);
  int experts_num_layers = value;
  layer->get_int_property("experts_internal_dim_size", value);
  int experts_internal_dim_size = value;
  layer->get_int_property("use_bias", value);
  bool use_bias = (bool)value;
  layer->get_int_property("activation", value);
  ActiMode activation = (ActiMode)value;
  return new Experts(model,
                     layer->layer_guid,
                     inputs.data(),
                     num_experts,
                     experts_start_idx,
                     experts_output_dim_size,
                     alpha,
                     experts_num_layers,
                     experts_internal_dim_size,
                     use_bias,
                     activation,
                     false /*allocate_weights*/,
                     layer->name);
}

ExpertsParams Experts::get_params() const {
  ExpertsParams params;
  params.layer_guid = this->layer_guid;
  params.num_experts = num_experts;
  params.experts_start_idx = experts_start_idx;
  params.experts_output_dim_size = experts_output_dim_size;
  params.alpha = alpha;
  params.experts_num_layers = experts_num_layers;
  params.experts_internal_dim_size = experts_internal_dim_size;
  params.use_bias = use_bias;
  params.activation = activation;
  return params;
}

bool ExpertsParams::is_valid(
    std::vector<ParallelTensorShape> const &inputs) const {
  if (inputs.size() != 3) {
    printf("Number of inputs to the Experts layer is wrong\n");
    return false;
  }
  if (!inputs[0].is_valid()) {
    printf("The first tensor passed to the Experts layer is not valid\n");
    return false;
  }
  if (!inputs[1].is_valid()) {
    printf("The second tensor passed to the Experts layer is not valid\n");
    return false;
  }
  if (!inputs[2].is_valid()) {
    printf("The third tensor passed to the Experts layer is not valid\n");
    return false;
  }
  if (inputs[0].num_dims != inputs[1].num_dims ||
      inputs[1].num_dims != inputs[2].num_dims) {
    printf("Mismatch found between the number of dimensions of the three input "
           "tensors for the Expert layer\n");
    return false;
  }
  if (inputs[0].data_type != DT_FLOAT) {
    printf("Data type of the first input to the Experts layer is wrong!\n");
    return false;
  }
  if (inputs[1].data_type != DT_INT32 && inputs[1].data_type != DT_INT64) {
    printf("Data type of the second input to the Experts layer is wrong!\n");
    return false;
  }
  if (inputs[2].data_type != DT_FLOAT) {
    printf("Data type of the third input to the Experts layer is wrong!\n");
    return false;
  }
  if (inputs[1].dims[0] != inputs[2].dims[0]) {
    printf(
        "Dimension mismatch between indices and topk_gate_preds tensors passed "
        "to the Experts layer.\n");
    return false;
  }
  for (int i = 1; i < inputs[0].num_dims; i++) {
    if (inputs[0].dims[i] != inputs[1].dims[i] ||
        inputs[1].dims[i] != inputs[2].dims[i]) {
      printf("Dimension mismatch among the input tensors passed to the Experts "
             "layer.\n");
      return false;
    }
  }
  return true;
}

bool operator==(ExpertsParams const &lhs, ExpertsParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid &&
         lhs.num_experts == rhs.num_experts &&
         lhs.experts_start_idx == rhs.experts_start_idx &&
         lhs.experts_output_dim_size == rhs.experts_output_dim_size &&
         lhs.alpha == rhs.alpha &&
         lhs.experts_num_layers == rhs.experts_num_layers &&
         lhs.experts_internal_dim_size == rhs.experts_internal_dim_size &&
         lhs.use_bias == rhs.use_bias && lhs.activation == rhs.activation;
}

Experts::Experts(FFModel &model,
                 ExpertsParams const &params,
                 std::vector<ParallelTensor> const &inputs,
                 bool allocate_weights,
                 char const *name)
    : Experts(model,
              params.layer_guid,
              inputs.data(),
              params.num_experts,
              params.experts_start_idx,
              params.experts_output_dim_size,
              params.alpha,
              params.experts_num_layers,
              params.experts_internal_dim_size,
              params.use_bias,
              params.activation,
              allocate_weights,
              name) {}

Experts::Experts(FFModel &model,
                 LayerID const &_layer_guid,
                 ParallelTensor const *inputs,
                 int _num_experts,
                 int _experts_start_idx,
                 int _experts_output_dim_size,
                 float _alpha,
                 int _experts_num_layers,
                 int _experts_internal_dim_size,
                 bool _use_bias,
                 ActiMode _activation,
                 bool allocate_weights,
                 char const *name)
    : Op(model,
         OP_EXPERTS,
         DT_FLOAT,
         name,
         3 /*inputs*/,
         _num_experts * (1 + _use_bias) /*weights*/,
         1 /*outputs*/,
         inputs),
      num_experts(_num_experts), experts_start_idx(_experts_start_idx),
      experts_output_dim_size(_experts_output_dim_size), alpha(_alpha),
      experts_num_layers(_experts_num_layers),
      experts_internal_dim_size(_experts_internal_dim_size),
      use_bias(_use_bias), activation(_activation) {

  // overwrite layer_guid
  layer_guid = _layer_guid;

  assert(num_experts > 0);
  assert(numInputs == 3);
  assert(numOutputs == 1);
  assert(numWeights == num_experts * (1 + use_bias));

  assert(inputs[0] != nullptr);
  int num_dims = inputs[0]->num_dims;
  assert(inputs[1]->num_dims == num_dims);
  assert(inputs[2]->num_dims == num_dims);

  int topk = inputs[1]->dims[0].size;
  assert(inputs[2]->dims[0].size == topk);

  for (int i = 1; i < num_dims; i++) {
    assert(inputs[0]->dims[i] == inputs[1]->dims[i]);
    assert(inputs[1]->dims[i] == inputs[2]->dims[i]);
  }

  assert(inputs[1]->data_type == DT_INT32 || inputs[1]->data_type == DT_INT64);
  assert(experts_num_layers == 1 && "Multi-layer experts not implemented yet.");
  assert(experts_num_layers == 1 || experts_internal_dim_size > 0);

  // Assume that we don't parallelize the channel dim of input
  // nor the expert_assigned dim of indices
  assert(inputs[0]->dims[0].degree == 1);
  assert(inputs[1]->dims[0].degree == 1);
  assert(inputs[2]->dims[0].degree == 1);

  ParallelDim out_dims[MAX_TENSOR_DIM];
  for (int i = 0; i < num_dims; i++) {
    out_dims[i] = inputs[0]->dims[i];
  }
  out_dims[0].size = experts_output_dim_size;
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      num_dims, out_dims, inputs[0]->data_type, this, 0 /*owner_idx*/);
  assert(outputs[0] != nullptr);

  //auto dimension_names = this->get_params().get_dimension_names(inputs[0]->get_shape());
  ParallelTensorShape input_shape = inputs[0]->get_shape();
  ParallelTensorShape output_shape, kernel_shape, bias_shape;
  ExpertsParams params = this->get_params();
  params.construct_mappings(*this->parallel_dims_mapping, input_shape);
  params.solve_dims(input_shape, output_shape, kernel_shape, bias_shape);

  if (allocate_weights) {
#ifdef USE_NCCL
    ParameterSyncType comm_type = ParameterSyncType::NCCL;
#else
    ParameterSyncType comm_type = ParameterSyncType::PS;
#endif
    for (int i = 0; i < num_experts; i++) {
      Initializer *kernel_initializer = new GlorotUniform(std::rand() /*seed*/);
      {
        //ParallelDim dims[2] = {inputs[0]->dims[0], out_dims[0]};
        weights[i * (1 + use_bias)] =
            model.create_parallel_weight_legion_ordering(kernel_shape.num_dims, //2,
                                                         kernel_shape.dims, //dims,
                                                         DT_FLOAT,
                                                         NULL /*owner_op*/,
                                                         true /*create_grad*/,
                                                         kernel_initializer,
                                                         comm_type);
        assert(weights[i * (1 + use_bias)] != nullptr);
      }
      if (use_bias) {
        Initializer *bias_initializer = new ZeroInitializer();
        ParallelDim dims[1] = {out_dims[0]};
        weights[i * (1 + use_bias) + use_bias] =
            model.create_parallel_weight_legion_ordering(bias_shape.num_dims, //1,
                                                         bias_shape.dims, //dims,
                                                         DT_FLOAT,
                                                         NULL /*owner_op*/,
                                                         true /*create_grad*/,
                                                         bias_initializer,
                                                         comm_type);
        assert(weights[i * (1 + use_bias) + use_bias] != nullptr);
      }
    }
  }
  assert(check_output_input_weight_parallel_dims(allocate_weights));
}

void Experts::serialize(Legion::Serializer &sez) const {
  ExpertsParams params = get_params();
  sez.serialize(params.layer_guid.id);
  sez.serialize(params.num_experts);
  sez.serialize(params.experts_start_idx);
  sez.serialize(params.experts_output_dim_size);
  sez.serialize(params.alpha);
  sez.serialize(params.experts_num_layers);
  sez.serialize(params.experts_internal_dim_size);
  sez.serialize(params.use_bias);
  sez.serialize(params.activation);
}

using PCG::Node;
Node Experts::deserialize(FFModel &ff,
                          Legion::Deserializer &dez,
                          std::vector<ParallelTensor> const &inputs,
                          int num_inputs) {
  int num_experts, experts_start_idx, experts_output_dim_size,
      experts_num_layers, experts_internal_dim_size;
  float alpha;
  ActiMode activation;
  bool use_bias;
  size_t id;
  dez.deserialize(id);
  LayerID layer_guid(id);
  dez.deserialize(num_experts);
  dez.deserialize(experts_start_idx);
  dez.deserialize(experts_output_dim_size);
  dez.deserialize(alpha);
  dez.deserialize(experts_num_layers);
  dez.deserialize(experts_internal_dim_size);
  dez.deserialize(use_bias);
  dez.deserialize(activation);

  assert(num_inputs == 3);

  ExpertsParams params;
  params.layer_guid = layer_guid;
  params.num_experts = num_experts;
  params.experts_start_idx = experts_start_idx;
  params.experts_output_dim_size = experts_output_dim_size;
  params.alpha = alpha;
  params.experts_num_layers = experts_num_layers;
  params.experts_internal_dim_size = experts_internal_dim_size;
  params.use_bias = use_bias;
  params.activation = activation;

  return ff.get_or_create_node<Experts>(inputs, params);
}

void Experts::init_inference(FFModel const &ff,
                             std::vector<ParallelTensor> const &batch_inputs,
                             std::vector<ParallelTensor> const &batch_outputs) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = batch_outputs[0]->parallel_is;
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
                         batch_outputs[0]->machine_view.hash());
  // expert predictions
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // expert assignment indices
  launcher.add_region_requirement(RegionRequirement(batch_inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // topk_gate_preds
  launcher.add_region_requirement(RegionRequirement(batch_inputs[2]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[2]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(3, FID_DATA);
  for (int i = 0; i < num_experts; i++) {
    launcher.add_region_requirement(
        RegionRequirement(weights[i * (1 + use_bias)]->part,
                          0 /*projection id*/,
                          READ_ONLY,
                          EXCLUSIVE,
                          weights[i * (1 + use_bias)]->region));
    launcher.add_field(4 + i * (1 + use_bias), FID_DATA);
    if (use_bias) {
      launcher.add_region_requirement(
          RegionRequirement(weights[i * (1 + use_bias) + use_bias]->part,
                            0 /*projection id*/,
                            READ_ONLY,
                            EXCLUSIVE,
                            weights[i * (1 + use_bias) + use_bias]->region));
      launcher.add_field(4 + i * (1 + use_bias) + use_bias, FID_DATA);
    }
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
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
  // expert predictions
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // expert assignment indices
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // topk_gate_preds
  launcher.add_region_requirement(RegionRequirement(inputs[2]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[2]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(3, FID_DATA);
  for (int i = 0; i < num_experts; i++) {
    launcher.add_region_requirement(
        RegionRequirement(weights[i * (1 + use_bias)]->part,
                          0 /*projection id*/,
                          READ_ONLY,
                          EXCLUSIVE,
                          weights[i * (1 + use_bias)]->region));
    launcher.add_field(4 + i * (1 + use_bias), FID_DATA);
    if (use_bias) {
      launcher.add_region_requirement(
          RegionRequirement(weights[i * (1 + use_bias) + use_bias]->part,
                            0 /*projection id*/,
                            READ_ONLY,
                            EXCLUSIVE,
                            weights[i * (1 + use_bias) + use_bias]->region));
      launcher.add_field(4 + i * (1 + use_bias) + use_bias, FID_DATA);
    }
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *Experts::init_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  Experts const *exp = (Experts *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  ExpertsMeta *m = new ExpertsMeta(handle,
                                   exp->num_experts,
                                   exp->experts_start_idx,
                                   exp->alpha,
                                   exp->use_bias,
                                   exp->activation);
  m->profiling = exp->profiling;
  return m;
}

void Experts::forward(FFModel const &ff) {
  // assert(false && "Experts is designed for inference only");
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(EXPERTS_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // expert predictions
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // expert assignment indices
  launcher.add_region_requirement(RegionRequirement(inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // topk_gate_preds
  launcher.add_region_requirement(RegionRequirement(inputs[2]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[2]->region));
  launcher.add_field(2, FID_DATA);
  // expert output per token (only the chosen experts have non-zero
  // contributions)
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(3, FID_DATA);
  for (int i = 0; i < num_experts; i++) {
    launcher.add_region_requirement(
        RegionRequirement(weights[i * (1 + use_bias)]->part,
                          0 /*projection id*/,
                          READ_ONLY,
                          EXCLUSIVE,
                          weights[i * (1 + use_bias)]->region));
    launcher.add_field(4 + i * (1 + use_bias), FID_DATA);
    if (use_bias) {
      launcher.add_region_requirement(
          RegionRequirement(weights[i * (1 + use_bias) + use_bias]->part,
                            0 /*projection id*/,
                            READ_ONLY,
                            EXCLUSIVE,
                            weights[i * (1 + use_bias) + use_bias]->region));
      launcher.add_field(4 + i * (1 + use_bias) + use_bias, FID_DATA);
    }
  }
  runtime->execute_index_space(ctx, launcher);
}

void Experts::inference(FFModel const &ff,
                        std::vector<ParallelTensor> const &batch_inputs,
                        std::vector<ParallelTensor> const &batch_outputs,
                        MachineView const *mv) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  size_t machine_view_hash =
      mv ? mv->hash() : batch_outputs[0]->machine_view.hash();
  IndexLauncher launcher(EXPERTS_INF_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  // expert predictions
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // expert assignment indices
  launcher.add_region_requirement(RegionRequirement(batch_inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  // topk_gate_preds
  launcher.add_region_requirement(RegionRequirement(batch_inputs[2]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[2]->region));
  launcher.add_field(2, FID_DATA);
  // expert output per token (only the chosen experts have non-zero
  // contributions)
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(3, FID_DATA);
  for (int i = 0; i < num_experts; i++) {
    launcher.add_region_requirement(
        RegionRequirement(weights[i * (1 + use_bias)]->part,
                          0 /*projection id*/,
                          READ_ONLY,
                          EXCLUSIVE,
                          weights[i * (1 + use_bias)]->region));
    launcher.add_field(4 + i * (1 + use_bias), FID_DATA);
    if (use_bias) {
      launcher.add_region_requirement(
          RegionRequirement(weights[i * (1 + use_bias) + use_bias]->part,
                            0 /*projection id*/,
                            READ_ONLY,
                            EXCLUSIVE,
                            weights[i * (1 + use_bias) + use_bias]->region));
      launcher.add_field(4 + i * (1 + use_bias) + use_bias, FID_DATA);
    }
  }
  runtime->execute_index_space(ctx, launcher);
}

void Experts::inference_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  assert(regions.size() == task->regions.size());

  ExpertsMeta const *m = *((ExpertsMeta **)task->local_args);

  int num_experts = m->num_experts;
  bool use_bias = m->use_bias;
  assert(regions.size() - 4 == num_experts * (1 + use_bias));

  // get input, indices, topk_gate_preds, outputs
  float const *input_ptr = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  int const *indices_ptr = helperGetTensorPointerRO<int>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  float const *topk_gate_pred_ptr = helperGetTensorPointerRO<float>(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  float *output_ptr = helperGetTensorPointerWO<float>(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);
  assert(input_ptr != nullptr && indices_ptr != nullptr &&
         topk_gate_pred_ptr != nullptr && output_ptr != nullptr);

  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain indices_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Domain topk_gate_pred_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
      ctx, task->regions[3].region.get_index_space());

  int input_dims = input_domain.get_dim();
  int indices_dims = indices_domain.get_dim();
  int topk_gate_pred_dims = topk_gate_pred_domain.get_dim();
  assert(input_dims == indices_dims);
  assert(indices_dims == topk_gate_pred_dims);

  int replica_dim = input_dims - 1;
  int samples_index = input_dims - 2;

  coord_t data_dim = input_domain.hi()[0] - input_domain.lo()[0] + 1;
  coord_t batch_size =
      input_domain.hi()[samples_index] - input_domain.lo()[samples_index] + 1;
  coord_t chosen_experts = indices_domain.hi()[0] - indices_domain.lo()[0];
  coord_t out_dim = output_domain.hi()[0] - output_domain.lo()[0] + 1;
  assert(chosen_experts ==
         topk_gate_pred_domain.hi()[0] - topk_gate_pred_domain.lo()[0]);

  for (int i = 1; i < input_dims; i++) {
    int a = input_domain.hi()[i] - input_domain.lo()[i] + 1;
    int b = indices_domain.hi()[i] - indices_domain.lo()[i] + 1;
    int c = topk_gate_pred_domain.hi()[i] - topk_gate_pred_domain.lo()[i] + 1;
    assert(a == b && b == c);
    if (i >= 1 && i < samples_index) {
      batch_size *= a;
    }
  }

  assert(batch_size <= MAX_BATCH_SIZE &&
         "batch size exceeds MAX_BATCH_SIZE defined in experts.h");
  assert(
      num_experts <= MAX_EXPERTS_PER_BLOCK &&
      "number of experts exceeds MAX_EXPERTS_PER_BLOCK defined in experts.h");

  for (int j = 1; j < input_dims; j++) {
    int a = input_domain.hi()[j] - input_domain.lo()[j] + 1;
    int b = output_domain.hi()[j] - output_domain.lo()[j] + 1;
    assert(a == b);
  }

  // get weights
  float const *weights_ptrs[num_experts * (1 + use_bias)];
  for (int i = 0; i < num_experts; i++) {
    weights_ptrs[i * (1 + use_bias)] =
        helperGetTensorPointerRO<float>(regions[4 + i * (1 + use_bias)],
                                        task->regions[4 + i * (1 + use_bias)],
                                        FID_DATA,
                                        ctx,
                                        runtime);
    Domain weights_domain = runtime->get_index_space_domain(
        ctx, task->regions[4 + i * (1 + use_bias)].region.get_index_space());
    int weights_dims = weights_domain.get_dim();
    assert(weights_dims == 2);
    assert(weights_domain.hi()[0] - weights_domain.lo()[0] + 1 == data_dim);
    assert(weights_domain.hi()[1] - weights_domain.lo()[1] + 1 == out_dim);
    if (use_bias) {
      weights_ptrs[i * (1 + use_bias) + use_bias] =
          helperGetTensorPointerRO<float>(
              regions[4 + i * (1 + use_bias) + use_bias],
              task->regions[4 + i * (1 + use_bias) + use_bias],
              FID_DATA,
              ctx,
              runtime);
      Domain bias_domain = runtime->get_index_space_domain(
          ctx,
          task->regions[4 + i * (1 + use_bias) + use_bias]
              .region.get_index_space());
      int bias_dims = bias_domain.get_dim();
      assert(bias_dims == 1);
      assert(bias_domain.hi()[0] - bias_domain.lo()[0] + 1 == out_dim);
    }
  }

  Experts::forward_kernel_wrapper(m,
                                  input_ptr,
                                  indices_ptr,
                                  topk_gate_pred_ptr,
                                  output_ptr,
                                  weights_ptrs,
                                  chosen_experts,
                                  batch_size,
                                  out_dim);
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

void Experts::print_layer(FFModel const &ff) {
  return;
}

bool Experts::measure_operator_cost(Simulator *sim,
                                    MachineView const &c,
                                    CostMetrics &cost_metrics) const {
  // This is an inference only operator
  assert(false && "Experts is designed for inference only");
  return false;
}

void ExpertsParams::solve_dims(const ParallelTensor input,
                              ParallelDim output_dims[MAX_TENSOR_DIM],
                              int *output_ndims,
                              ParallelDim kernel_dims[MAX_TENSOR_DIM],
                              int *kernel_ndims,
                              ParallelDim bias_dims[MAX_TENSOR_DIM],
                              int *bias_ndims) const {
  this->solve_dims(input->get_shape(),
                   output_dims,
                   output_ndims,
                   kernel_dims,
                   kernel_ndims,
                   bias_dims,
                   bias_ndims);
}

void ExpertsParams::solve_dims(ParallelTensorShape const &input_shape,
                              ParallelTensorShape &output_shape,
                              ParallelTensorShape &kernel_shape,
                              ParallelTensorShape &bias_shape) const {
  this->solve_dims(input_shape,
                   output_shape.dims,
                   &output_shape.num_dims,
                   kernel_shape.dims,
                   &kernel_shape.num_dims,
                   bias_shape.dims,
                   &bias_shape.num_dims);
}

void ExpertsParams::solve_dims(ParallelTensorShape const &input_shape,
                              ParallelDim output_dims[MAX_TENSOR_DIM],
                              int *output_ndims,
                              ParallelDim kernel_dims[MAX_TENSOR_DIM],
                              int *kernel_ndims,
                              ParallelDim bias_dims[MAX_TENSOR_DIM],
                              int *bias_ndims) const {
  assert((output_dims == nullptr) == (output_ndims == nullptr));
  assert((kernel_dims == nullptr) == (kernel_ndims == nullptr));
  assert((bias_dims == nullptr) == (bias_ndims == nullptr));

  std::vector<ParallelDimMappingRecord> mapping;
  this->construct_mappings(mapping, input_shape);
  this->mark_replica_dims(input_shape, output_dims, kernel_dims, bias_dims);

  solve_parallel_dim_mappings(
      mapping, {input_shape.dims}, {kernel_dims, bias_dims}, {output_dims});

  this->calculate_nonreplica_dim_sizes(input_shape,
                                       output_dims,
                                       output_ndims,
                                       kernel_dims,
                                       kernel_ndims,
                                       bias_dims,
                                       bias_ndims);
}

std::unordered_map<ExpertsParams::NamedDimensions, int>
    ExpertsParams::get_dimension_names(
        ParallelTensorShape const &input_shape) const {
  int num_dims = input_shape.num_dims;

  return {{INPUT_CHANNEL, 0},
          {INPUT_SAMPLE, num_dims - 2},
          {INPUT_REPLICA, num_dims - 1},
          {OUTPUT_CHANNEL, 0},
          {OUTPUT_SAMPLE, num_dims - 2},
          {OUTPUT_REPLICA, num_dims - 1},
          {KERNEL_CHANNEL_IN, 0},
          {KERNEL_CHANNEL_OUT, 1},
          {BIAS_CHANNEL_OUT, 0}};
}

void ExpertsParams::calculate_nonreplica_dim_sizes(
    ParallelTensorShape const &input_shape,
    ParallelDim output_dims[MAX_TENSOR_DIM],
    int *output_ndims,
    ParallelDim kernel_dims[MAX_TENSOR_DIM],
    int *kernel_ndims,
    ParallelDim bias_dims[MAX_TENSOR_DIM],
    int *bias_ndims) const {
  auto dimension_names = this->get_dimension_names(input_shape);
  int num_dims = input_shape.num_dims;

  if (output_dims != nullptr) {
    for (int i = 1; i < input_shape.num_dims - 1; i++) {
      output_dims[i].size = input_shape.dims[i].size;
    }
    output_dims[dimension_names.at(OUTPUT_CHANNEL)].size = this->out_channels;
    *output_ndims = num_dims;
  }
  if (kernel_dims != nullptr) {
    kernel_dims[dimension_names.at(KERNEL_CHANNEL_IN)].size =
        input_shape.dims[INPUT_CHANNEL].size /
        input_shape.dims[INPUT_CHANNEL].degree;
    kernel_dims[dimension_names.at(KERNEL_CHANNEL_OUT)].size =
        this->out_channels;
    *kernel_ndims = num_dims;
  }
  if (bias_dims != nullptr) {
    bias_dims[dimension_names.at(BIAS_CHANNEL_OUT)].size = this->out_channels;
    *bias_ndims = num_dims;
  }
}

void ExpertsParams::mark_replica_dims(
    ParallelTensorShape const &input_shape,
    ParallelDim output_dims[MAX_TENSOR_DIM],
    ParallelDim kernel_dims[MAX_TENSOR_DIM],
    ParallelDim bias_dims[MAX_TENSOR_DIM]) const {
  int num_dims = input_shape.num_dims;
  auto dimension_names = this->get_dimension_names(input_shape);
  if (output_dims != nullptr) {
    output_dims[dimension_names.at(OUTPUT_REPLICA)].is_replica_dim = true;
  }
  if (kernel_dims != nullptr) {
    for (int i = 2; i < num_dims; i++) {
      kernel_dims[i].is_replica_dim = true;
    }
  }
  if (bias_dims != nullptr) {
    for (int i = 1; i < num_dims; i++) {
      bias_dims[i].is_replica_dim = true;
    }
  }
}

void ExpertsParams::construct_mappings(
    std::vector<ParallelDimMappingRecord> &mappings,
    ParallelTensorShape const &input_shape) const {
  std::unordered_map<NamedDimensions, int> dimension_names =
      this->get_dimension_names(input_shape);

  Op::construct_output_parallel_dims(
      mappings,
      {{dimension_names.at(INPUT_CHANNEL), dimension_names.at(OUTPUT_REPLICA)},
       {dimension_names.at(INPUT_REPLICA),
        dimension_names.at(OUTPUT_CHANNEL)}});
  for (int i = 1; i < input_shape.num_dims - 1; i++) {
    Op::construct_output_parallel_dims(mappings, i, i);
  }

  Op::construct_weight_parallel_dims(mappings,
                                     {{dimension_names.at(INPUT_CHANNEL),
                                       dimension_names.at(KERNEL_CHANNEL_IN)},
                                      {dimension_names.at(INPUT_REPLICA),
                                       dimension_names.at(KERNEL_CHANNEL_OUT)}},
                                     0 /*input_idx*/,
                                     KERNEL_IDX);
  // map a bunch of replica dimensions for the unnamed dimensions in the input
  for (int i = 1; i < input_shape.num_dims - 1; i++) {
    Op::construct_weight_parallel_dims(
        mappings, i, i + 1, 0 /*input_idx*/, KERNEL_IDX);
  }

  Op::construct_weight_parallel_dims(mappings,
                                     {
                                         {dimension_names.at(INPUT_REPLICA),
                                          dimension_names.at(BIAS_CHANNEL_OUT)},
                                     },
                                     0 /*input_idx*/,
                                     BIAS_IDX);
  for (int i = 0; i < input_shape.num_dims - 1; i++) {
    Op::construct_weight_parallel_dims(
        mappings, i, i + 1, 0 /*input_idx*/, BIAS_IDX);
  }
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ExpertsParams>::operator()(
    FlexFlow::ExpertsParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.num_experts);
  hash_combine(key, params.experts_start_idx);
  hash_combine(key, params.experts_output_dim_size);
  hash_combine(key, params.alpha);
  hash_combine(key, params.experts_num_layers);
  hash_combine(key, params.experts_internal_dim_size);
  hash_combine(key, params.use_bias);
  hash_combine(key, params.activation);
  return key;
}
}; // namespace std

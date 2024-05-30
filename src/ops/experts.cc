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
#ifdef INFERENCE_TESTS
#include "flexflow/utils/cuda_helper.h"
#endif
#include "legion/legion_utilities.h"

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::Future;
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
#ifdef INFERENCE_TESTS
static bool DEBUG_MODE = false;
#endif

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

  assert(experts_num_layers >= 1);
  assert(experts_num_layers <= 2 && "Multi-layer experts not implemented yet.");
  assert(experts_num_layers == 1 || experts_internal_dim_size > 0);

  // parameters for the FFN implementing the experts. We can make these
  // FFModel::experts(...) function parameters if needed.
  bool use_bias = true;
  ActiMode activation = AC_MODE_RELU;

  Layer *e = new Layer(this,
                       OP_EXPERTS,
                       DT_FLOAT,
                       name,
                       3 /*inputs*/,
                       (1 + use_bias) /*weights*/,
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
  {
    int nparams = (experts_num_layers == 1)
                      ? (inputs[0]->dims[0] * experts_output_dim_size)
                      : experts_internal_dim_size *
                            (inputs[0]->dims[0] + experts_output_dim_size);
    int dims[2] = {nparams, num_experts};
    e->weights[0] = create_weight_legion_ordering(
        2, dims, DT_FLOAT, e, true /*create_grad*/, nullptr, CHOSEN_SYNC_TYPE);
  }
  if (use_bias) {
    int nparams = (experts_num_layers == 1)
                      ? experts_output_dim_size
                      : (experts_internal_dim_size + experts_output_dim_size);
    int dims[2] = {nparams, num_experts};
    e->weights[1] = create_weight_legion_ordering(
        2, dims, DT_FLOAT, e, true /*create_grad*/, nullptr, CHOSEN_SYNC_TYPE);
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
              params.name) {}

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
         (1 + _use_bias) /*weights*/,
         1 /*outputs*/,
         inputs),
      num_experts(_num_experts), experts_start_idx(_experts_start_idx),
      experts_output_dim_size(_experts_output_dim_size), alpha(_alpha),
      experts_num_layers(_experts_num_layers),
      experts_internal_dim_size(_experts_internal_dim_size),
      use_bias(_use_bias), activation(_activation) {

  // overwrite layer_guid
  layer_guid = _layer_guid;

  // Check number of inputs, output, weights
  assert(num_experts > 0);
  assert(numInputs == 3);
  assert(numOutputs == 1);
  assert(numWeights == (1 + use_bias));

  // Check input dimensions
  int num_dims = inputs[0]->num_dims;
  int topk = inputs[1]->dims[0].size;
  assert(inputs[0] != nullptr);
  assert(inputs[1]->num_dims == num_dims);
  assert(inputs[2]->num_dims == num_dims);
  assert(inputs[2]->dims[0].size == topk);
  for (int i = 1; i < num_dims; i++) {
    assert(inputs[0]->dims[i] == inputs[1]->dims[i]);
    assert(inputs[1]->dims[i] == inputs[2]->dims[i]);
  }
  // Assume that we don't parallelize the channel dim of input
  // nor the expert_assigned dim of indices
  assert(inputs[0]->dims[0].degree == 1);
  assert(inputs[1]->dims[0].degree == 1);
  assert(inputs[2]->dims[0].degree == 1);
  // check data type of indices input
  assert(inputs[1]->data_type == DT_INT32 || inputs[1]->data_type == DT_INT64);
  assert(experts_num_layers >= 1);
  assert(experts_num_layers <= 2 && "Multi-layer experts not implemented yet.");
  assert(experts_num_layers == 1 || experts_internal_dim_size > 0);

  // save the token embedding dimension (data_dim) and the effective batch size
  data_dim = inputs[0]->dims[0].size;
  effective_batch_size = 1;
  for (int i = 1; i <= num_dims - 2; i++) {
    effective_batch_size *= inputs[0]->dims[i].size;
  }
  num_chosen_experts = topk;

  out_dim = _experts_output_dim_size;

  // Create the parallel tensor for the output
  ParallelDim out_dims[MAX_TENSOR_DIM];
  for (int i = 0; i < num_dims; i++) {
    out_dims[i] = inputs[0]->dims[i];
  }
  out_dims[0].size = experts_output_dim_size;
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      num_dims, out_dims, inputs[0]->data_type, this, 0 /*owner_idx*/);
  assert(outputs[0] != nullptr);

  if (allocate_weights) {
    {
      ParallelDim dims[3];
      int nparams = (experts_num_layers == 1)
                        ? (data_dim * experts_output_dim_size)
                        : experts_internal_dim_size *
                              (data_dim + experts_output_dim_size);
      dims[0].size = nparams;
      dims[0].degree = 1;
      dims[0].parallel_idx = -1;
      dims[1] = inputs[0]->dims[num_dims - 1];
      dims[1].size = num_experts;
      dims[2] = inputs[0]->dims[num_dims - 2];
      dims[2].size = dims[0].degree;
      Initializer *kernel_initializer = new GlorotUniform(std::rand() /*seed*/);
      // assert(kernel_shape.dims[2].size == num_experts);
      weights[0] =
          model.create_parallel_weight_legion_ordering(3,
                                                       dims,
                                                       DT_FLOAT,
                                                       NULL /*owner_op*/,
                                                       true /*create_grad*/,
                                                       kernel_initializer,
                                                       CHOSEN_SYNC_TYPE);
      assert(weights[0] != nullptr);
    }
    if (use_bias) {
      Initializer *bias_initializer = new ZeroInitializer();
      // assert(bias_shape.dims[1].size == num_experts);
      ParallelDim dims[3];
      int nparams = (experts_num_layers == 1)
                        ? experts_output_dim_size
                        : (experts_internal_dim_size + experts_output_dim_size);
      dims[0].size = nparams;
      dims[0].degree = 1;
      dims[0].parallel_idx = -1;
      dims[1] = inputs[0]->dims[num_dims - 1];
      dims[1].size = num_experts;
      dims[2] = inputs[0]->dims[num_dims - 2];
      dims[2].size = dims[0].degree;
      weights[1] =
          model.create_parallel_weight_legion_ordering(3,
                                                       dims,
                                                       DT_FLOAT,
                                                       NULL /*owner_op*/,
                                                       true /*create_grad*/,
                                                       bias_initializer,
                                                       CHOSEN_SYNC_TYPE);
      assert(weights[1] != nullptr);
    }
  }
  assert(check_output_input_weight_parallel_dims(allocate_weights));
}

void Experts::serialize(Legion::Serializer &sez) const {
  ExpertsParams params = get_params();
  sez.serialize(params.layer_guid.id);
  sez.serialize(params.layer_guid.transformer_layer_id);
  sez.serialize(params.layer_guid.model_id);
  sez.serialize(params.num_experts);
  sez.serialize(params.experts_start_idx);
  sez.serialize(params.experts_output_dim_size);
  sez.serialize(params.alpha);
  sez.serialize(params.experts_num_layers);
  sez.serialize(params.experts_internal_dim_size);
  sez.serialize(params.use_bias);
  sez.serialize(params.activation);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
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
  size_t id, transformer_layer_id, deserialized_model_id;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  dez.deserialize(deserialized_model_id);
  LayerID layer_guid(id, transformer_layer_id, deserialized_model_id);
  dez.deserialize(num_experts);
  dez.deserialize(experts_start_idx);
  dez.deserialize(experts_output_dim_size);
  dez.deserialize(alpha);
  dez.deserialize(experts_num_layers);
  dez.deserialize(experts_internal_dim_size);
  dez.deserialize(use_bias);
  dez.deserialize(activation);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);

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
  strcpy(params.name, name);

  return ff.get_or_create_node<Experts>(inputs, params);
}

void Experts::init_inference(FFModel const &ff,
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
  IndexLauncher launcher(EXPERTS_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Experts)),
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
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(3, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(4, FID_DATA);
  if (use_bias) {
    launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[1]->region));
    launcher.add_field(5, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
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
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(4, FID_DATA);
  if (use_bias) {
    launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[1]->region));
    launcher.add_field(5, FID_DATA);
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
                                   exp->data_dim,
                                   exp->out_dim,
                                   exp->experts_num_layers,
                                   exp->experts_internal_dim_size,
                                   exp->effective_batch_size,
                                   exp->num_chosen_experts,
                                   exp->alpha,
                                   exp->use_bias,
                                   exp->activation);
  m->profiling = exp->profiling;
  m->inference_debugging = exp->inference_debugging;
  std::strcpy(m->op_name, exp->name);
  m->layer_guid = exp->layer_guid;
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
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(4, FID_DATA);
  if (use_bias) {
    launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[1]->region));
    launcher.add_field(5, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

FutureMap Experts::inference(FFModel const &ff,
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
  /* std::cout << "Experts op machine_view: " << *(MachineView const *)mv
            << std::endl; */
  // int num_active_tokens = bc->num_active_tokens();
  IndexLauncher launcher(EXPERTS_INF_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
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
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(4, FID_DATA);
  if (use_bias) {
    launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[1]->region));
    launcher.add_field(5, FID_DATA);
  }
  return runtime->execute_index_space(ctx, launcher);
}

void Experts::inference_task(Task const *task,
                             std::vector<PhysicalRegion> const &regions,
                             Context ctx,
                             Runtime *runtime) {
  assert(regions.size() == task->regions.size());

  ExpertsMeta *m = *((ExpertsMeta **)task->local_args);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_tokens == 0) {
    return;
  }

  int num_experts = m->num_experts;
  bool use_bias = m->use_bias;
  assert(regions.size() - 4 == (1 + use_bias));

  // get input, indices, topk_gate_preds, outputs
  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      DT_FLOAT, regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR indices = helperGetGenericTensorAccessorRO(
      DT_INT32, regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorR topk_gate_preds = helperGetGenericTensorAccessorRO(
      DT_FLOAT, regions[2], task->regions[2], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      DT_FLOAT, regions[3], task->regions[3], FID_DATA, ctx, runtime);

  float const *input_ptr = input.get_float_ptr();
  int const *indices_ptr = indices.get_int32_ptr();
  float const *topk_gate_pred_ptr = topk_gate_preds.get_float_ptr();
  float *output_ptr = output.get_float_ptr();
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
  int output_dims = output_domain.get_dim();
  assert(input_dims == indices_dims);
  assert(indices_dims == topk_gate_pred_dims);
  assert(input_dims == output_dims);

  int replica_dim = input_dims - 1;
  int samples_index = input_dims - 2;

  coord_t data_dim = input_domain.hi()[0] - input_domain.lo()[0] + 1;
  coord_t batch_size =
      input_domain.hi()[samples_index] - input_domain.lo()[samples_index] + 1;
  coord_t chosen_experts = indices_domain.hi()[0] - indices_domain.lo()[0] + 1;
  coord_t out_dim = output_domain.hi()[0] - output_domain.lo()[0] + 1;
  coord_t num_replicas =
      input_domain.hi()[replica_dim] - input_domain.lo()[replica_dim] + 1;
  assert(data_dim == m->data_dim);
  assert(out_dim == m->out_dim);
  assert(chosen_experts == m->num_chosen_experts);
  assert(chosen_experts ==
         topk_gate_pred_domain.hi()[0] - topk_gate_pred_domain.lo()[0] + 1);

  for (int i = 1; i < input_dims; i++) {
    int a = input_domain.hi()[i] - input_domain.lo()[i] + 1;
    int b = indices_domain.hi()[i] - indices_domain.lo()[i] + 1;
    int c = topk_gate_pred_domain.hi()[i] - topk_gate_pred_domain.lo()[i] + 1;
    assert(a == b && b == c);
    if (i >= 1 && i < samples_index) {
      batch_size *= a;
    }
  }
  assert(batch_size == m->effective_batch_size);

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
  float const *weights_ptr = helperGetTensorPointerRO<float>(
      regions[4], task->regions[4], FID_DATA, ctx, runtime);
  assert(weights_ptr != nullptr);
  Domain weights_domain = runtime->get_index_space_domain(
      ctx, task->regions[4].region.get_index_space());
  int weights_dims = weights_domain.get_dim();
  assert(weights_dims == 3);
  int nparams_weight =
      (m->experts_num_layers == 1)
          ? (data_dim * out_dim)
          : m->experts_internal_dim_size * (data_dim + out_dim);
  assert(weights_domain.hi()[0] - weights_domain.lo()[0] + 1 == nparams_weight);
  assert(weights_domain.hi()[1] - weights_domain.lo()[1] + 1 == num_experts);
  assert(weights_domain.hi()[2] - weights_domain.lo()[2] + 1 == num_replicas);

  float const *bias_ptr = nullptr;
  int nparams_bias = -1;
  if (use_bias) {
    bias_ptr = helperGetTensorPointerRO<float>(
        regions[5], task->regions[5], FID_DATA, ctx, runtime);
    Domain bias_domain = runtime->get_index_space_domain(
        ctx, task->regions[5].region.get_index_space());
    int bias_dims = bias_domain.get_dim();
    assert(bias_dims == 3);
    nparams_bias = (m->experts_num_layers == 1)
                       ? out_dim
                       : (m->experts_internal_dim_size + out_dim);
    assert(bias_domain.hi()[0] - bias_domain.lo()[0] + 1 == nparams_bias);
    assert(bias_domain.hi()[1] - bias_domain.lo()[1] + 1 == num_experts);
    assert(bias_domain.hi()[2] - bias_domain.lo()[2] + 1 == num_replicas);
  }

#ifdef INFERENCE_TESTS
  if (DEBUG_MODE) {
    std::cout << "forward_kernel_wrapper" << std::endl
              << "-------------------------------" << std::endl;
    std::cout << m->data_dim << std::endl;
    std::cout << m->out_dim << std::endl;
    std::cout << m->num_chosen_experts << std::endl;
    std::cout << m->effective_batch_size << std::endl;
    std::cout << m->experts_num_layers << std::endl;
    std::cout << m->experts_internal_dim_size << std::endl;
    std::cout << m->num_experts << std::endl;
    std::cout << m->use_bias << std::endl;

    /* ----------------Input Token--------------*/
    float *cpu_input_ptr = new float[data_dim];
    checkCUDA(cudaMemcpy(cpu_input_ptr,
                         input_ptr,
                         data_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));

    srand(42);
    float cpu_sum = 0;
    for (int i = 0; i < data_dim; i++) {
      // cpu_input_ptr[i] = (float)rand() / (float)RAND_MAX;
      cpu_input_ptr[i] = float(i) / (float)data_dim;
      cpu_sum += cpu_input_ptr[i];
    }
    std::cout << "[CPU] Token 0 sum = " << cpu_sum << std::endl;
    std::cout << "Total token number = " << batch_size << std::endl;
    for (int i = 0; i < batch_size; i++) {
      checkCUDA(cudaMemcpy((float *)(input_ptr + i * data_dim),
                           cpu_input_ptr,
                           data_dim * sizeof(float),
                           cudaMemcpyHostToDevice));
    }
    free(cpu_input_ptr);

    /* ----------------indices--------------*/
    int *cpu_indices_ptr = new int[chosen_experts * batch_size];
    checkCUDA(cudaMemcpy(cpu_indices_ptr,
                         indices_ptr,
                         chosen_experts * batch_size * sizeof(int),
                         cudaMemcpyDeviceToHost));
    for (int i = 0; i < chosen_experts * 10; i++) {
      if (i % 2 == 1) {
        cpu_indices_ptr[i] += chosen_experts;
      }
    }
    checkCUDA(cudaMemcpy((int *)indices_ptr,
                         cpu_indices_ptr,
                         chosen_experts * batch_size * sizeof(int),
                         cudaMemcpyHostToDevice));
    free(cpu_indices_ptr);

    /* ----------------coefficient--------------*/
    float *cpu_topk_gate_pred_ptr = new float[chosen_experts * batch_size];
    checkCUDA(cudaMemcpy(cpu_topk_gate_pred_ptr,
                         topk_gate_pred_ptr,
                         chosen_experts * batch_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    for (int i = 0; i < chosen_experts * batch_size; i++) {
      if (i % 2 == 0) {
        cpu_topk_gate_pred_ptr[i] = 0.5;
      } else {
        cpu_topk_gate_pred_ptr[i] = 0.1;
      }
    }
    checkCUDA(cudaMemcpy((float *)topk_gate_pred_ptr,
                         cpu_topk_gate_pred_ptr,
                         chosen_experts * batch_size * sizeof(float),
                         cudaMemcpyHostToDevice));
    free(cpu_topk_gate_pred_ptr);

    /* ----------------Expert Weights--------------*/
    assert(m->experts_num_layers == 2 || m->experts_num_layers == 1);
    size_t layer0_size = m->experts_num_layers == 1
                             ? data_dim * out_dim
                             : data_dim * m->experts_internal_dim_size;
    size_t layer1_size = m->experts_internal_dim_size * out_dim;
    float *cpu_experts_0_layer0 = new float[layer0_size];
    float *cpu_experts_1_layer0 = new float[layer0_size];
    float *cpu_experts_0_layer1 =
        m->experts_num_layers == 1 ? nullptr : new float[layer1_size];
    float *cpu_experts_1_layer1 =
        m->experts_num_layers == 1 ? nullptr : new float[layer1_size];
    /*checkCUDA(cudaMemcpy(cpu_experts_0_layer0,
                         weights_ptr,
                         layer0_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(cpu_experts_1_layer0,
                         weights_ptr[nparams_weight],
                         layer0_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    if (m->experts_num_layers == 2) {
      checkCUDA(cudaMemcpy(cpu_experts_0_layer1,
                         weights_ptr[layer0_size],
                         layer1_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
      checkCUDA(cudaMemcpy(cpu_experts_1_layer1,
                           weights_ptr[nparams_weight + layer0_size],
                           layer1_size * sizeof(float),
                           cudaMemcpyDeviceToHost));
    }*/
    cpu_sum = 0;
    for (int i = 0; i < layer0_size; i++) {
      cpu_experts_0_layer0[i] = float(i) / float(nparams_weight);
      cpu_sum += cpu_experts_0_layer0[i];
    }
    if (m->experts_num_layers == 2) {
      for (int i = 0; i < layer1_size; i++) {
        cpu_experts_0_layer1[i] =
            float(layer0_size + i) / float(nparams_weight);
        cpu_sum += cpu_experts_0_layer1[i];
      }
    }
    std::cout << "[CPU] Experts 0 weights sum = " << cpu_sum << std::endl;

    cpu_sum = 0;
    for (int i = 0; i < layer0_size; i++) {
      cpu_experts_1_layer0[i] =
          float(nparams_weight - i) / float(nparams_weight);
      assert(cpu_experts_1_layer0[i] > 0);
      cpu_sum += cpu_experts_1_layer0[i];
    }
    if (m->experts_num_layers == 2) {
      for (int i = 0; i < layer1_size; i++) {
        cpu_experts_1_layer1[i] =
            float(nparams_weight - layer0_size + i) / float(nparams_weight);
        assert(cpu_experts_1_layer1[i] > 0);
        cpu_sum += cpu_experts_1_layer1[i];
      }
    }
    std::cout << "[CPU] Experts 1 weights sum = " << cpu_sum << std::endl;

    for (int i = 0; i < num_experts; i++) {
      // first layer
      checkCUDA(
          cudaMemcpy((float *)&weights_ptr[nparams_weight * i],
                     i % 2 == 0 ? cpu_experts_0_layer0 : cpu_experts_1_layer0,
                     layer0_size * sizeof(float),
                     cudaMemcpyHostToDevice));
      // second layer
      if (m->experts_num_layers == 2) {
        checkCUDA(
            cudaMemcpy((float *)&weights_ptr[nparams_weight * i + layer0_size],
                       i % 2 == 0 ? cpu_experts_0_layer1 : cpu_experts_1_layer1,
                       layer1_size * sizeof(float),
                       cudaMemcpyHostToDevice));
      }
    }
    free(cpu_experts_0_layer0);
    free(cpu_experts_1_layer0);
    free(cpu_experts_0_layer1);
    free(cpu_experts_1_layer1);

    /* ----------------Expert Bias--------------*/
    if (use_bias) {
      size_t layer0_size =
          m->experts_num_layers == 1 ? out_dim : m->experts_internal_dim_size;
      size_t layer1_size = out_dim;
      float *bias_experts_0_layer0 = new float[layer0_size];
      float *bias_experts_0_layer1 =
          m->experts_num_layers == 1 ? nullptr : new float[layer1_size];

      checkCUDA(cudaMemcpy(bias_experts_0_layer0,
                           bias_ptr,
                           layer0_size * sizeof(float),
                           cudaMemcpyDeviceToHost));
      cpu_sum = 0;
      for (int i = 0; i < layer0_size; i++) {
        cpu_sum += bias_experts_0_layer0[i];
        // bias_experts_1[i] = 1.0f;
      }
      std::cout << "[CPU] Bias expert 0 (layer 0) sum = " << cpu_sum
                << std::endl;

      if (m->experts_num_layers == 2) {
        checkCUDA(cudaMemcpy(bias_experts_0_layer1,
                             (float *)&bias_ptr[layer0_size],
                             layer1_size * sizeof(float),
                             cudaMemcpyDeviceToHost));
        cpu_sum = 0;
        for (int i = 0; i < layer1_size; i++) {
          cpu_sum += bias_experts_0_layer1[i];
          // bias_experts_1[i] = 1.0f;
        }
        std::cout << "[CPU] Bias expert 0 (layer 1) sum = " << cpu_sum
                  << std::endl;
      }

      for (int i = 0; i < num_experts; i++) {
        checkCUDA(cudaMemcpy((float *)&bias_ptr[nparams_bias * i],
                             bias_experts_0_layer0,
                             layer0_size * sizeof(float),
                             cudaMemcpyHostToDevice));
        if (m->experts_num_layers == 2) {
          checkCUDA(
              cudaMemcpy((float *)&bias_ptr[nparams_bias * i + layer0_size],
                         bias_experts_0_layer1,
                         layer1_size * sizeof(float),
                         cudaMemcpyHostToDevice));
        }
      }
      free(bias_experts_0_layer0);
      free(bias_experts_0_layer1);
    }
  }
#endif
  Experts::forward_kernel_wrapper(m,
                                  input_ptr,
                                  indices_ptr,
                                  topk_gate_pred_ptr,
                                  output_ptr,
                                  weights_ptr,
                                  bias_ptr,
                                  bc->num_active_tokens(),
                                  chosen_experts,
                                  batch_size,
                                  out_dim);
#ifdef INFERENCE_TESTS
  if (DEBUG_MODE) {
    /* ----------------Output after computation--------------*/
    float *cpu_output_ptr = new float[batch_size * out_dim];
    float cpu_sum = 0;
    checkCUDA(cudaMemcpy(cpu_output_ptr,
                         output_ptr,
                         batch_size * out_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    for (int j = 0; j < batch_size * out_dim; j += out_dim) {
      cpu_sum = 0;
      for (int i = 0; i < out_dim; i++) {
        cpu_sum += cpu_output_ptr[j + i];
      }
      // if ((j/out_dim) < 50) std::cout << "[CPU] output " << (j/out_dim) << "
      // sum = " << cpu_sum << std::endl;
      if (cpu_sum > 0.0f) {
        std::cout << "[CPU] output " << (j / out_dim) << " sum = " << cpu_sum
                  << std::endl;
      }
    }
    std::cout << "[CPU] output 0's 10th element = " << cpu_output_ptr[10]
              << std::endl;
    std::cout << "[CPU] output 0's 99th element = " << cpu_output_ptr[99]
              << std::endl;
    std::cout << "[CPU] output 0's 123th element = " << cpu_output_ptr[123]
              << std::endl;

    /* refrence output */
    /*
     * Input token sum = 391.5
     * Expert 0 weights sum = 307327.5
     * Expert 1 weights sum = 307328.47
     *  ------------------
     * experts 0's reulst = 153533.1
     * experts 1's reulst = 153402.9
     * Aggreated Result = 92106.836
     * 10th element = 41.28053
     * 99th element = 59.057823
     * 123th element = 63.8517
     */

    free(cpu_output_ptr);
  }
#endif

  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    Experts::save_inference_tensors_to_file(
        m, shard_id, bc, {input, indices, topk_gate_preds}, {}, {output});
  }
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

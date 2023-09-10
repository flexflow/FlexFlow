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

#include "flexflow/ops/add_bias_residual_layer_norm.h"
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

bool operator==(AddBiasResidualLayerNormParams const &lhs,
                AddBiasResidualLayerNormParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid && lhs.axes == rhs.axes &&
         lhs.elementwise_affine == rhs.elementwise_affine &&
         lhs.use_bias == rhs.use_bias;
}

bool AddBiasResidualLayerNormParams::is_valid(
    ParallelTensorShape const &input) const {
  return input.is_valid();
}

AddBiasResidualLayerNormParams AddBiasResidualLayerNorm::get_params() const {
  AddBiasResidualLayerNormParams params;
  params.layer_guid = this->layer_guid;
  params.axes = this->axes;
  params.elementwise_affine = this->elementwise_affine;
  params.eps = this->eps;
  params.use_bias = this->use_bias;
  return params;
}

Tensor FFModel::add_bias_residual_layer_norm(const Tensor input,
                                             std::vector<int> const &axes,
                                             bool elementwise_affine,
                                             float eps,
                                             bool use_bias,
                                             DataType data_type,
                                             char const *name) {
  // In PyTorch, axes must be the sizes of the last axes.size() dimensions of
  // the input tensor. However, since the tensor dimensions are reversed in
  // FlexFlow (batch size is the last dimension), we require that axes must be
  // the sizes of the FIRST axes.size() dimensions of the input tensor.

  // Another difference is that in PyTorch, the axes vector should contain the
  // sizes of the dimensions with respect to which you want to compute the
  // layernorm. In FlexFlow, instead, axes should contain the INDICES of the
  // dimensions in question. We do this because the size of a dimension might be
  // different when splitting a tensor in model parallelism.
  assert(
      axes.size() <= input->num_dims &&
      "number of axes must be less than tensor dimensions"); // input does not
                                                             // have replica
                                                             // dimension here
  for (int i = 0; i < axes.size(); i++) {
    assert(axes[i] == i && "axes must be the first axes.size() dimensions");
  }

  if (data_type == DT_NONE) {
    data_type = input->data_type;
  }
  int num_weights = elementwise_affine ? (use_bias ? 2 : 1) : 0;
  Layer *ln = nullptr;
  if (data_type != input->data_type) {
    Tensor casted_input =
        cast(input, data_type, "type cast for add_bias_residual_layer_norm");
    ln = new Layer(this,
                   OP_ADD_BIAS_RESIDUAL_LAYERNORM,
                   data_type,
                   name,
                   1 /*inputs*/,
                   num_weights,
                   1 /*outputs*/,
                   casted_input);
  } else {
    ln = new Layer(this,
                   OP_ADD_BIAS_RESIDUAL_LAYERNORM,
                   data_type,
                   name,
                   1 /*inputs*/,
                   num_weights,
                   1 /*outputs*/,
                   input);
  }

  ln->outputs[0] = create_tensor_legion_ordering(input->num_dims,
                                                 input->dims,
                                                 input->data_type,
                                                 ln,
                                                 0,
                                                 true /*create_grad*/);
  if (num_weights > 0) {
    int numdims = axes.size();
    int dims[numdims];
    for (int i = 0; i < numdims; i++) {
      dims[i] = input->dims[axes[i]];
    }
    ln->weights[0] = create_weight_legion_ordering(numdims,
                                                   dims,
                                                   input->data_type,
                                                   ln,
                                                   true /*create_grad*/,
                                                   nullptr,
                                                   CHOSEN_SYNC_TYPE);
    if (num_weights == 2) {
      ln->weights[1] = create_weight_legion_ordering(numdims,
                                                     dims,
                                                     input->data_type,
                                                     ln,
                                                     true /*create_grad*/,
                                                     nullptr,
                                                     CHOSEN_SYNC_TYPE);
    }
  }
  ln->add_int_property("elementwise_affine", elementwise_affine);
  ln->add_int_property("use_bias", use_bias);
  ln->add_int_vector_property("axes", axes);
  ln->add_float_property("eps", eps);
  layers.push_back(ln);
  return ln->outputs[0];
}

Op *AddBiasResidualLayerNorm::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("elementwise_affine", value);
  bool elementwise_affine = (bool)value;
  layer->get_int_property("use_bias", value);
  bool use_bias = (bool)value;
  std::vector<int> axes;
  layer->get_int_vector_property("axes", axes);
  float eps;
  layer->get_float_property("eps", eps);
  return new AddBiasResidualLayerNorm(model,
                                      layer->layer_guid,
                                      inputs[0],
                                      axes,
                                      elementwise_affine,
                                      use_bias,
                                      eps,
                                      false, // allocate_weights
                                      layer->name);
}

AddBiasResidualLayerNorm::AddBiasResidualLayerNorm(
    FFModel &model,
    AddBiasResidualLayerNormParams const &params,
    ParallelTensor const input,
    char const *name,
    bool allocate_weights)
    : AddBiasResidualLayerNorm(model,
                               params.layer_guid,
                               input,
                               params.axes,
                               params.elementwise_affine,
                               params.use_bias,
                               params.eps,
                               allocate_weights,
                               name) {}

AddBiasResidualLayerNorm::AddBiasResidualLayerNorm(
    FFModel &model,
    LayerID const &_layer_guid,
    const ParallelTensor _input,
    std::vector<int> const &_axes,
    bool _elementwise_affine,
    bool _use_bias,
    float _eps,
    bool allocate_weights,
    char const *name)
    : Op(model,
         OP_ADD_BIAS_RESIDUAL_LAYERNORM,
         _input->data_type,
         name,
         1 /*inputs*/,
         _elementwise_affine ? (_use_bias ? 2 : 1) : 0 /*weights*/,
         1 /*outputs*/,
         _input),
      elementwise_affine(_elementwise_affine), eps(_eps), axes(_axes),
      use_bias(_use_bias) {
  // overwrite layer_guid
  layer_guid = _layer_guid;
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      _input->num_dims, _input->dims, _input->data_type, this);
  assert(check_output_input_weight_parallel_dims(allocate_weights));
  ParallelDim output_dims[MAX_TENSOR_DIM];
  int M = 1;
  for (int i = 0; i < axes.size(); i++) {
    M *= inputs[0]->dims[axes[i]].size;
  }
  int num_replicas = 1;
  for (int i = 0; i < inputs[0]->num_dims; i++) {
    if (inputs[0]->dims[i].is_replica_dim) {
      num_replicas *= inputs[0]->dims[i].size;
    }
  }
  effective_num_elements = M;
  effective_batch_size = (inputs[0]->get_volume() / num_replicas) / M;
  assert(use_bias == (numWeights == 2));
  if (numWeights > 0 && allocate_weights) {
    ParallelTensorShape beta_gamma_shape = _input->get_shape();
    for (int i = axes.size(); i < beta_gamma_shape.num_dims - 1; i++) {
      beta_gamma_shape.dims[i].size = 1;
    }
    int seed = std::rand();
    Initializer *gamma_initializer = new UniformInitializer(seed, 1.0f, 1.0f);
    Initializer *beta_initializer = new UniformInitializer(seed, 0.0f, 0.0f);
    weights[0] = model.create_parallel_weight_legion_ordering(
        beta_gamma_shape.num_dims, // axes.size(),
        beta_gamma_shape.dims,
        _input->data_type,
        NULL /*owner_op*/,
        true /*create_grad*/,
        gamma_initializer,
        CHOSEN_SYNC_TYPE);
    weights[1] = model.create_parallel_weight_legion_ordering(
        beta_gamma_shape.num_dims, //.size(),
        beta_gamma_shape.dims,
        _input->data_type,
        NULL /*owner_op*/,
        true /*create_grad*/,
        beta_initializer,
        CHOSEN_SYNC_TYPE);
  }
}

void AddBiasResidualLayerNorm::init_inference(
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
  IndexLauncher launcher(LAYERNORM_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(AddBiasResidualLayerNorm)),
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
  if (elementwise_affine) {
    launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[0]->region));
    launcher.add_field(2, FID_DATA);

    if (use_bias) {
      launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                        0 /*projection id*/,
                                                        READ_ONLY,
                                                        EXCLUSIVE,
                                                        weights[1]->region));
      launcher.add_field(3, FID_DATA);
    }
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

void AddBiasResidualLayerNorm::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(LAYERNORM_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(AddBiasResidualLayerNorm)),
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
  if (elementwise_affine) {
    launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[0]->region));
    launcher.add_field(2, FID_DATA);
    if (use_bias) {
      launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                        0 /*projection id*/,
                                                        READ_ONLY,
                                                        EXCLUSIVE,
                                                        weights[1]->region));
      launcher.add_field(3, FID_DATA);
    }
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta *AddBiasResidualLayerNorm::init_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  AddBiasResidualLayerNorm *ln = (AddBiasResidualLayerNorm *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .only_kind(Memory::GPU_FB_MEM)
                       .best_affinity_to(task->target_proc)
                       .first();
  MemoryAllocator gpu_mem_allocator(gpu_mem);
  AddBiasResidualLayerNormMeta *meta =
      new AddBiasResidualLayerNormMeta(handle, ln, gpu_mem_allocator);
  meta->input_type[0] = ln->inputs[0]->data_type;
  meta->output_type[0] = ln->outputs[0]->data_type;
  return meta;
}

void AddBiasResidualLayerNorm::forward(FFModel const &ff) {
  assert(false);
}

void AddBiasResidualLayerNorm::backward(FFModel const &ff) {
  assert(false);
}

FutureMap AddBiasResidualLayerNorm::inference(
    FFModel const &ff,
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
  /* std::cout << "AddBiasResidualLayerNorm op machine_view: " << *(MachineView
     const *)mv
            << std::endl; */
  IndexLauncher launcher(LAYERNORM_FWD_TASK_ID,
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
  if (elementwise_affine) {
    launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[0]->region));
    launcher.add_field(2, FID_DATA);

    if (use_bias) {
      launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                        0 /*projection id*/,
                                                        READ_ONLY,
                                                        EXCLUSIVE,
                                                        weights[1]->region));
      launcher.add_field(3, FID_DATA);
    }
  }
  return runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input
  regions[1](O): output
  regions[2](I/O): gamma
  regions[3](I/O): beta
*/
void AddBiasResidualLayerNorm::inference_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  AddBiasResidualLayerNormMeta const *m =
      *((AddBiasResidualLayerNormMeta **)task->local_args);
  assert(task->regions.size() == regions.size());
  float const *in_ptr = NULL;
  float *out_ptr = NULL, *gamma_ptr = NULL, *beta_ptr = NULL;
  GenericTensorAccessorR in, gamma, beta;
  GenericTensorAccessorW out;

  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  // in_ptr = helperGetTensorPointerRO<float>(
  //     regions[0], task->regions[0], FID_DATA, ctx, runtime);
  in = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  // out_ptr = helperGetTensorPointerWO<float>(
  //     regions[1], task->regions[1], FID_DATA, ctx, runtime);
  out = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  assert(in_domain == out_domain);
  assert(in_domain.get_volume() ==
         m->effective_num_elements * m->effective_batch_size);
  if (m->elementwise_affine) {
    assert(m->use_bias == (regions.size() == 4));
    Domain gamma_domain = runtime->get_index_space_domain(
        ctx, task->regions[2].region.get_index_space());
    gamma = helperGetGenericTensorAccessorRO(
        m->input_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);
    if (m->use_bias) {
      Domain beta_domain = runtime->get_index_space_domain(
          ctx, task->regions[3].region.get_index_space());
      beta = helperGetGenericTensorAccessorRO(m->input_type[0],
                                              regions[3],
                                              task->regions[3],
                                              FID_DATA,
                                              ctx,
                                              runtime);
      assert(gamma_domain == beta_domain);
    }

    assert(gamma_domain.get_volume() == m->effective_num_elements);
    int numdims = gamma_domain.get_dim();
    size_t vol = 1;
    int i = 0;
    while (vol < gamma_domain.get_volume()) {
      int g_d = gamma_domain.hi()[i] - gamma_domain.lo()[i] + 1;
      int in_d = in_domain.hi()[i] - in_domain.lo()[i] + 1;
      assert(g_d == in_d);
      vol *= g_d;
      i++;
    }
  } else {
    assert(regions.size() == 2);
  }
  AddBiasResidualLayerNorm::inference_kernel_wrapper(m, in, out, gamma, beta);
}

bool AddBiasResidualLayerNorm::measure_operator_cost(
    Simulator *sim, MachineView const &mv, CostMetrics &cost_metrics) const {
  return false;
}

void AddBiasResidualLayerNorm::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->axes.size());
  for (size_t i = 0; i < this->axes.size(); i++) {
    sez.serialize(this->axes[i]);
  }
  sez.serialize(this->elementwise_affine);
  sez.serialize(this->eps);
  sez.serialize(this->use_bias);
}

using PCG::Node;
/*static*/
Node AddBiasResidualLayerNorm::deserialize(FFModel &ff,
                                           Legion::Deserializer &dez,
                                           ParallelTensor inputs[],
                                           int num_inputs) {
  assert(num_inputs == 1);
  size_t num_axes;
  std::vector<int> axes;
  bool elementwise_affine;
  bool use_bias;
  float eps;
  size_t id, transformer_layer_id;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  LayerID layer_guid(id, transformer_layer_id);
  dez.deserialize(num_axes);
  for (size_t i = 0; i < num_axes; i++) {
    int axis_idx;
    dez.deserialize(axis_idx);
    axes.push_back(axis_idx);
  }
  dez.deserialize(elementwise_affine);
  dez.deserialize(eps);
  dez.deserialize(use_bias);

  AddBiasResidualLayerNormParams params;
  params.layer_guid = layer_guid;
  params.axes = axes;
  params.elementwise_affine = elementwise_affine;
  params.eps = eps;
  params.use_bias = use_bias;
  return ff.get_or_create_node<AddBiasResidualLayerNorm>(inputs[0], params);
}

Op *AddBiasResidualLayerNorm::materialize(FFModel &ff,
                                          ParallelTensor inputs[],
                                          int num_inputs) const {
  AddBiasResidualLayerNormParams params = get_params();
  return new AddBiasResidualLayerNorm(
      ff, params, inputs[0], this->name, true /*allocate_weights*/);
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::AddBiasResidualLayerNormParams>::operator()(
    FlexFlow::AddBiasResidualLayerNormParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.axes.size());
  for (int n : params.axes) {
    hash_combine(key, n);
  }
  hash_combine(key, params.elementwise_affine);
  hash_combine(key, params.use_bias);
  return key;
}
}; // namespace std

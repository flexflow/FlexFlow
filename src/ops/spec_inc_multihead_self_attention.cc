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

#include "flexflow/ops/spec_inc_multihead_self_attention.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/model.h"
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include "flexflow/utils/cuda_helper.h"
#else
#include "flexflow/utils/hip_helper.h"
#endif
#include "flexflow/utils/hash_utils.h"
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

bool SpecIncMultiHeadSelfAttentionParams::is_valid(
    ParallelTensorShape const &input) const {
  bool is_valid = input.is_valid();
  return is_valid;
}

Tensor
    FFModel::spec_inc_multihead_self_attention(Tensor const input,
                                               int embed_dim,
                                               int num_heads,
                                               int kdim,
                                               int vdim,
                                               float dropout,
                                               bool qkv_bias,
                                               bool final_bias,
                                               bool add_zero_attn,
                                               DataType data_type,
                                               Initializer *kernel_initializer,
                                               bool apply_rotary_embedding,
                                               bool scaling_query,
                                               float scaling_factor,
                                               bool qk_prod_scaling,
                                               bool position_bias,
                                               char const *name) {
  return spec_inc_multiquery_self_attention(input,
                                            embed_dim,
                                            num_heads,
                                            num_heads,
                                            kdim,
                                            vdim,
                                            dropout,
                                            qkv_bias,
                                            final_bias,
                                            add_zero_attn,
                                            data_type,
                                            kernel_initializer,
                                            apply_rotary_embedding,
                                            scaling_query,
                                            scaling_factor,
                                            qk_prod_scaling,
                                            position_bias,
                                            name);
}

Tensor
    FFModel::spec_inc_multiquery_self_attention(Tensor const input,
                                                int embed_dim,
                                                int num_q_heads,
                                                int num_kv_heads,
                                                int kdim,
                                                int vdim,
                                                float dropout,
                                                bool qkv_bias,
                                                bool final_bias,
                                                bool add_zero_attn,
                                                DataType data_type,
                                                Initializer *kernel_initializer,
                                                bool apply_rotary_embedding,
                                                bool scaling_query,
                                                float scaling_factor,
                                                bool qk_prod_scaling,
                                                bool position_bias,
                                                char const *name) {
  if (data_type == DT_NONE) {
    data_type = input->data_type;
  }
  Layer *li = nullptr;
  int weight_num = (qkv_bias || final_bias) ? 2 : 1;
  if (data_type != input->data_type) {
    Tensor casted_input = cast(input, data_type, "type cast for IncMHA");
    li = new Layer(this,
                   OP_SPEC_INC_MULTIHEAD_SELF_ATTENTION,
                   data_type,
                   name,
                   1 /*inputs*/,
                   weight_num /*weights*/,
                   1 /*outputs*/,
                   casted_input);
  } else {
    li = new Layer(this,
                   OP_SPEC_INC_MULTIHEAD_SELF_ATTENTION,
                   data_type,
                   name,
                   1 /*inputs*/,
                   weight_num /*weights*/,
                   1 /*outputs*/,
                   input);
  }
  {
    int numdims = input->num_dims;
    int dims[MAX_TENSOR_DIM];
    for (int i = 0; i < numdims; i++) {
      dims[i] = input->dims[i];
    }
    dims[0] = embed_dim;
    li->outputs[0] = create_tensor_legion_ordering(
        numdims, dims, data_type, li, 0, true /*create_grad*/);
  }
  // Compute weight size
  int qProjSize = kdim, kProjSize = kdim, vProjSize = kdim,
      oProjSize = embed_dim;
  int qSize = input->dims[0], kSize = input->dims[0], vSize = input->dims[0];
  int qParas = qProjSize * qSize;
  int kParas = kProjSize * kSize;
  int vParas = vProjSize * vSize;
  int oParas = oProjSize * (vProjSize > 0 ? vProjSize : vSize);
  int weight_size = qParas * num_q_heads + kParas * num_q_heads +
                    vParas * num_q_heads + oParas * num_q_heads;
  {
    int dims[1] = {weight_size};
    li->weights[0] = create_weight_legion_ordering(1,
                                                   dims,
                                                   data_type,
                                                   li,
                                                   true /*create_grad*/,
                                                   kernel_initializer,
                                                   CHOSEN_SYNC_TYPE);
  }
  if (qkv_bias || final_bias) {
    // q, k, v, o
    int qkv_bias_size =
        qProjSize * num_q_heads + (kProjSize + vProjSize) * num_q_heads;
    int dims[1] = {(qkv_bias ? qkv_bias_size : 0) +
                   (final_bias ? oProjSize : 0)};
    li->weights[1] = create_weight_legion_ordering(1,
                                                   dims,
                                                   data_type,
                                                   li,
                                                   true /*create_grad*/,
                                                   kernel_initializer,
                                                   CHOSEN_SYNC_TYPE);
  }
  li->data_type = data_type;
  li->add_int_property("embed_dim", embed_dim);
  li->add_int_property("num_q_heads", num_q_heads);
  li->add_int_property("num_kv_heads", num_kv_heads);
  li->add_int_property("kdim", kdim);
  li->add_int_property("vdim", vdim);
  li->add_int_property("qkv_bias", qkv_bias);
  li->add_int_property("final_bias", final_bias);
  li->add_int_property("add_zero_attn", add_zero_attn);
  li->add_float_property("dropout", dropout);
  li->add_int_property("apply_rotary_embedding", apply_rotary_embedding);
  li->add_int_property("scaling_query", scaling_query);
  li->add_float_property("scaling_factor", scaling_factor);
  li->add_int_property("qk_prod_scaling", qk_prod_scaling);
  li->add_int_property("position_bias", position_bias);
  layers.push_back(li);
  return li->outputs[0];
}

Op *SpecIncMultiHeadSelfAttention::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {

  std::cout << "spec create operator: " << layer->name << "\n";
  long long value;
  layer->get_int_property("embed_dim", value);
  int embed_dim = value;
  layer->get_int_property("num_q_heads", value);
  int num_q_heads = value;
  layer->get_int_property("num_kv_heads", value);
  int num_kv_heads = value;
  layer->get_int_property("kdim", value);
  int kdim = value;
  layer->get_int_property("vdim", value);
  int vdim = value;
  float dropout;
  layer->get_float_property("dropout", dropout);
  layer->get_int_property("qkv_bias", value);
  bool qkv_bias = (bool)value;
  layer->get_int_property("final_bias", value);
  bool final_bias = (bool)value;
  layer->get_int_property("add_zero_attn", value);
  bool add_zero_attn = (bool)value;
  layer->get_int_property("apply_rotary_embedding", value);
  bool apply_rotary_embedding = (bool)value;
  layer->get_int_property("scaling_query", value);
  bool scaling_query = (bool)value;
  float scaling_factor;
  layer->get_float_property("scaling_factor", scaling_factor);
  layer->get_int_property("qk_prod_scaling", value);
  bool qk_prod_scaling = (bool)value;
  layer->get_int_property("position_bias", value);
  bool position_bias = (bool)value;

  return new SpecIncMultiHeadSelfAttention(model,
                                           layer->layer_guid,
                                           inputs[0],
                                           embed_dim,
                                           num_q_heads,
                                           num_kv_heads,
                                           kdim,
                                           vdim,
                                           dropout,
                                           qkv_bias,
                                           final_bias,
                                           add_zero_attn,
                                           apply_rotary_embedding,
                                           scaling_query,
                                           scaling_factor,
                                           qk_prod_scaling,
                                           position_bias,
                                           false /*allocate_weights*/,
                                           layer->name);
}

SpecIncMultiHeadSelfAttention::SpecIncMultiHeadSelfAttention(
    FFModel &model,
    LayerID const &_layer_guid,
    ParallelTensor const _input,
    int _embed_dim,
    int _num_q_heads,
    int _num_kv_heads,
    int _kdim,
    int _vdim,
    float _dropout,
    bool _qkv_bias,
    bool _final_bias,
    bool _add_zero_attn,
    bool _apply_rotary_embedding,
    bool _scaling_query,
    float _scaling_factor,
    bool _qk_prod_scaling,
    bool _position_bias,
    bool allocate_weights,
    char const *name)
    // Initializer* _bias_initializer)
    : Op(model,
         OP_SPEC_INC_MULTIHEAD_SELF_ATTENTION,
         _input->data_type,
         name,
         1 /*inputs*/,
         (_qkv_bias || _final_bias ? 2 : 1) /*weights*/,
         1 /*outputs*/,
         _input),
      num_q_heads(_num_q_heads), num_kv_heads(_num_kv_heads), dropout(_dropout),
      qkv_bias(_qkv_bias), final_bias(_final_bias),
      add_zero_attn(_add_zero_attn),
      apply_rotary_embedding(_apply_rotary_embedding),
      qSize(_input->dims[0].size), kSize(_input->dims[0].size),
      vSize(_input->dims[0].size), qProjSize(_kdim), kProjSize(_kdim),
      vProjSize(_vdim), oProjSize(_embed_dim),
      qoSeqLength(_input->dims[1].size), kvSeqLength(_input->dims[1].size),
      scaling_query(_scaling_query), scaling_factor(_scaling_factor),
      qk_prod_scaling(_qk_prod_scaling), position_bias(_position_bias) {
  // overwrite layer_guid
  layer_guid = _layer_guid;

  numOutputs = 1;
  int numdim = _input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = _input->dims[i];
  }
  dims[0].size = _embed_dim;
  // Currently require no parallelism along this dim
  assert(dims[0].degree == 1);
  if (allocate_weights) {
    // Create weight tensor
    int num_dims = inputs[0]->num_dims;
    // Compute weight size
    int qParas = this->qProjSize * this->qSize;
    int kParas = this->kProjSize * this->kSize;
    int vParas = this->vProjSize * this->vSize;
    int oParas =
        this->oProjSize * (this->vProjSize > 0 ? this->vProjSize : this->vSize);
    ParallelDim dims[2];
    dims[0] = inputs[0]->dims[num_dims - 2];
    dims[0].size = dims[0].degree;
    dims[1] = inputs[0]->dims[num_dims - 1];
    dims[1].size = this->num_q_heads * (qParas + oParas) +
                   this->num_q_heads * (kParas + vParas);
    dims[1].is_replica_dim = false;
    int seed = std::rand();
    Initializer *initializer = new GlorotUniform(seed);
    weights[0] = model.create_parallel_weight<2>(dims,
                                                 this->data_type,
                                                 NULL /*owner_op*/,
                                                 true /*create_grad*/,
                                                 initializer,
                                                 CHOSEN_SYNC_TYPE);
    if (qkv_bias || final_bias) {
      ParallelTensorShape bias_shape = _input->get_shape();
      int qkv_bias_size =
          qProjSize * num_q_heads + (kProjSize + vProjSize) * num_q_heads;
      bias_shape.dims[0].size =
          (qkv_bias ? qkv_bias_size : 0) + (final_bias ? oProjSize : 0);
      bias_shape.dims[1].size = bias_shape.dims[2].size = 1;
      weights[1] =
          model.create_parallel_weight_legion_ordering(bias_shape.num_dims,
                                                       bias_shape.dims,
                                                       this->data_type,
                                                       nullptr /*owner_op*/,
                                                       true /*create_grad*/,
                                                       initializer,
                                                       CHOSEN_SYNC_TYPE);
    }
  }

  outputs[0] = model.create_parallel_tensor_legion_ordering(
      _input->num_dims, dims, this->data_type, this);
  /* for (int i = 0; i < numdim; i++) { */
  /*   register_output_input_parallel_dims(outputs[0], i, inputs[0], i); */
  /* } */
  /* // Check correctness */
  /* assert(check_output_input_weight_parallel_dims()); */
}

SpecIncMultiHeadSelfAttention::SpecIncMultiHeadSelfAttention(
    FFModel &model,
    ParallelTensor const _input,
    ParallelTensor const _weight,
    int _embed_dim,
    int _num_q_heads,
    int _num_kv_heads,
    int _kdim,
    int _vdim,
    float _dropout,
    bool _qkv_bias,
    bool _final_bias,
    bool _add_zero_attn,
    bool _apply_rotary_embedding,
    bool _scaling_query,
    float _scaling_factor,
    bool _qk_prod_scaling,
    bool _position_bias,
    bool allocate_weights,
    char const *name)
    // Initializer* _bias_initializer)
    : Op(model,
         OP_SPEC_INC_MULTIHEAD_SELF_ATTENTION,
         _input->data_type,
         name,
         1 /*inputs*/,
         (_qkv_bias || _final_bias ? 2 : 1) /*weights*/,
         1 /*outputs*/,
         _input,
         _weight),
      num_q_heads(_num_q_heads), num_kv_heads(_num_kv_heads), dropout(_dropout),
      qkv_bias(_qkv_bias), final_bias(_final_bias),
      add_zero_attn(_add_zero_attn),
      apply_rotary_embedding(_apply_rotary_embedding),
      qSize(_input->dims[0].size), kSize(_input->dims[0].size),
      vSize(_input->dims[0].size), qProjSize(_kdim), kProjSize(_kdim),
      vProjSize(_vdim), oProjSize(_embed_dim),
      qoSeqLength(_input->dims[1].size), kvSeqLength(_input->dims[1].size),
      scaling_query(_scaling_query), scaling_factor(_scaling_factor),
      qk_prod_scaling(_qk_prod_scaling), position_bias(_position_bias)
// bias_initializer(_bias_initializer)
{
  numOutputs = 1;
  int numdim = _input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++) {
    dims[i] = _input->dims[i];
  }
  dims[0].size = _embed_dim;
  // Currently require no parallelism along this dim
  assert(dims[0].degree == 1);
  if (allocate_weights) {
    // Create weight tensor
    int num_dims = inputs[0]->num_dims;
    // Compute weight size
    int qParas = this->qProjSize * this->qSize;
    int kParas = this->kProjSize * this->kSize;
    int vParas = this->vProjSize * this->vSize;
    int oParas =
        this->oProjSize * (this->vProjSize > 0 ? this->vProjSize : this->vSize);
    ParallelDim dims[2];
    dims[0] = inputs[0]->dims[num_dims - 2];
    dims[0].size = dims[0].degree;
    dims[1] = inputs[0]->dims[num_dims - 1];
    dims[1].size = this->num_q_heads * (qParas + oParas) +
                   this->num_q_heads * (kParas + vParas);
    dims[1].is_replica_dim = false;
    // dims[2].size = qParas + kParas + vParas + oParas;
    int seed = std::rand();
    Initializer *initializer = new GlorotUniform(seed);
    weights[0] = model.create_parallel_weight<2>(dims,
                                                 this->data_type,
                                                 NULL /*owner_op*/,
                                                 true /*create_grad*/,
                                                 initializer,
                                                 CHOSEN_SYNC_TYPE);
    if (qkv_bias || final_bias) {
      ParallelTensorShape bias_shape = _input->get_shape();
      int qkv_bias_size =
          qProjSize * num_q_heads + (kProjSize + vProjSize) * num_q_heads;
      bias_shape.dims[0].size =
          (qkv_bias ? qkv_bias_size : 0) + (final_bias ? oProjSize : 0);
      bias_shape.dims[1].size = bias_shape.dims[2].size = 1;
      weights[1] =
          model.create_parallel_weight_legion_ordering(bias_shape.num_dims,
                                                       bias_shape.dims,
                                                       this->data_type,
                                                       nullptr /*owner_op*/,
                                                       true /*create_grad*/,
                                                       initializer,
                                                       CHOSEN_SYNC_TYPE);
    }
  }

  outputs[0] = model.create_parallel_tensor_legion_ordering(
      _input->num_dims, dims, this->data_type, this);

  /* for (int i = 0; i < numdim; i++) { */
  /*   register_output_input_parallel_dims(outputs[0], i, inputs[0], i); */
  /* } */
  /* register_output_weight_parallel_dims(outputs[0], numdim-1, _weight, 1); */
  /* register_output_weight_parallel_dims(outputs[0], numdim-2, _weight, 2); */
  // Check correctness
  /* assert(check_output_input_weight_parallel_dims()); */
}

SpecIncMultiHeadSelfAttention::SpecIncMultiHeadSelfAttention(
    FFModel &model,
    SpecIncMultiHeadSelfAttention const &other,
    ParallelTensor const input,
    bool allocate_weights)
    : SpecIncMultiHeadSelfAttention(model,
                                    other.layer_guid,
                                    input,
                                    other.oProjSize,
                                    other.num_q_heads,
                                    other.num_kv_heads,
                                    other.qProjSize,
                                    other.vProjSize,
                                    other.dropout,
                                    other.qkv_bias,
                                    other.final_bias,
                                    other.add_zero_attn,
                                    other.apply_rotary_embedding,
                                    other.scaling_query,
                                    other.scaling_factor,
                                    other.qk_prod_scaling,
                                    other.position_bias,
                                    allocate_weights,
                                    other.name) {}

SpecIncMultiHeadSelfAttention::SpecIncMultiHeadSelfAttention(
    FFModel &model,
    SpecIncMultiHeadSelfAttentionParams const &params,
    ParallelTensor const &input,
    bool allocate_weights,
    char const *name)
    : SpecIncMultiHeadSelfAttention(model,
                                    params.layer_guid,
                                    input,
                                    params.embed_dim,
                                    params.num_q_heads,
                                    params.num_kv_heads,
                                    params.kdim,
                                    params.vdim,
                                    params.dropout,
                                    params.qkv_bias,
                                    params.final_bias,
                                    params.add_zero_attn,
                                    params.apply_rotary_embedding,
                                    params.scaling_query,
                                    params.scaling_factor,
                                    params.qk_prod_scaling,
                                    params.position_bias,
                                    allocate_weights,
                                    params.name) {}

void SpecIncMultiHeadSelfAttention::init_inference(
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
  IndexLauncher launcher(
      SPEC_INC_MULTIHEAD_SELF_ATTENTION_INIT_TASK_ID,
      parallel_is,
      TaskArgument(this, sizeof(SpecIncMultiHeadSelfAttention)),
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
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

void SpecIncMultiHeadSelfAttention::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(
      SPEC_INC_MULTIHEAD_SELF_ATTENTION_INIT_TASK_ID,
      parallel_is,
      TaskArgument(this, sizeof(SpecIncMultiHeadSelfAttention)),
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
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
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

/*
  regions[0](I): input
  regions[1](I): weight
  regions[2](O): output
*/
OpMeta *SpecIncMultiHeadSelfAttention::init_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  SpecIncMultiHeadSelfAttention const *attn =
      (SpecIncMultiHeadSelfAttention *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);

  GenericTensorAccessorR input =
      helperGetGenericTensorAccessorRO(attn->inputs[0]->data_type,
                                       regions[0],
                                       task->regions[0],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  GenericTensorAccessorR weight =
      helperGetGenericTensorAccessorRO(attn->weights[0]->data_type,
                                       regions[1],
                                       task->regions[1],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  GenericTensorAccessorW output =
      helperGetGenericTensorAccessorWO(attn->outputs[0]->data_type,
                                       regions[2],
                                       task->regions[2],
                                       FID_DATA,
                                       ctx,
                                       runtime);

  int num_samples = input.domain.hi()[2] - input.domain.lo()[2] + 1;
  assert(attn->qoSeqLength == input.domain.hi()[1] - input.domain.lo()[1] + 1);
  assert(attn->kvSeqLength == input.domain.hi()[1] - input.domain.lo()[1] + 1);
  int num_q_heads = attn->num_q_heads;
  int num_kv_heads = attn->num_kv_heads;
  assert(attn->oProjSize == output.domain.hi()[0] - output.domain.lo()[0] + 1);

  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .only_kind(Memory::GPU_FB_MEM)
                       .best_affinity_to(task->target_proc)
                       .first();
  MemoryAllocator gpu_mem_allocator(gpu_mem);
  // We don't do offloading for SSMs (small speculative models)
  SpecIncMultiHeadSelfAttentionMeta *m =
      new SpecIncMultiHeadSelfAttentionMeta(handle,
                                            attn,
                                            weight,
                                            gpu_mem_allocator,
                                            num_samples,
                                            num_q_heads,
                                            num_kv_heads);
  // assert that we didn't over allocate memory
  assert(gpu_mem_allocator.instance_allocated_size ==
         gpu_mem_allocator.instance_total_size);
  m->profiling = attn->profiling;
  m->inference_debugging = attn->inference_debugging;
  std::strcpy(m->op_name, attn->name);
  m->layer_guid = attn->layer_guid;
  assert(weight.domain.get_volume() * data_type_size(weight.data_type) ==
         m->weightSize);
  return m;
}

void SpecIncMultiHeadSelfAttention::forward(FFModel const &ff) {
  // SpecIncMultiHeadSelfAttention doesn't support forward
  assert(false);
}

FutureMap SpecIncMultiHeadSelfAttention::inference(
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
  int idx = 0;
  IndexLauncher launcher(SPEC_INC_MULTIHEAD_SELF_ATTENTION_INF_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
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
  launcher.add_field(idx++, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(idx++, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(idx++, FID_DATA);

  if (qkv_bias || final_bias) {
    launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[1]->region));
    launcher.add_field(idx++, FID_DATA);
  }
  return runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input
  regions[3](I): weight
  regions[4](O): output
*/
void SpecIncMultiHeadSelfAttention::inference_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(task->regions.size() == regions.size());

  BeamSearchBatchConfig const &bc =
      Future(task->futures[0]).get_result<BeamSearchBatchConfig>();
  if (bc.num_tokens == 0) {
    return;
  }

  SpecIncMultiHeadSelfAttentionMeta *m =
      *((SpecIncMultiHeadSelfAttentionMeta **)task->local_args);
  assert(((*m->qkv_bias || *m->final_bias) ? regions.size() == 4
                                           : regions.size() == 3));

  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR weight = helperGetGenericTensorAccessorRO(
      m->weight_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  GenericTensorAccessorR biases;
  if (*m->qkv_bias || *m->final_bias) {
    biases = helperGetGenericTensorAccessorRO(m->weight_type[1],
                                              regions[3],
                                              task->regions[3],
                                              FID_DATA,
                                              ctx,
                                              runtime);
    Domain bias_domain = runtime->get_index_space_domain(
        ctx, task->regions[3].region.get_index_space());
    assert(bias_domain.get_dim() == 4);
  }
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain weight_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());

  assert(input_domain.get_dim() == 4);
  assert(weight_domain.get_dim() == 2);
  assert(output_domain.get_dim() == 4);

  assert(task->index_point.get_dim() == 1);
  SpecIncMultiHeadSelfAttention::inference_kernel_wrapper(
      m, &bc, task->index_point.point_data[0], input, weight, output, biases);
  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];
    std::vector<GenericTensorAccessorR> weights_accessors;
    weights_accessors.push_back(weight);
    if (*m->qkv_bias || *m->final_bias) {
      weights_accessors.push_back(biases);
    }
    SpecIncMultiHeadSelfAttention::save_inference_tensors_to_file(
        m, shard_id, &bc, {input}, weights_accessors, {output});
  }
}

void SpecIncMultiHeadSelfAttention::backward(FFModel const &ff) {
  // SpecIncMultiHeadSelfAttention does not support backward
  assert(false);
}

bool SpecIncMultiHeadSelfAttention::get_int_parameter(PMParameter para,
                                                      int *value) const {
  switch (para) {
    case PM_NUM_HEADS:
      *value = num_q_heads;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

Op *SpecIncMultiHeadSelfAttention::materialize(FFModel &ff,
                                               ParallelTensor inputs[],
                                               int num_inputs) const {
  SpecIncMultiHeadSelfAttentionParams params = get_params();
  return new SpecIncMultiHeadSelfAttention(
      ff, params, inputs[0], true, this->name);
}

bool SpecIncMultiHeadSelfAttention::measure_operator_cost(
    Simulator *sim, MachineView const &mv, CostMetrics &cost_metrics) const {
  return false;
}

bool operator==(SpecIncMultiHeadSelfAttentionParams const &lhs,
                SpecIncMultiHeadSelfAttentionParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid && lhs.embed_dim == rhs.embed_dim &&
         lhs.num_q_heads == rhs.num_q_heads && lhs.kdim == rhs.kdim &&
         lhs.vdim == rhs.vdim && lhs.dropout == rhs.dropout &&
         lhs.qkv_bias == rhs.qkv_bias && lhs.final_bias == rhs.final_bias &&
         lhs.add_zero_attn == rhs.add_zero_attn &&
         lhs.apply_rotary_embedding == rhs.apply_rotary_embedding &&
         lhs.scaling_query == rhs.scaling_query &&
         lhs.scaling_factor == rhs.scaling_factor &&
         lhs.qk_prod_scaling == rhs.qk_prod_scaling &&
         lhs.position_bias == rhs.position_bias;
}

SpecIncMultiHeadSelfAttentionParams
    SpecIncMultiHeadSelfAttention::get_params() const {
  SpecIncMultiHeadSelfAttentionParams params;
  params.layer_guid = this->layer_guid;
  params.embed_dim = this->oProjSize;
  params.num_q_heads = this->num_q_heads;
  params.num_kv_heads = this->num_kv_heads;
  params.kdim = this->kProjSize;
  params.vdim = this->vProjSize;
  params.dropout = this->dropout;
  params.qkv_bias = this->qkv_bias;
  params.final_bias = this->final_bias;
  params.add_zero_attn = this->add_zero_attn;
  params.apply_rotary_embedding = this->apply_rotary_embedding;
  params.scaling_query = this->scaling_query;
  params.scaling_factor = this->scaling_factor;
  params.qk_prod_scaling = this->qk_prod_scaling;
  params.position_bias = this->position_bias;
  if (this->name != nullptr) {
    strcpy(params.name, this->name);
  }

  return params;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::SpecIncMultiHeadSelfAttentionParams>::operator()(
    FlexFlow::SpecIncMultiHeadSelfAttentionParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.embed_dim);
  hash_combine(key, params.num_q_heads);
  hash_combine(key, params.num_kv_heads);
  hash_combine(key, params.kdim);
  hash_combine(key, params.vdim);
  hash_combine(key, params.dropout);
  hash_combine(key, params.qkv_bias);
  hash_combine(key, params.final_bias);
  hash_combine(key, params.add_zero_attn);
  hash_combine(key, params.apply_rotary_embedding);
  hash_combine(key, params.scaling_query);
  hash_combine(key, params.scaling_factor);
  hash_combine(key, params.qk_prod_scaling);
  hash_combine(key, params.position_bias);
  return key;
}
}; // namespace std

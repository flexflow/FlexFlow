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

#include "flexflow/ops/tree_inc_multihead_self_attention.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/model.h"
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include "flexflow/utils/cuda_helper.h"
#else
#include "flexflow/utils/hip_helper.h"
#endif
#include "flexflow/utils/hash_utils.h"
#include "legion/legion_utilities.h"
#ifdef INFERENCE_TESTS
#include <torch/torch.h>
using namespace at::indexing;
#endif

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

LegionRuntime::Logger::Category log_tree_verify("TreeVerifyIncMHA");

bool TreeIncMultiHeadSelfAttentionParams::is_valid(
    ParallelTensorShape const &input) const {
  bool is_valid = input.is_valid();
  return is_valid;
}

Tensor FFModel::inc_multihead_self_attention_verify(
    const Tensor input,
    int embed_dim,
    int num_heads,
    int kdim,
    int vdim,
    float dropout,
    bool bias,
    bool add_bias_kv,
    bool add_zero_attn,
    DataType data_type,
    Initializer *kernel_initializer,
    bool apply_rotary_embedding,
    bool scaling_query,
    float scaling_factor,
    bool qk_prod_scaling,
    char const *name) {
  return inc_multiquery_self_attention_verify(input,
                                              embed_dim,
                                              num_heads,
                                              num_heads,
                                              kdim,
                                              vdim,
                                              dropout,
                                              bias,
                                              add_bias_kv,
                                              add_zero_attn,
                                              data_type,
                                              kernel_initializer,
                                              apply_rotary_embedding,
                                              scaling_query,
                                              scaling_factor,
                                              qk_prod_scaling,
                                              name);
}

Tensor FFModel::inc_multiquery_self_attention_verify(
    const Tensor input,
    int embed_dim,
    int num_q_heads,
    int num_kv_heads,
    int kdim,
    int vdim,
    float dropout,
    bool bias,
    bool add_bias_kv,
    bool add_zero_attn,
    DataType data_type,
    Initializer *kernel_initializer,
    bool apply_rotary_embedding,
    bool scaling_query,
    float scaling_factor,
    bool qk_prod_scaling,
    char const *name) {
  if (data_type == DT_NONE) {
    data_type = input->data_type;
  }
  DataType quantization_type = cpu_offload ? config.quantization_type : DT_NONE;
  bool offload = cpu_offload;
  Layer *li = nullptr;
  int weight_num = bias ? 2 : 1;
  if (data_type != input->data_type) {
    Tensor casted_input = cast(input, data_type, "type cast for IncMHA");
    li = new Layer(this,
                   OP_TREE_INC_MULTIHEAD_SELF_ATTENTION,
                   data_type,
                   name,
                   1 /*inputs*/,
                   weight_num /*weights*/,
                   1 /*outputs*/,
                   casted_input);
  } else {
    li = new Layer(this,
                   OP_TREE_INC_MULTIHEAD_SELF_ATTENTION,
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
  int one_head_size = qParas + kParas + vParas + oParas;
  int weight_size = qParas * num_q_heads + kParas * num_kv_heads +
                    vParas * num_kv_heads + oParas * num_q_heads;
  {
    // compress the weight size if quantization.
    if (quantization_type != DT_NONE) {
      one_head_size = get_quantization_to_byte_size(
          data_type, quantization_type, one_head_size);
    }

    int dims[1] = {weight_size};
    li->weights[0] = create_weight_legion_ordering(
        1,
        dims,
        quantization_type == DT_NONE ? data_type : quantization_type,
        li,
        true /*create_grad*/,
        kernel_initializer,
        CHOSEN_SYNC_TYPE);
  }
  if (bias) {
    // q, k, v, o
    int dims[1] = {qProjSize * num_q_heads +
                   (kProjSize + vProjSize) * num_kv_heads + oProjSize};
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
  li->add_int_property("bias", bias);
  li->add_int_property("add_bias_kv", add_bias_kv);
  li->add_int_property("add_zero_attn", add_zero_attn);
  li->add_float_property("dropout", dropout);
  li->add_int_property("apply_rotary_embedding", apply_rotary_embedding);
  li->add_int_property("scaling_query", scaling_query);
  li->add_float_property("scaling_factor", scaling_factor);
  li->add_int_property("qk_prod_scaling", qk_prod_scaling);
  li->add_int_property("quantization_type", quantization_type);
  li->add_int_property("offload", offload);
  li->add_int_property("tensor_parallelism_degree",
                       config.tensor_parallelism_degree);
  layers.push_back(li);
  return li->outputs[0];
}

Op *TreeIncMultiHeadSelfAttention::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
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
  layer->get_int_property("bias", value);
  bool bias = (bool)value;
  layer->get_int_property("add_bias_kv", value);
  bool add_bias_kv = (bool)value;
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
  layer->get_int_property("quantization_type", value);
  DataType quantization_type = (DataType)value;
  layer->get_int_property("offload", value);
  bool offload = (bool)value;
  layer->get_int_property("tensor_parallelism_degree", value);
  int tensor_parallelism_degree = (int)value;
  return new TreeIncMultiHeadSelfAttention(model,
                                           layer->layer_guid,
                                           inputs[0],
                                           embed_dim,
                                           num_q_heads,
                                           num_kv_heads,
                                           kdim,
                                           vdim,
                                           dropout,
                                           bias,
                                           add_bias_kv,
                                           add_zero_attn,
                                           apply_rotary_embedding,
                                           scaling_query,
                                           scaling_factor,
                                           qk_prod_scaling,
                                           false /*allocate_weights*/,
                                           quantization_type,
                                           offload,
                                           tensor_parallelism_degree,
                                           layer->name);
}

TreeIncMultiHeadSelfAttention::TreeIncMultiHeadSelfAttention(
    FFModel &model,
    LayerID const &_layer_guid,
    const ParallelTensor _input,
    int _embed_dim,
    int _num_q_heads,
    int _num_kv_heads,
    int _kdim,
    int _vdim,
    float _dropout,
    bool _bias,
    bool _add_bias_kv,
    bool _add_zero_attn,
    bool _apply_rotary_embedding,
    bool _scaling_query,
    float _scaling_factor,
    bool _qk_prod_scaling,
    bool allocate_weights,
    DataType _quantization_type,
    bool _offload,
    int _tensor_parallelism_degree,
    char const *name)
    // Initializer* _bias_initializer)
    : Op(model,
         OP_TREE_INC_MULTIHEAD_SELF_ATTENTION,
         _input->data_type,
         name,
         1 /*inputs*/,
         (_bias ? 2 : 1) /*weights*/,
         1 /*outputs*/,
         _input),
      num_q_heads(_num_q_heads), num_kv_heads(_num_kv_heads), dropout(_dropout),
      bias(_bias), add_bias_kv(_add_bias_kv), add_zero_attn(_add_zero_attn),
      apply_rotary_embedding(_apply_rotary_embedding),
      qSize(_input->dims[0].size), kSize(_input->dims[0].size),
      vSize(_input->dims[0].size), qProjSize(_kdim), kProjSize(_kdim),
      vProjSize(_vdim), oProjSize(_embed_dim),
      qoSeqLength(_input->dims[1].size), kvSeqLength(_input->dims[1].size),
      scaling_query(_scaling_query), scaling_factor(_scaling_factor),
      qk_prod_scaling(_qk_prod_scaling), quantization_type(_quantization_type),
      offload(_offload), tensor_parallelism_degree(_tensor_parallelism_degree) {
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
                   this->num_kv_heads * (kParas + vParas);
    dims[1].is_replica_dim = false;
    // dims[2].size = qParas + kParas + vParas + oParas;
    if (quantization_type != DT_NONE) {
      dims[1].size = get_quantization_to_byte_size(
          data_type, quantization_type, dims[1].size);
    }
    // dims[2].degree = 1;
    // dims[2].parallel_idx = -1;
    int seed = std::rand();
    Initializer *initializer = new GlorotUniform(seed);
    weights[0] = model.create_parallel_weight<2>(
        dims,
        quantization_type == DT_NONE ? this->data_type : quantization_type,
        NULL /*owner_op*/,
        true /*create_grad*/,
        initializer,
        CHOSEN_SYNC_TYPE);
    if (bias) {
      ParallelTensorShape bias_shape = _input->get_shape();
      bias_shape.dims[0].size = qProjSize * num_q_heads +
                                (kProjSize + vProjSize) * num_kv_heads +
                                oProjSize;
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

TreeIncMultiHeadSelfAttention::TreeIncMultiHeadSelfAttention(
    FFModel &model,
    const ParallelTensor _input,
    const ParallelTensor _weight,
    int _embed_dim,
    int _num_q_heads,
    int _num_kv_heads,
    int _kdim,
    int _vdim,
    float _dropout,
    bool _bias,
    bool _add_bias_kv,
    bool _add_zero_attn,
    bool _apply_rotary_embedding,
    bool _scaling_query,
    float _scaling_factor,
    bool _qk_prod_scaling,
    bool allocate_weights,
    DataType _quantization_type,
    bool _offload,
    int _tensor_parallelism_degree,
    char const *name)
    // Initializer* _bias_initializer)
    : Op(model,
         OP_TREE_INC_MULTIHEAD_SELF_ATTENTION,
         _input->data_type,
         name,
         1 /*inputs*/,
         (_bias ? 2 : 1) /*weights*/,
         1 /*outputs*/,
         _input,
         _weight),
      num_q_heads(_num_q_heads), num_kv_heads(_num_kv_heads), dropout(_dropout),
      bias(_bias), add_bias_kv(_add_bias_kv), add_zero_attn(_add_zero_attn),
      apply_rotary_embedding(_apply_rotary_embedding),
      qSize(_input->dims[0].size), kSize(_input->dims[0].size),
      vSize(_input->dims[0].size), qProjSize(_kdim), kProjSize(_kdim),
      vProjSize(_vdim), oProjSize(_embed_dim),
      qoSeqLength(_input->dims[1].size), kvSeqLength(_input->dims[1].size),
      scaling_query(_scaling_query), scaling_factor(_scaling_factor),
      qk_prod_scaling(_qk_prod_scaling), quantization_type(_quantization_type),
      offload(_offload), tensor_parallelism_degree(_tensor_parallelism_degree)
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
                   this->num_kv_heads * (kParas + vParas);
    dims[1].is_replica_dim = false;
    // dims[2].size = qParas + kParas + vParas + oParas;
    if (quantization_type != DT_NONE) {
      dims[1].size = get_quantization_to_byte_size(
          data_type, quantization_type, dims[1].size);
    }
    int seed = std::rand();
    Initializer *initializer = new GlorotUniform(seed);
    weights[0] = model.create_parallel_weight<2>(
        dims,
        quantization_type == DT_NONE ? this->data_type : quantization_type,
        NULL /*owner_op*/,
        true /*create_grad*/,
        initializer,
        CHOSEN_SYNC_TYPE);
    if (bias) {
      ParallelTensorShape bias_shape = _input->get_shape();
      bias_shape.dims[0].size = qProjSize * num_q_heads +
                                (kProjSize + vProjSize) * num_kv_heads +
                                oProjSize;
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

TreeIncMultiHeadSelfAttention::TreeIncMultiHeadSelfAttention(
    FFModel &model,
    TreeIncMultiHeadSelfAttention const &other,
    const ParallelTensor input,
    bool allocate_weights)
    : TreeIncMultiHeadSelfAttention(model,
                                    other.layer_guid,
                                    input,
                                    other.oProjSize,
                                    other.num_q_heads,
                                    other.num_kv_heads,
                                    other.qProjSize,
                                    other.vProjSize,
                                    other.dropout,
                                    other.bias,
                                    other.add_bias_kv,
                                    other.add_zero_attn,
                                    other.apply_rotary_embedding,
                                    other.scaling_query,
                                    other.scaling_factor,
                                    other.qk_prod_scaling,
                                    allocate_weights,
                                    other.quantization_type,
                                    other.offload,
                                    other.tensor_parallelism_degree,
                                    other.name) {}

TreeIncMultiHeadSelfAttention::TreeIncMultiHeadSelfAttention(
    FFModel &model,
    TreeIncMultiHeadSelfAttentionParams const &params,
    ParallelTensor const &input,
    bool allocate_weights,
    char const *name)
    : TreeIncMultiHeadSelfAttention(model,
                                    params.layer_guid,
                                    input,
                                    params.embed_dim,
                                    params.num_q_heads,
                                    params.num_kv_heads,
                                    params.kdim,
                                    params.vdim,
                                    params.dropout,
                                    params.bias,
                                    params.add_bias_kv,
                                    params.add_zero_attn,
                                    params.apply_rotary_embedding,
                                    params.scaling_query,
                                    params.scaling_factor,
                                    params.qk_prod_scaling,
                                    allocate_weights,
                                    params.quantization_type,
                                    params.offload,
                                    params.tensor_parallelism_degree,
                                    name) {}

void TreeIncMultiHeadSelfAttention::init_inference(
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
      TREE_INC_MULTIHEAD_SELF_ATTENTION_INIT_TASK_ID,
      parallel_is,
      TaskArgument(this, sizeof(TreeIncMultiHeadSelfAttention)),
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
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part,
                        0 /*projection id*/,
                        READ_ONLY,
                        EXCLUSIVE,
                        weights[0]->region,
                        ff.cpu_offload ? MAP_TO_ZC_MEMORY : 0));
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

void TreeIncMultiHeadSelfAttention::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(
      TREE_INC_MULTIHEAD_SELF_ATTENTION_INIT_TASK_ID,
      parallel_is,
      TaskArgument(this, sizeof(TreeIncMultiHeadSelfAttention)),
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
OpMeta *TreeIncMultiHeadSelfAttention::init_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  TreeIncMultiHeadSelfAttention const *attn =
      (TreeIncMultiHeadSelfAttention *)task->args;
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
  // int num_q_heads = weight.domain.hi()[1] - weight.domain.lo()[1] + 1;
  int num_q_heads = attn->num_q_heads / attn->tensor_parallelism_degree;
  int num_kv_heads =
      attn->num_kv_heads / attn->tensor_parallelism_degree +
      (attn->num_kv_heads % attn->tensor_parallelism_degree != 0);

  assert(attn->oProjSize == output.domain.hi()[0] - output.domain.lo()[0] + 1);

  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .only_kind(Memory::GPU_FB_MEM)
                       .best_affinity_to(task->target_proc)
                       .first();
  MemoryAllocator gpu_mem_allocator(gpu_mem);
  if (attn->offload) {
    // cpu-offload enabled
    // use offload_reserved_space
    gpu_mem_allocator.register_reserved_work_space(
        handle.offload_reserve_space, handle.offload_reserve_space_size);
  }
  TreeIncMultiHeadSelfAttentionMeta *m =
      new TreeIncMultiHeadSelfAttentionMeta(handle,
                                            attn,
                                            weight,
                                            gpu_mem_allocator,
                                            num_samples,
                                            num_q_heads,
                                            num_kv_heads);
  if (!attn->offload) {
    // assert that we didn't over allocate memory
    assert(gpu_mem_allocator.reserved_allocated_size ==
           gpu_mem_allocator.reserved_total_size);
  }
  m->profiling = attn->profiling;

  if (attn->quantization_type == DT_NONE) {
    assert(weight.domain.get_volume() * data_type_size(weight.data_type) ==
           m->weightSize);
  }
  return m;
}

void TreeIncMultiHeadSelfAttention::forward(FFModel const &ff) {
  // TreeIncMultiHeadSelfAttention doesn't support forward
  assert(false);
}

FutureMap TreeIncMultiHeadSelfAttention::inference(
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
  IndexLauncher launcher(TREE_INC_MULTIHEAD_SELF_ATTENTION_INF_TASK_ID,
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
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part,
                        0 /*projection id*/,
                        READ_ONLY,
                        EXCLUSIVE,
                        weights[0]->region,
                        ff.cpu_offload ? MAP_TO_ZC_MEMORY : 0));
  launcher.add_field(idx++, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(idx++, FID_DATA);
  if (bias) {
    launcher.add_region_requirement(
        RegionRequirement(weights[1]->part,
                          0 /*projection id*/,
                          READ_ONLY,
                          EXCLUSIVE,
                          weights[1]->region,
                          ff.cpu_offload ? MAP_TO_ZC_MEMORY : 0));
    launcher.add_field(idx++, FID_DATA);
  }
  return runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input
  regions[3](I): weight
  regions[4](O): output
*/
void TreeIncMultiHeadSelfAttention::inference_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(task->regions.size() == regions.size());

  // TreeVerifyBatchConfig const *bc = (TreeVerifyBatchConfig *)task->args;
  TreeVerifyBatchConfig const &bc =
      Future(task->futures[0]).get_result<TreeVerifyBatchConfig>();
  log_tree_verify.debug(
      "TreeVerifyBatchConfig, num_tokens: %d, num_requests: %d",
      bc.num_tokens,
      bc.num_active_requests());
  if (bc.num_tokens == 0) {
    return;
  }

  TreeIncMultiHeadSelfAttentionMeta *m =
      *((TreeIncMultiHeadSelfAttentionMeta **)task->local_args);
  assert((*m->bias ? regions.size() == 4 : regions.size() == 3));

  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR weight = helperGetGenericTensorAccessorRO(
      m->weight_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  GenericTensorAccessorR biases;
  if (*m->bias) {
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

  /* print_tensor<float>(input.get_float_ptr(),
                      input_domain.get_volume(),
                      "[Attention:forward:query]"); */

  assert(task->index_point.get_dim() == 1);

  TreeIncMultiHeadSelfAttention::inference_kernel_wrapper(
      m, &bc, task->index_point.point_data[0], input, weight, output, biases);
#ifdef INFERENCE_TESTS
  printf("Checking TreeIncMultiHeadSelfAttention computations...\n");

  // =============================================================================
  //  Define helper functions to handle row-major arrays
  // =============================================================================

  auto set_value_row_major = [](float *arr,
                                std::vector<int> const &shape,
                                std::vector<int> const &indices,
                                float value) -> void {
    int offset = 0;
    for (int i = 0; i < shape.size(); i++) {
      int index = indices[i];
      int stride = 1;
      for (int j = i + 1; j < shape.size(); j++) {
        stride *= shape[j];
      }
      offset += index * stride;
    }
    *(arr + offset) = value;
  };

  // =============================================================================
  //  Load input/output/weights and parse general configs
  // =============================================================================

  float *input_cpu =
      download_tensor<float>(input.get_float_ptr(), input_domain.get_volume());
  assert(input_cpu != nullptr);
  float *weight_cpu = download_tensor<float>(weight.get_float_ptr(),
                                             weight_domain.get_volume());
  assert(weight_cpu != nullptr);
  float *output_cpu = download_tensor<float>(output.get_float_ptr(),
                                             output_domain.get_volume());
  assert(output_cpu != nullptr);

  // Input tensor dimensions
  coord_t data_dim = input_domain.hi()[0] - input_domain.lo()[0] + 1;
  coord_t max_sequence_length = input_domain.hi()[1] - input_domain.lo()[1] + 1;
  coord_t batch_size = input_domain.hi()[2] - input_domain.lo()[2] + 1;
  coord_t replica_dim = input_domain.hi()[3] - input_domain.lo()[3] + 1;
  assert(replica_dim == 1);

  size_t effective_batch_size = max_sequence_length * batch_size;
  float inputs_arr[data_dim][effective_batch_size] = {0};
  for (size_t i = 0; i < data_dim * bc.num_active_tokens(); i++) {
    size_t data_index = i % data_dim;
    size_t token_index = i / data_dim;
    assert(data_index < data_dim);
    assert(token_index < effective_batch_size);
    inputs_arr[data_index][token_index] = input_cpu[i];
  }
  torch::Tensor torch_input = torch::from_blob(
      inputs_arr, {data_dim, (long int)effective_batch_size}, torch::kFloat32);

  // Weight tensor dimensions
  coord_t all_weight_params = weight_domain.hi()[0] - weight_domain.lo()[0] + 1;
  coord_t num_q_heads = weight_domain.hi()[1] - weight_domain.lo()[1] + 1;
  replica_dim = weight_domain.hi()[2] - weight_domain.lo()[2] + 1;
  size_t qParas = m->qProjSize * m->qSize;
  size_t kParas = m->kProjSize * m->kSize;
  size_t vParas = m->vProjSize * m->vSize;
  size_t oParas = m->oProjSize * (m->vProjSize > 0 ? m->vProjSize : m->vSize);

  assert(all_weight_params == qParas + kParas + vParas + oParas);
  assert(num_q_heads == m->num_q_heads);
  assert(replica_dim == 1);

  assert(m->qSize == m->kSize && m->kSize == m->vSize);
  // printf("m->qSize: %i\n", m->qSize);
  //  keep things simple for now
  assert(m->qProjSize == m->kProjSize && m->kProjSize == m->vProjSize);
  long int proj_sum = m->qProjSize + m->kProjSize + m->vProjSize;
  // load weight manually because Torch can't easily read a tensor serialized in
  // column-major order.

  // printf("m->kProjSize: %i, TreeVerifyBatchConfig::MAX_NUM_TOKENS: %i, "
  //     "bc.num_active_tokens(): %i, num_q_heads: %lli,
  //     TreeVerifyBatchConfig::MAX_NUM_REQUESTS: %i, "
  //     "bc.num_active_requests(): %i\n", m->kProjSize,
  //     TreeVerifyBatchConfig::MAX_NUM_TOKENS, bc.num_active_tokens(),
  //     num_q_heads, TreeVerifyBatchConfig::MAX_NUM_REQUESTS,
  //     bc.num_active_requests());
  // for (int t=0; t < bc.num_active_tokens(); t++) {
  //   printf("token %i has request_index: %li and token_position: %li\n",
  //   t, bc.token2ids.token_indexes[t].request_index,
  //   bc.token2ids.token_indexes[t].token_position);
  // }

  // =============================================================================
  //  Load the output tensor (with CUDA results), and create a Torch tensor
  // =============================================================================

  float output_cuda[m->oProjSize][effective_batch_size] = {0};
  for (int i = 0; i < m->oProjSize * effective_batch_size; i++) {
    int row_idx = i % m->oProjSize;
    int col_idx = i / m->oProjSize;
    assert(row_idx < m->oProjSize && col_idx < effective_batch_size);
    output_cuda[row_idx][col_idx] = output_cpu[i];
  }
  torch::Tensor torch_out_cuda =
      torch::from_blob(output_cuda,
                       {m->oProjSize, (int64_t)effective_batch_size},
                       torch::kFloat32);

  // =============================================================================
  //  Load the Q/K/V projection weights, and create a Torch tensor
  // =============================================================================
  std::vector<int> w_qkv_shape = {m->qSize, m->qProjSize, 3, (int)num_q_heads};
  float *w_qkv =
      (float *)calloc(m->qSize * m->qProjSize * 3 * num_q_heads, sizeof(float));
  assert(w_qkv[0] == 0.0f);

  for (int h = 0; h < num_q_heads; h++) {
    for (size_t i = 0; i < m->qProjSize * m->qSize; i++) {
      int row_index = i % m->qSize;
      int column_index = i / m->qSize;
      // Q
      set_value_row_major(w_qkv,
                          w_qkv_shape,
                          {row_index, column_index, 0, h},
                          weight_cpu[all_weight_params * h +
                                     m->qSize * column_index + row_index]);
      // K
      set_value_row_major(
          w_qkv,
          w_qkv_shape,
          {row_index, column_index, 1, h},
          weight_cpu[all_weight_params * h + m->qProjSize * m->qSize +
                     m->qSize * column_index + row_index]);
      // V
      set_value_row_major(
          w_qkv,
          w_qkv_shape,
          {row_index, column_index, 2, h},
          weight_cpu[all_weight_params * h + 2 * m->qProjSize * m->qSize +
                     m->qSize * column_index + row_index]);
    }
  }
  // convert weights to torch tensor
  torch::Tensor torch_w_qkv = torch::from_blob(
      w_qkv, {m->qSize, m->qProjSize, 3, (int)num_q_heads}, torch::kFloat32);

  /* std::cout << "Torch projection weights size: " << torch_w_qkv.sizes()
            << std::endl;
  std::cout << "Torch input size: " << torch_input.sizes() << std::endl;
  std::cout << "Number of active tokens: " << bc.num_active_tokens()
            << std::endl; */
  // std::cout << "torch_w_qkv:" << std::endl << torch_w_qkv << std::endl;

  // =============================================================================
  //  Compute the Q/K/V projections, and compare the results with CUDA
  // =============================================================================

  //  ----------------------- C++ computations & checks ------------------------
  torch::Tensor qkv_projs = torch::einsum(
      "ijkl,im->jmkl",
      {torch_w_qkv,
       torch_input.index({Slice(), Slice(0, bc.num_active_tokens())})});
  // std::cout << "qkv_projs size: " << qkv_projs.sizes() << std::endl;
  assert(qkv_projs.sizes()[0] == m->qProjSize);
  assert(qkv_projs.sizes()[1] == bc.num_active_tokens() &&
         qkv_projs.sizes()[1] <= effective_batch_size);
  assert(qkv_projs.sizes()[2] == 3);
  assert(qkv_projs.sizes()[3] == num_q_heads);
  free(w_qkv);

  //  ----------------------- Loading CUDA results for this step ---------------
  float *QKVProjArray_cpu = download_tensor<float>(
      m->devQKVProjArray,
      TreeVerifyBatchConfig::MAX_NUM_TOKENS * proj_sum * m->num_q_heads);
  assert(QKVProjArray_cpu != nullptr);

  std::vector<int> QKVProjArray_converted_shape = {
      m->qProjSize, bc.num_active_tokens(), 3, (int)num_q_heads};
  float *QKVProjArray_converted = (float *)calloc(
      m->qProjSize * bc.num_active_tokens() * 3 * num_q_heads, sizeof(float));

  // skip over padding at the end of QKVProjArray_cpu
  // convert from column order to 3D matrix because torch cannot automatically
  // import matrices flattened in column order
  for (size_t i = 0; i < proj_sum * bc.num_active_tokens() * num_q_heads; i++) {
    int proj_size_index = i % m->qProjSize;
    int head_index = i / (proj_sum * bc.num_active_tokens());
    int token_index =
        ((i - head_index * proj_sum * bc.num_active_tokens()) / m->qProjSize) %
        bc.num_active_tokens();
    int qkv_offset = (i - head_index * proj_sum * bc.num_active_tokens()) /
                     (m->qProjSize * bc.num_active_tokens());
    assert(proj_size_index < proj_sum);
    assert(head_index < num_q_heads);
    assert(token_index < bc.num_active_tokens());
    assert(qkv_offset < 3);
    set_value_row_major(QKVProjArray_converted,
                        QKVProjArray_converted_shape,
                        {proj_size_index, token_index, qkv_offset, head_index},
                        QKVProjArray_cpu[i]);
  }
  torch::Tensor QKVProjArray_torch =
      torch::from_blob(QKVProjArray_converted,
                       {m->qProjSize, bc.num_active_tokens(), 3, num_q_heads},
                       torch::kFloat32);

  //  ----------------------- Comparing C++ & CUDA results ---------------------
  // std::cout << "QKVProjArray_torch" << std::endl;
  // for (int i=0; i<num_q_heads; i++) {
  //   for (int j=0; j<3; j++) {
  //     std::cout << QKVProjArray_torch.index({Slice(), Slice(), j, i}) <<
  //     std::endl;
  //   }
  // }
  // std::cout << "qkv_projs" << std::endl;
  // for (int i=0; i<num_q_heads; i++) {
  //   for (int j=0; j<3; j++) {
  //     std::cout << qkv_projs.index({Slice(), Slice(), j, i}) << std::endl;
  //   }
  // }
  assert(torch::allclose(QKVProjArray_torch, qkv_projs, 1e-05, 1e-05));
  free(QKVProjArray_converted);

  // =============================================================================
  //  Store the K/V projections into the cache
  // =============================================================================

  //  ----------------------- C++ operations & checks --------------------------
  // Store projections into k/v cache arrays
  for (size_t h = 0; h < num_q_heads; h++) {
    for (size_t t = 0; t < bc.num_active_tokens(); t++) {
      for (size_t d = 0; d < m->kProjSize; d++) {
        size_t kcache_idx = d * MAX_SEQ_LEN * m->num_q_heads *
                                TreeVerifyBatchConfig::MAX_NUM_REQUESTS +
                            bc.tokensInfo[t].abs_depth_in_request *
                                m->num_q_heads *
                                TreeVerifyBatchConfig::MAX_NUM_REQUESTS +
                            h * TreeVerifyBatchConfig::MAX_NUM_REQUESTS +
                            bc.tokensInfo[t].request_index;
        m->kcache[kcache_idx] =
            qkv_projs.index({(int64_t)d, (int64_t)t, 1, (int64_t)h})
                .item<float>();
      }
      for (size_t d = 0; d < m->vProjSize; d++) {
        size_t vcache_idx = d * MAX_SEQ_LEN * m->num_q_heads *
                                TreeVerifyBatchConfig::MAX_NUM_REQUESTS +
                            bc.tokensInfo[t].abs_depth_in_request *
                                m->num_q_heads *
                                TreeVerifyBatchConfig::MAX_NUM_REQUESTS +
                            h * TreeVerifyBatchConfig::MAX_NUM_REQUESTS +
                            bc.tokensInfo[t].request_index;
        m->vcache[vcache_idx] =
            qkv_projs.index({(int64_t)d, (int64_t)t, 2, (int64_t)h})
                .item<float>();
      }
    }
  }
  // Create torch tensors from the arrays
  torch::Tensor K_t =
      torch::from_blob(m->kcache,
                       {m->kProjSize,
                        MAX_SEQ_LEN,
                        num_q_heads,
                        TreeVerifyBatchConfig::MAX_NUM_REQUESTS},
                       torch::kFloat32);
  torch::Tensor V_t =
      torch::from_blob(m->vcache,
                       {m->vProjSize,
                        MAX_SEQ_LEN,
                        num_q_heads,
                        TreeVerifyBatchConfig::MAX_NUM_REQUESTS},
                       torch::kFloat32);

  // Compute useful indices
  std::vector<size_t> req_idxs;
  std::vector<size_t> r_first_idx;
  std::vector<size_t> r_num_tokens;
  for (size_t t = 0; t < bc.num_active_tokens(); t++) {
    size_t rid = bc.tokensInfo[t].request_index;
    if (req_idxs.size() == 0 || req_idxs[req_idxs.size() - 1] != rid) {
      req_idxs.push_back(rid);
      r_first_idx.push_back(t);
      r_num_tokens.push_back(1);
    } else {
      r_num_tokens[r_num_tokens.size() - 1]++;
    }
    assert(req_idxs.size() == r_first_idx.size() &&
           r_first_idx.size() == r_num_tokens.size());
  }
  assert(req_idxs.size() == bc.num_active_requests());
  assert(std::accumulate(r_num_tokens.begin(),
                         r_num_tokens.end(),
                         decltype(r_num_tokens)::value_type(0)) ==
         bc.num_active_tokens());

  //  ----------------------- Loading CUDA results for this step ---------------
  float *keyCache_cpu = download_tensor<float>(
      m->keyCache,
      m->num_q_heads * m->kProjSize * TreeVerifyBatchConfig::MAX_NUM_REQUESTS *
          MAX_SEQ_LEN);
  float *valueCache_cpu = download_tensor<float>(
      m->valueCache,
      m->num_q_heads * m->vProjSize * TreeVerifyBatchConfig::MAX_NUM_REQUESTS *
          MAX_SEQ_LEN);
  assert(keyCache_cpu != nullptr);
  assert(valueCache_cpu != nullptr);

  float *kcache_cuda =
      (float *)calloc(m->kProjSize * MAX_SEQ_LEN * m->num_q_heads *
                          TreeVerifyBatchConfig::MAX_NUM_REQUESTS,
                      sizeof(float));
  float *vcache_cuda =
      (float *)calloc(m->vProjSize * MAX_SEQ_LEN * m->num_q_heads *
                          TreeVerifyBatchConfig::MAX_NUM_REQUESTS,
                      sizeof(float));
  int index = 0;
  for (int i = 0; i < m->kProjSize; i++) {
    for (int j = 0; j < MAX_SEQ_LEN; j++) {
      for (int k = 0; k < m->num_q_heads; k++) {
        for (int l = 0; l < TreeVerifyBatchConfig::MAX_NUM_REQUESTS; l++) {
          int col_major_index =
              l * m->kProjSize * MAX_SEQ_LEN * m->num_q_heads +
              k * m->kProjSize * MAX_SEQ_LEN + j * m->kProjSize + i;
          kcache_cuda[index++] = keyCache_cpu[col_major_index];
        }
      }
    }
  }
  index = 0;
  for (int i = 0; i < m->vProjSize; i++) {
    for (int j = 0; j < MAX_SEQ_LEN; j++) {
      for (int k = 0; k < m->num_q_heads; k++) {
        for (int l = 0; l < TreeVerifyBatchConfig::MAX_NUM_REQUESTS; l++) {
          int col_major_index =
              l * m->vProjSize * MAX_SEQ_LEN * m->num_q_heads +
              k * m->vProjSize * MAX_SEQ_LEN + j * m->vProjSize + i;
          vcache_cuda[index++] = valueCache_cpu[col_major_index];
        }
      }
    }
  }
  torch::Tensor K_t_cuda =
      torch::from_blob(kcache_cuda,
                       {m->kProjSize,
                        MAX_SEQ_LEN,
                        num_q_heads,
                        TreeVerifyBatchConfig::MAX_NUM_REQUESTS},
                       torch::kFloat32);
  torch::Tensor V_t_cuda =
      torch::from_blob(vcache_cuda,
                       {m->vProjSize,
                        MAX_SEQ_LEN,
                        num_q_heads,
                        TreeVerifyBatchConfig::MAX_NUM_REQUESTS},
                       torch::kFloat32);

  //  ----------------------- Comparing C++ & CUDA results ---------------------

  // std::cout << "kcache differences:" << std::endl;
  // for (int i=0; i < bc.num_active_requests() + 1; i++) {
  //   for (int j=0; j < num_q_heads; j++) {
  //     for (int l=0; l < m->kProjSize; l++) {
  //       for (int k=0; k < MAX_SEQ_LEN; k++) {
  //         size_t kcache_idx =
  //           l * MAX_SEQ_LEN * num_q_heads *
  //           TreeVerifyBatchConfig::MAX_NUM_REQUESTS + k * num_q_heads *
  //           TreeVerifyBatchConfig::MAX_NUM_REQUESTS + j *
  //           TreeVerifyBatchConfig::MAX_NUM_REQUESTS + i; if (
  //           abs(m->kcache[kcache_idx] - keyCache_cpu[
  //               i * m->kProjSize * MAX_SEQ_LEN * num_q_heads +
  //               j * m->kProjSize * MAX_SEQ_LEN +
  //               k * m->kProjSize +
  //               l
  //           ]) > 0.00001) {
  //             printf("req: %i (rid: %i), head: %i, data_dim: %i, token_pos:
  //             %i\n",
  //                   i, req_idxs[i], j, l, k);
  //           }
  //       }
  //     }
  //   }
  // }

  //  std::cout << "keyCache from CUDA:" << std::endl;
  //  for (int i=0; i<bc.num_active_requests()+1; i++) {
  //    for (int j=0; j<num_q_heads; j++) {
  //     for (int l=0; l<m->kProjSize; l++) {
  //       for (int k=0; k< MAX_SEQ_LEN; k++) {
  //         printf("%f ",
  //           keyCache_cpu[i * m->kProjSize * MAX_SEQ_LEN * num_q_heads +
  //               j * m->kProjSize * MAX_SEQ_LEN +
  //               k * m->kProjSize +
  //               l
  //         ]);
  //       }
  //       printf("\n");
  //     }
  //     printf("\n");
  //    }
  //    printf("\n");
  //  }

  //  std::cout << "valueCache from CUDA:" << std::endl;
  //  for (int i=0; i<bc.num_active_requests()+1; i++) {
  //    for (int j=0; j<num_q_heads; j++) {
  //       for (int l=0; l<m->vProjSize; l++) {
  //         for (int k=0; k< MAX_SEQ_LEN; k++) {
  //           printf("%f ",
  //             valueCache_cpu[
  //                 i * m->vProjSize * MAX_SEQ_LEN * num_q_heads +
  //                 j * m->vProjSize * MAX_SEQ_LEN +
  //                 k * m->vProjSize +
  //             l]);
  //         }
  //         printf("\n");
  //       }
  //       printf("\n");
  //    }
  //    printf("\n");
  //  }

  //  printf("\n");

  //  std::cout << "C++ kcache:" << std::endl;
  //  for (int i=0; i<bc.num_active_requests()+1; i++) {
  //    for (int j=0; j < num_q_heads; j++) {
  //       for (int l=0; l < m->kProjSize; l++) {
  //         for (int k=0; k < MAX_SEQ_LEN; k++) {
  //           size_t kcache_idx =
  //             l * MAX_SEQ_LEN * num_q_heads *
  //             TreeVerifyBatchConfig::MAX_NUM_REQUESTS + k * num_q_heads *
  //             TreeVerifyBatchConfig::MAX_NUM_REQUESTS + j *
  //             TreeVerifyBatchConfig::MAX_NUM_REQUESTS + i;
  //           printf("%f ", m->kcache[kcache_idx]);
  //         }
  //         printf("\n");
  //       }
  //       printf("\n");
  //    }
  //    printf("\n");
  //  }

  //  std::cout << "C++ vcache:" << std::endl;
  //  for (int i=0; i<bc.num_active_requests()+1; i++) {
  //    for (int j=0; j<num_q_heads; j++) {
  //       for (int l=0; l<m->vProjSize; l++) {
  //         for (int k=0; k< MAX_SEQ_LEN; k++) {
  //             size_t vcache_idx =
  //               l * MAX_SEQ_LEN * num_q_heads *
  //               TreeVerifyBatchConfig::MAX_NUM_REQUESTS + k * num_q_heads *
  //               TreeVerifyBatchConfig::MAX_NUM_REQUESTS + j *
  //               TreeVerifyBatchConfig::MAX_NUM_REQUESTS + i;
  //             printf("%f ", m->vcache[vcache_idx]);
  //         }
  //         printf("\n");
  //       }
  //       printf("\n");
  //    }
  //    printf("\n");
  //  }

  assert(torch::allclose(K_t_cuda, K_t, 1e-05, 1e-05));
  assert(torch::allclose(V_t_cuda, V_t, 1e-05, 1e-05));
  free(kcache_cuda);
  free(vcache_cuda);

  // =============================================================================
  //  Load the W_out projection weights
  // =============================================================================

  //  ----------------------- C++ operations & checks --------------------------
  float *w_out = (float *)calloc(m->vProjSize * m->num_q_heads * m->oProjSize,
                                 sizeof(float));
  std::vector<int> w_out_shape = {m->vProjSize, m->num_q_heads, m->oProjSize};
  assert(m->qProjSize == m->kProjSize && m->kProjSize == m->vProjSize);
  for (int h = 0; h < num_q_heads; h++) {
    for (int v = 0; v < m->vProjSize; v++) {
      for (int o = 0; o < m->oProjSize; o++) {
        set_value_row_major(
            w_out,
            w_out_shape,
            {v, h, o},
            weight_cpu[all_weight_params * h + 3 * m->qProjSize * m->qSize +
                       m->vProjSize * o + v]);
      }
    }
  }
  // convert weights to torch tensor
  torch::Tensor torch_w_out = torch::from_blob(
      w_out, {m->vProjSize, m->num_q_heads, m->oProjSize}, torch::kFloat32);

  //  ----------------------- Loading CUDA results for this step ---------------
  float *w_out_cuda = download_tensor<float>(
      m->W_out_contiguous, m->vProjSize * m->oProjSize * m->num_q_heads);
  assert(w_out_cuda != nullptr);
  float *converted_wout_tensor = (float *)calloc(
      m->vProjSize * m->num_q_heads * m->oProjSize, sizeof(float));
  std::vector<int> converted_wout_tensor_shape = {
      m->vProjSize, m->num_q_heads, m->oProjSize};

  for (int i = 0; i < m->vProjSize * m->num_q_heads * m->oProjSize; i++) {
    int v_idx = i % m->vProjSize;
    int h_idx = (i / m->vProjSize) % m->num_q_heads;
    int o_idx = i / (m->vProjSize * m->num_q_heads);
    assert(v_idx < m->vProjSize && h_idx < m->num_q_heads &&
           o_idx < m->oProjSize);
    set_value_row_major(converted_wout_tensor,
                        converted_wout_tensor_shape,
                        {v_idx, h_idx, o_idx},
                        w_out_cuda[i]);
  }
  torch::Tensor w_out_cuda_tensor =
      torch::from_blob(converted_wout_tensor,
                       {m->vProjSize, m->num_q_heads, m->oProjSize},
                       torch::kFloat32);

  //  ----------------------- Comparing C++ & CUDA results ---------------------
  assert(torch::allclose(w_out_cuda_tensor, torch_w_out, 1e-05, 1e-05));
  free(converted_wout_tensor);

  // =============================================================================
  //  Compute the softmax(QK^T/sqrt(d_k))V product, request by request
  // =============================================================================

  //  ----------------------- C++ initialization steps -------------------------
  torch::Tensor Q_projs = qkv_projs.index({Slice(), Slice(), 0, Slice()})
                              .reshape({qkv_projs.sizes()[0],
                                        qkv_projs.sizes()[1],
                                        qkv_projs.sizes()[3]});

  torch::Tensor qk_products[bc.num_active_requests()];
  torch::Tensor qk_softmax[bc.num_active_requests()];
  torch::Tensor attn_heads[bc.num_active_requests()];

  torch::Tensor cpp_output =
      torch::zeros({m->oProjSize, bc.num_active_tokens()});

  //  ----------------------- Loading CUDA results for this step ---------------
  float *qk_prods_cpu = download_tensor<float>(
      m->qk_prods,
      TreeVerifyBatchConfig::MAX_NUM_TOKENS *
          TreeVerifyBatchConfig::MAX_NUM_TOKENS * num_q_heads);
  assert(qk_prods_cpu != nullptr);

  float *qk_prods_softmax_cpu = download_tensor<float>(
      m->qk_prods_softmax,
      TreeVerifyBatchConfig::MAX_NUM_TOKENS *
          TreeVerifyBatchConfig::MAX_NUM_TOKENS * num_q_heads);
  assert(qk_prods_softmax_cpu != nullptr);

  float *attn_heads_cpu = download_tensor<float>(
      m->attn_heads,
      TreeVerifyBatchConfig::MAX_NUM_TOKENS * m->num_q_heads * m->vProjSize);
  assert(attn_heads_cpu != nullptr);

  //  ----------------------- Main loop (request by request) -------------------
  size_t qk_prods_cpu_offset = 0;

  for (size_t r = 0; r < bc.num_active_requests(); r++) {
    // Compute pre-request parameters
    size_t num_new_tokens = r_num_tokens[r];
    int64_t rid = (int64_t)(req_idxs[r]);
    int64_t num_tokens_received_so_far =
        (int64_t)(bc.requestsInfo[rid].token_start_offset +
                  bc.requestsInfo[rid].num_tokens_in_batch);
    assert(num_new_tokens == bc.requestsInfo[rid].num_tokens_in_batch);
    assert(num_tokens_received_so_far >= (int64_t)num_new_tokens);

    //  ----------------------- C++ computations -------------------------------
    // Get the slice of the Q projection tensor with the tokens in the current
    // request
    torch::Tensor Q_req =
        Q_projs.index({Slice(),
                       Slice(r_first_idx[r], r_first_idx[r] + num_new_tokens),
                       Slice()});
    // std::cout << "Q_req.sizes(): " << Q_req.sizes() << std::endl;
    assert(Q_req.sizes()[0] == m->qProjSize);
    assert(Q_req.sizes()[1] == num_new_tokens);
    assert(Q_req.sizes()[2] == num_q_heads);

    /*printf("\n------------ QK multiplication (C++) -------------\n");
    printf("Request r=%lu. num_new_tokens: %lu, num_tokens_received_so_far: %li,
    rid: %li, Qproj slice: (%i, %i)\n", r, num_new_tokens,
    num_tokens_received_so_far, rid, r_first_idx[r], r_first_idx[r] +
    num_new_tokens);

    std::cout << "Q_req matrix (idk dims):" << std::endl <<
    Q_req.index({Slice(), Slice(), 0}) << std::endl << std::endl; std::cout <<
    "K_t matrix (ilk dims):" << std::endl << K_t.index({Slice(), Slice(0,
    num_tokens_received_so_far), 0, rid}) << std::endl << std::endl; std::cout
    << "C++ alpha: " << (1.0f / sqrt(m->kProjSize)) << std::endl;*/

    // Compute (Q*K^T)/sqrt(d_k) matmul
    qk_products[r] =
        torch::einsum("ijk,ilk->jlk",
                      {Q_req,
                       K_t.index({Slice(),
                                  Slice(0, num_tokens_received_so_far),
                                  Slice(),
                                  rid})}) *
        (1.0f / sqrt(m->kProjSize));

    // Set entries above diagonal to -inf to make attention causal.
    for (int h = 0; h < num_q_heads; h++) {
      qk_products[r].index(
          {Slice(), Slice(num_tokens_received_so_far - num_new_tokens), h}) =
          qk_products[r]
              .index({Slice(),
                      Slice(num_tokens_received_so_far - num_new_tokens),
                      h})
              .tril() +
          torch::full({(int64_t)num_new_tokens, (int64_t)num_new_tokens},
                      -INFINITY)
              .triu()
              .fill_diagonal_(0);
    }
    // Compute softmax for each request block
    qk_softmax[r] = torch::softmax(qk_products[r], -2);
    assert(qk_softmax[r].sizes()[0] == num_new_tokens);
    assert(qk_softmax[r].sizes()[1] == num_tokens_received_so_far);
    assert(qk_softmax[r].sizes()[2] == m->num_q_heads);

    //  ------------------- Loading CUDA results for this step ---------------
    float *converted_qk_prod = (float *)calloc(
        num_new_tokens * num_tokens_received_so_far * num_q_heads,
        sizeof(float));
    float *converted_qk_prod_softmax = (float *)calloc(
        num_new_tokens * num_tokens_received_so_far * num_q_heads,
        sizeof(float));
    std::vector<int> converted_qk_prod_shape = {
        (int)num_new_tokens, (int)num_tokens_received_so_far, (int)num_q_heads};

    for (size_t i = 0;
         i < num_new_tokens * num_tokens_received_so_far * num_q_heads;
         i++) {
      size_t new_t_idx = i % num_new_tokens;
      size_t all_t_idx = (i / num_new_tokens) % num_tokens_received_so_far;
      size_t head_idx = i / (num_new_tokens * num_tokens_received_so_far);
      assert(new_t_idx < num_new_tokens &&
             all_t_idx < num_tokens_received_so_far && head_idx < num_q_heads);
      set_value_row_major(converted_qk_prod,
                          converted_qk_prod_shape,
                          {(int)new_t_idx, (int)all_t_idx, (int)head_idx},
                          qk_prods_cpu[i + qk_prods_cpu_offset]);
      set_value_row_major(converted_qk_prod_softmax,
                          converted_qk_prod_shape,
                          {(int)new_t_idx, (int)all_t_idx, (int)head_idx},
                          qk_prods_softmax_cpu[i + qk_prods_cpu_offset]);
    }
    torch::Tensor qk_prods_cuda = torch::from_blob(
        converted_qk_prod,
        {(int64_t)num_new_tokens, num_tokens_received_so_far, num_q_heads},
        torch::kFloat32);
    torch::Tensor qk_prods_softmax_cuda = torch::from_blob(
        converted_qk_prod_softmax,
        {(int64_t)num_new_tokens, num_tokens_received_so_far, num_q_heads},
        torch::kFloat32);

    //  ------------------- Comparing C++ & CUDA results ------------------
    /* std::cout << "C++:" <<std::endl;
    for (int h=0; h<num_q_heads; h++) {
      std::cout << qk_products[r].index({Slice(), Slice(), h}) << std::endl;
    }
    std::cout << "CUDA:" <<std::endl;
    for (int h=0; h<num_q_heads; h++) {
      std::cout << qk_prods_cuda.index({Slice(), Slice(), h}) << std::endl;
    } */
    /* //
    std::cout << "C++:" <<std::endl;
    for (int h=0; h<num_q_heads; h++) {
      std::cout << qk_softmax[r].index({Slice(), Slice(), h}) << std::endl;
    }
    std::cout << "CUDA:" <<std::endl;
    for (int h=0; h<num_q_heads; h++) {
      std::cout << qk_prods_softmax_cuda.index({Slice(), Slice(), h}) <<
    std::endl;
    } */
    // std::cout << "C++ tril:" <<std::endl;
    // for (int h=0; h<num_q_heads; h++) {
    //   std::cout << qk_products[r].tril().index({Slice(), Slice(), h}) <<
    //   std::endl;
    // }
    assert(torch::allclose(qk_prods_cuda, qk_products[r], 1e-05, 1e-05));
    assert(torch::allclose(qk_prods_softmax_cuda, qk_softmax[r], 1e-05, 1e-05));
    free(converted_qk_prod);
    free(converted_qk_prod_softmax);

    //  --------------------- C++ computations --------------------------
    // Multiply softmax results by V
    assert(
        V_t.index({Slice(), Slice(0, num_tokens_received_so_far), Slice(), rid})
            .sizes()[0] == m->vProjSize);
    assert(
        V_t.index({Slice(), Slice(0, num_tokens_received_so_far), Slice(), rid})
            .sizes()[1] == num_tokens_received_so_far);
    assert(
        V_t.index({Slice(), Slice(0, num_tokens_received_so_far), Slice(), rid})
            .sizes()[2] == m->num_q_heads);
    attn_heads[r] = torch::einsum(
        "ijk,ljk->ilk",
        {qk_softmax[r],
         V_t.index(
             {Slice(), Slice(0, num_tokens_received_so_far), Slice(), rid})});
    assert(attn_heads[r].sizes()[0] == num_new_tokens);
    assert(attn_heads[r].sizes()[1] == m->vProjSize);
    assert(attn_heads[r].sizes()[2] == m->num_q_heads);

    //  ------------------- Loading CUDA results for this step  ---------------
    float converted_attn_heads_cpu[num_new_tokens][m->vProjSize]
                                  [m->num_q_heads] = {0};
    for (int i = 0; i < num_new_tokens * m->vProjSize * m->num_q_heads; i++) {
      int token_ix = i % num_new_tokens;
      int vproj_idx = (i / num_new_tokens) % m->vProjSize;
      int head_idx = i / (num_new_tokens * m->vProjSize);
      assert(token_ix < num_new_tokens && vproj_idx < m->vProjSize &&
             head_idx < m->num_q_heads);
      converted_attn_heads_cpu[token_ix][vproj_idx][head_idx] =
          attn_heads_cpu[r_first_idx[r] * m->vProjSize * m->num_q_heads + i];
    }
    torch::Tensor converted_attn_heads_cuda = torch::from_blob(
        converted_attn_heads_cpu,
        {(int64_t)num_new_tokens, m->vProjSize, m->num_q_heads},
        torch::kFloat32);

    //  -------------------- Comparing C++ & CUDA results -------------------
    /* std::cout << "CUDA attn head for req " << r << ":" <<std::endl;
    for (int h=0; h<m->num_q_heads; h++) {
      std::cout << converted_attn_heads_cuda.index({Slice(), Slice(), h}) <<
    std::endl;
    }
    std::cout << "C++ attn head for req " << r << ":" <<std::endl;
    for (int h=0; h<m->num_q_heads; h++) {
      std::cout << attn_heads[r].index({Slice(), Slice(), h}) << std::endl;
    } */
    assert(torch::allclose(
        converted_attn_heads_cuda, attn_heads[r], 1e-05, 1e-05));

    //  ----------------------- C++ computations ----------------------------
    // Compute output values by projecting all heads to output space
    cpp_output.index(
        {Slice(),
         Slice(r_first_idx[r], r_first_idx[r] + (int64_t)num_new_tokens)}) =
        torch::einsum("jkl,ijk->li", {torch_w_out, attn_heads[r]});

    // increment main loop's auxiliary index
    qk_prods_cpu_offset +=
        num_new_tokens * num_tokens_received_so_far * num_q_heads;
  }

  //  ----------------------- Comparing C++ & CUDA results ---------------------
  /* std::cout << "C++:" <<std::endl;
  for (int i=0; i<m->oProjSize; i++) {
    std::cout << cpp_output.index({i, Slice()}) << std::endl;
  }
  std::cout << "CUDA:" <<std::endl;
  for (int i=0; i<m->oProjSize; i++) {
    std::cout << torch_out_cuda.index({i, Slice(0,
  (int64_t)bc.num_active_tokens())}) << std::endl;
  } */

  assert(
      torch::allclose(torch_out_cuda.index(
                          {Slice(), Slice(0, (int64_t)bc.num_active_tokens())}),
                      cpp_output,
                      1e-05,
                      1e-05));

  // =============================================================================
  //  Cleanup
  // =============================================================================
  free(w_out);
  checkCUDA(cudaFreeHost(input_cpu));
  checkCUDA(cudaFreeHost(weight_cpu));
  checkCUDA(cudaFreeHost(output_cpu));
  checkCUDA(cudaFreeHost(QKVProjArray_cpu));
  checkCUDA(cudaFreeHost(keyCache_cpu));
  checkCUDA(cudaFreeHost(valueCache_cpu));
  checkCUDA(cudaFreeHost(qk_prods_cpu));
  checkCUDA(cudaFreeHost(qk_prods_softmax_cpu));
  checkCUDA(cudaFreeHost(attn_heads_cpu));
  checkCUDA(cudaFreeHost(w_out_cuda));
  // assert(false && "All good if you see this assert failure! :)");
#endif
  // Done with INFERENCE_TESTS block
}

void TreeIncMultiHeadSelfAttention::backward(FFModel const &ff) {
  // TreeIncMultiHeadSelfAttention does not support backward
  assert(false);
}

bool TreeIncMultiHeadSelfAttention::get_int_parameter(PMParameter para,
                                                      int *value) const {
  switch (para) {
    case PM_NUM_HEADS:
      *value = num_q_heads;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

bool TreeIncMultiHeadSelfAttention::measure_operator_cost(
    Simulator *sim, MachineView const &mv, CostMetrics &cost_metrics) const {
  return false;
}

bool operator==(TreeIncMultiHeadSelfAttentionParams const &lhs,
                TreeIncMultiHeadSelfAttentionParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid && lhs.embed_dim == rhs.embed_dim &&
         lhs.num_q_heads == rhs.num_q_heads && lhs.kdim == rhs.kdim &&
         lhs.vdim == rhs.vdim && lhs.dropout == rhs.dropout &&
         lhs.bias == rhs.bias && lhs.add_bias_kv == rhs.add_bias_kv &&
         lhs.add_zero_attn == rhs.add_zero_attn &&
         lhs.apply_rotary_embedding == rhs.apply_rotary_embedding &&
         lhs.scaling_query == rhs.scaling_query &&
         lhs.scaling_factor == rhs.scaling_factor &&
         lhs.qk_prod_scaling == rhs.qk_prod_scaling;
}

TreeIncMultiHeadSelfAttentionParams
    TreeIncMultiHeadSelfAttention::get_params() const {
  TreeIncMultiHeadSelfAttentionParams params;
  params.layer_guid = this->layer_guid;
  params.embed_dim = this->oProjSize;
  params.num_q_heads = this->num_q_heads;
  params.num_kv_heads = this->num_kv_heads;
  params.kdim = this->kProjSize;
  params.vdim = this->vProjSize;
  params.dropout = this->dropout;
  params.bias = this->bias;
  params.add_bias_kv = this->add_bias_kv;
  params.add_zero_attn = this->add_zero_attn;
  params.apply_rotary_embedding = this->apply_rotary_embedding;
  params.scaling_query = this->scaling_query;
  params.scaling_factor = this->scaling_factor;
  params.qk_prod_scaling = this->qk_prod_scaling;
  params.tensor_parallelism_degree = this->tensor_parallelism_degree;
  return params;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::TreeIncMultiHeadSelfAttentionParams>::operator()(
    FlexFlow::TreeIncMultiHeadSelfAttentionParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.embed_dim);
  hash_combine(key, params.num_q_heads);
  hash_combine(key, params.num_kv_heads);
  hash_combine(key, params.kdim);
  hash_combine(key, params.vdim);
  hash_combine(key, params.dropout);
  hash_combine(key, params.bias);
  hash_combine(key, params.add_bias_kv);
  hash_combine(key, params.add_zero_attn);
  hash_combine(key, params.apply_rotary_embedding);
  hash_combine(key, params.scaling_query);
  hash_combine(key, params.scaling_factor);
  hash_combine(key, params.qk_prod_scaling);
  hash_combine(key, params.quantization_type);
  hash_combine(key, params.offload);
  hash_combine(key, params.tensor_parallelism_degree);
  return key;
}
}; // namespace std

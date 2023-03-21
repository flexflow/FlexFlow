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

#include "flexflow/ops/inc_multihead_self_attention.h"
#include "flexflow/model.h"
#include "flexflow/utils/cuda_helper.h"
#include "flexflow/utils/hash_utils.h"
#include <torch/torch.h>
using namespace at::indexing;

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
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

bool IncMultiHeadSelfAttentionParams::is_valid(
    ParallelTensorShape const &input) const {
  bool is_valid = input.is_valid();
  return is_valid;
}

Tensor FFModel::inc_multihead_self_attention(const Tensor input,
                                             int embed_dim,
                                             int num_heads,
                                             int kdim,
                                             int vdim,
                                             float dropout,
                                             bool bias,
                                             bool add_bias_kv,
                                             bool add_zero_attn,
                                             Initializer *kernel_initializer,
                                             char const *name) {
  // Currently assume that
  Layer *li = new Layer(this,
                        OP_INC_MULTIHEAD_SELF_ATTENTION,
                        DT_FLOAT,
                        name,
                        1 /*inputs*/,
                        1 /*weights*/,
                        1 /*outputs*/,
                        input);
  {
    int numdims = input->num_dims;
    int dims[MAX_TENSOR_DIM];
    for (int i = 0; i < numdims; i++) {
      dims[i] = input->dims[i];
    }
    dims[0] = embed_dim;
    li->outputs[0] = create_tensor_legion_ordering(
        numdims, dims, DT_FLOAT, li, 0, true /*create_grad*/);
  }
  {
    // Compute weight size
    int qProjSize = kdim, kProjSize = kdim, vProjSize = kdim,
        oProjSize = embed_dim;
    int qSize = input->dims[0], kSize = input->dims[0], vSize = input->dims[0];
    int qParas = qProjSize * qSize;
    int kParas = kProjSize * kSize;
    int vParas = vProjSize * vSize;
    int oParas = oProjSize * (vProjSize > 0 ? vProjSize : vSize);
    int dims[2] = {qParas + kParas + vParas + oParas, num_heads};
    li->weights[0] = create_weight_legion_ordering(2,
                                                   dims,
                                                   DT_FLOAT,
                                                   li,
                                                   true /*create_grad*/,
                                                   kernel_initializer,
                                                   CHOSEN_SYNC_TYPE);
  }
  li->data_type = DT_FLOAT;
  li->add_int_property("embed_dim", embed_dim);
  li->add_int_property("num_heads", num_heads);
  li->add_int_property("kdim", kdim);
  li->add_int_property("vdim", vdim);
  li->add_int_property("bias", bias);
  li->add_int_property("add_bias_kv", add_bias_kv);
  li->add_int_property("add_zero_attn", add_zero_attn);
  li->add_float_property("dropout", dropout);
  layers.push_back(li);
  return li->outputs[0];
}

Op *IncMultiHeadSelfAttention::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("embed_dim", value);
  int embed_dim = value;
  layer->get_int_property("num_heads", value);
  int num_heads = value;
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
  return new IncMultiHeadSelfAttention(model,
                                       layer->layer_guid,
                                       inputs[0],
                                       embed_dim,
                                       num_heads,
                                       kdim,
                                       vdim,
                                       dropout,
                                       bias,
                                       add_bias_kv,
                                       add_zero_attn,
                                       false /*allocate_weights*/,
                                       layer->name);
}

IncMultiHeadSelfAttention::IncMultiHeadSelfAttention(
    FFModel &model,
    LayerID const &_layer_guid,
    const ParallelTensor _input,
    int _embed_dim,
    int _num_heads,
    int _kdim,
    int _vdim,
    float _dropout,
    bool _bias,
    bool _add_bias_kv,
    bool _add_zero_attn,
    bool allocate_weights,
    char const *name)
    // Initializer* _bias_initializer)
    : Op(model,
         OP_INC_MULTIHEAD_SELF_ATTENTION,
         DT_FLOAT,
         name,
         1 /*inputs*/,
         1 /*weights*/,
         1 /*outputs*/,
         _input),
      num_heads(_num_heads), dropout(_dropout), bias(_bias),
      add_bias_kv(_add_bias_kv), add_zero_attn(_add_zero_attn),
      qSize(_input->dims[0].size), kSize(_input->dims[0].size),
      vSize(_input->dims[0].size), qProjSize(_kdim), kProjSize(_kdim),
      vProjSize(_vdim), oProjSize(_embed_dim),
      qoSeqLength(_input->dims[1].size), kvSeqLength(_input->dims[1].size) {
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
    ParallelDim dims[3];
    dims[0] = inputs[0]->dims[num_dims - 2];
    dims[0].size = dims[0].degree;
    dims[1] = inputs[0]->dims[num_dims - 1];
    dims[1].size = this->num_heads;
    dims[2].size = qParas + kParas + vParas + oParas;
    dims[2].degree = 1;
    dims[2].parallel_idx = -1;
    int seed = std::rand();
    Initializer *initializer = new GlorotUniform(seed);
#ifdef USE_NCCL
    ParameterSyncType comm_type = ParameterSyncType::NCCL;
#else
    ParameterSyncType comm_type = ParameterSyncType::PS;
#endif
    weights[0] = model.create_parallel_weight<3>(dims,
                                                 DT_FLOAT,
                                                 NULL /*owner_op*/,
                                                 true /*create_grad*/,
                                                 initializer,
                                                 comm_type);
  }

  outputs[0] = model.create_parallel_tensor_legion_ordering(
      _input->num_dims, dims, DT_FLOAT, this);
  /* for (int i = 0; i < numdim; i++) { */
  /*   register_output_input_parallel_dims(outputs[0], i, inputs[0], i); */
  /* } */
  /* // Check correctness */
  /* assert(check_output_input_weight_parallel_dims()); */
}

IncMultiHeadSelfAttention::IncMultiHeadSelfAttention(
    FFModel &model,
    const ParallelTensor _input,
    const ParallelTensor _weight,
    int _embed_dim,
    int _num_heads,
    int _kdim,
    int _vdim,
    float _dropout,
    bool _bias,
    bool _add_bias_kv,
    bool _add_zero_attn,
    bool allocate_weights,
    char const *name)
    // Initializer* _bias_initializer)
    : Op(model,
         OP_INC_MULTIHEAD_SELF_ATTENTION,
         DT_FLOAT,
         name,
         1 /*inputs*/,
         1 /*weights*/,
         1 /*outputs*/,
         _input,
         _weight),
      num_heads(_num_heads), dropout(_dropout), bias(_bias),
      add_bias_kv(_add_bias_kv), add_zero_attn(_add_zero_attn),
      qSize(_input->dims[0].size), kSize(_input->dims[0].size),
      vSize(_input->dims[0].size), qProjSize(_kdim), kProjSize(_kdim),
      vProjSize(_vdim), oProjSize(_embed_dim),
      qoSeqLength(_input->dims[1].size), kvSeqLength(_input->dims[1].size)
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
    ParallelDim dims[3];
    dims[0] = inputs[0]->dims[num_dims - 2];
    dims[0].size = dims[0].degree;
    dims[1] = inputs[0]->dims[num_dims - 1];
    dims[1].size = this->num_heads;
    dims[2].size = qParas + kParas + vParas + oParas;
    int seed = std::rand();
    Initializer *initializer = new GlorotUniform(seed);
#ifdef USE_NCCL
    ParameterSyncType comm_type = ParameterSyncType::NCCL;
#else
    ParameterSyncType comm_type = ParameterSyncType::PS;
#endif
    weights[0] = model.create_parallel_weight<3>(dims,
                                                 DT_FLOAT,
                                                 NULL /*owner_op*/,
                                                 true /*create_grad*/,
                                                 initializer,
                                                 comm_type);
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      _input->num_dims, dims, DT_FLOAT, this);

  /* for (int i = 0; i < numdim; i++) { */
  /*   register_output_input_parallel_dims(outputs[0], i, inputs[0], i); */
  /* } */
  /* register_output_weight_parallel_dims(outputs[0], numdim-1, _weight, 1); */
  /* register_output_weight_parallel_dims(outputs[0], numdim-2, _weight, 2); */
  // Check correctness
  /* assert(check_output_input_weight_parallel_dims()); */
}

IncMultiHeadSelfAttention::IncMultiHeadSelfAttention(
    FFModel &model,
    IncMultiHeadSelfAttention const &other,
    const ParallelTensor input,
    bool allocate_weights)
    : IncMultiHeadSelfAttention(model,
                                other.layer_guid,
                                input,
                                other.oProjSize,
                                other.num_heads,
                                other.qProjSize,
                                other.vProjSize,
                                other.dropout,
                                other.bias,
                                other.add_bias_kv,
                                other.add_zero_attn,
                                allocate_weights,
                                other.name) {}

IncMultiHeadSelfAttention::IncMultiHeadSelfAttention(
    FFModel &model,
    IncMultiHeadSelfAttentionParams const &params,
    ParallelTensor const &input,
    bool allocate_weights,
    char const *name)
    : IncMultiHeadSelfAttention(model,
                                params.layer_guid,
                                input,
                                params.embed_dim,
                                params.num_heads,
                                params.kdim,
                                params.vdim,
                                params.dropout,
                                params.bias,
                                params.add_bias_kv,
                                params.add_zero_attn,
                                allocate_weights,
                                name) {}

void IncMultiHeadSelfAttention::init_inference(
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
  IndexLauncher launcher(INC_MULTIHEAD_SELF_ATTENTION_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(IncMultiHeadSelfAttention)),
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
  set_opmeta_from_futuremap_inference(ff, fm, view);
}

void IncMultiHeadSelfAttention::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(INC_MULTIHEAD_SELF_ATTENTION_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(IncMultiHeadSelfAttention)),
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
OpMeta *IncMultiHeadSelfAttention::init_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  IncMultiHeadSelfAttention const *attn =
      (IncMultiHeadSelfAttention *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);

  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      DT_FLOAT, regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR weight = helperGetGenericTensorAccessorRO(
      DT_FLOAT, regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      DT_FLOAT, regions[2], task->regions[2], FID_DATA, ctx, runtime);

  int num_samples = input.domain.hi()[2] - input.domain.lo()[2] + 1;
  assert(attn->qoSeqLength == input.domain.hi()[1] - input.domain.lo()[1] + 1);
  assert(attn->kvSeqLength == input.domain.hi()[1] - input.domain.lo()[1] + 1);
  int num_heads = weight.domain.hi()[1] - weight.domain.lo()[1] + 1;
  assert(attn->oProjSize == output.domain.hi()[0] - output.domain.lo()[0] + 1);

  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .only_kind(Memory::GPU_FB_MEM)
                       .best_affinity_to(task->target_proc)
                       .first();
  IncMultiHeadSelfAttentionMeta *m = new IncMultiHeadSelfAttentionMeta(
      handle, attn, weight.get_float_ptr(), gpu_mem, num_samples, num_heads);
  m->profiling = attn->profiling;
  assert(weight.domain.get_volume() * sizeof(float) == m->weightSize);
  return m;
}

void IncMultiHeadSelfAttention::forward(FFModel const &ff) {
  // IncMultiHeadSelfAttention doesn't support forward
  assert(false);
}

FutureMap IncMultiHeadSelfAttention::inference(
    FFModel const &ff,
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
  int idx = 0;

  printf("BatchConfig, num_tokens: %d, num_requests: %d\n",
         bc.num_tokens,
         bc.num_requests);
  IndexLauncher launcher(INC_MULTIHEAD_SELF_ATTENTION_INF_TASK_ID,
                         parallel_is,
                         TaskArgument(&bc, sizeof(BatchConfig)),
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
  return runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input
  regions[3](I): weight
  regions[4](O): output
*/
void IncMultiHeadSelfAttention::inference_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == regions.size());

  BatchConfig const *bc = (BatchConfig *)task->args;
  IncMultiHeadSelfAttentionMeta const *m =
      *((IncMultiHeadSelfAttentionMeta **)task->local_args);

  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR weight = helperGetGenericTensorAccessorRO(
      m->weight_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);

  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Domain weight_domain = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Domain output_domain = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());

  assert(input_domain.get_dim() == 4);
  assert(weight_domain.get_dim() == 3);
  assert(output_domain.get_dim() == 4);
  /* print_tensor<float>(input.get_float_ptr(),
                      input_domain.get_volume(),
                      "[Attention:forward:query]"); */

  IncMultiHeadSelfAttention::inference_kernel_wrapper(m,
                                                      bc,
                                                      input.get_float_ptr(),
                                                      weight.get_float_ptr(),
                                                      output.get_float_ptr());

  // Now re-implement manually
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
  for (size_t i = 0; i < data_dim * bc->num_active_tokens(); i++) {
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
  coord_t num_heads = weight_domain.hi()[1] - weight_domain.lo()[1] + 1;
  replica_dim = weight_domain.hi()[2] - weight_domain.lo()[2] + 1;
  size_t qParas = m->qProjSize * m->qSize;
  size_t kParas = m->kProjSize * m->kSize;
  size_t vParas = m->vProjSize * m->vSize;
  size_t oParas = m->oProjSize * (m->vProjSize > 0 ? m->vProjSize : m->vSize);

  assert(all_weight_params == qParas + kParas + vParas + oParas);
  assert(num_heads == m->num_heads);
  assert(replica_dim == 1);

  assert(m->qSize == m->kSize && m->kSize == m->vSize);
  printf("m->qSize: %i\n", m->qSize);
  // keep things simple for now
  assert(m->qProjSize == m->kProjSize && m->kProjSize == m->vProjSize);
  long int proj_sum = m->qProjSize + m->kProjSize + m->vProjSize;
  // load weight manually because Torch can't easily read a tensor serialized in
  // column-major order.
  float w_qkv[m->qSize][m->qProjSize][3][num_heads];
  memset(&w_qkv,
         0,
         m->qSize * m->qProjSize * 3 * num_heads *
             sizeof(float)); // assuming that 0.0f is encoded as all zero bytes
  assert(w_qkv[0][0][0][0] == 0.0f);

  for (int h = 0; h < num_heads; h++) {
    for (size_t i = 0; i < m->qProjSize * m->qSize; i++) {
      size_t row_index = i % m->qSize;
      size_t column_index = i / m->qSize;
      // Q
      w_qkv[row_index][column_index][0][h] =
          weight_cpu[all_weight_params * h + m->qSize * column_index +
                     row_index];
      // K
      w_qkv[row_index][column_index][1][h] =
          weight_cpu[all_weight_params * h + m->qProjSize * m->qSize +
                     m->qSize * column_index + row_index];
      // V
      w_qkv[row_index][column_index][2][h] =
          weight_cpu[all_weight_params * h + 2 * m->qProjSize * m->qSize +
                     m->qSize * column_index + row_index];
    }
  }
  // convert weights to torch tensor
  torch::Tensor torch_w_qkv = torch::from_blob(
      w_qkv, {m->qSize, m->qProjSize, 3, num_heads}, torch::kFloat32);

  std::cout << "Torch projection weights size: " << torch_w_qkv.sizes()
            << std::endl;
  std::cout << "Torch input size: " << torch_input.sizes() << std::endl;
  std::cout << "Number of active tokens: " << bc->num_active_tokens()
            << std::endl;
  // std::cout << "torch_w_qkv:" << std::endl << torch_w_qkv << std::endl;
  torch::Tensor qkv_projs = torch::einsum(
      "ijkl,im->jmkl",
      {torch_w_qkv,
       torch_input.index({Slice(), Slice(0, bc->num_active_tokens())})});
  std::cout << "qkv_projs size: " << qkv_projs.sizes() << std::endl;
  assert(qkv_projs.sizes()[0] == m->qProjSize);
  assert(qkv_projs.sizes()[1] == bc->num_active_tokens() &&
         qkv_projs.sizes()[1] <= effective_batch_size);
  assert(qkv_projs.sizes()[2] == 3);
  assert(qkv_projs.sizes()[3] == num_heads);

  float *QKVProjArray_cpu = download_tensor<float>(m->devQKVProjArray,
                                                   BatchConfig::MAX_NUM_TOKENS *
                                                       proj_sum * m->num_heads);
  assert(QKVProjArray_cpu != nullptr);

  float QKVProjArray_converted[m->qProjSize][bc->num_active_tokens()][3]
                              [num_heads];
  memset(&QKVProjArray_converted,
         0,
         m->qProjSize * bc->num_active_tokens() * 3 * num_heads *
             sizeof(float)); // assuming that 0.0f is encoded as all zero bytes
  assert(QKVProjArray_converted[0][0][0][0] == 0.0f);

  // skip over padding at the end of QKVProjArray_cpu
  // convert from column order to 3D matrix because torch cannot automatically
  // import matrices flattened in column order
  for (size_t i = 0; i < proj_sum * bc->num_active_tokens() * num_heads; i++) {
    size_t proj_size_index = i % m->qProjSize;
    size_t head_index = i / (proj_sum * bc->num_active_tokens());
    size_t token_index =
        ((i - head_index * proj_sum * bc->num_active_tokens()) / m->qProjSize) %
        bc->num_active_tokens();
    size_t qkv_offset = (i - head_index * proj_sum * bc->num_active_tokens()) /
                        (m->qProjSize * bc->num_active_tokens());
    assert(proj_size_index < proj_sum);
    assert(head_index < num_heads);
    assert(token_index < bc->num_active_tokens());
    assert(qkv_offset < 3);
    QKVProjArray_converted[proj_size_index][token_index][qkv_offset]
                          [head_index] = QKVProjArray_cpu[i];
  }
  torch::Tensor QKVProjArray_torch =
      torch::from_blob(QKVProjArray_converted,
                       {m->qProjSize, bc->num_active_tokens(), 3, num_heads},
                       torch::kFloat32);

  // std::cout << "QKVProjArray_torch" << std::endl;
  // for (int i=0; i<num_heads; i++) {
  //   for (int j=0; j<3; j++) {
  //     std::cout << QKVProjArray_torch.index({Slice(), Slice(), j, i}) <<
  //     std::endl;
  //   }
  // }
  // std::cout << "qkv_projs" << std::endl;
  // for (int i=0; i<num_heads; i++) {
  //   for (int j=0; j<3; j++) {
  //     std::cout << qkv_projs.index({Slice(), Slice(), j, i}) << std::endl;
  //   }
  // }

  assert(torch::allclose(QKVProjArray_torch, qkv_projs));

  float kcache[m->kProjSize][BatchConfig::MAX_NUM_TOKENS][num_heads]
              [BatchConfig::MAX_NUM_REQUESTS];
  float vcache[m->vProjSize][BatchConfig::MAX_NUM_TOKENS][num_heads]
              [BatchConfig::MAX_NUM_REQUESTS];
  float *keyCache_cpu =
      download_tensor<float>(m->keyCache,
                             m->num_heads * m->kProjSize *
                                 BatchConfig::MAX_NUM_REQUESTS * MAX_SEQ_LEN);
  float *valueCache_cpu =
      download_tensor<float>(m->valueCache,
                             m->num_heads * m->vProjSize *
                                 BatchConfig::MAX_NUM_REQUESTS * MAX_SEQ_LEN);
  assert(keyCache_cpu != nullptr);
  assert(valueCache_cpu != nullptr);

  /* std::cout << "qkv proj array:" << std::endl;
  for (int i=0; i<num_heads; i++) {
    for (int j=0; j<proj_sum; j++) {
        for (int k=0; k<bc->num_active_tokens(); k++) {
            std::cout << qkv_projs.index({j, k, i}).item<float>() << " ";
        }
        std::cout << std::endl;
    }
    std::cout <<std::endl;
    //std::cout << torch_w_qkv.index({Slice(), Slice(), i}) << std::endl;
  } */

  assert(m->qProjSize == m->vProjSize && m->vProjSize == m->kProjSize);
  // printf("m->kProjSize: %i, BatchConfig::MAX_NUM_TOKENS: %i,
  // bc->num_active_tokens(): %i, num_heads: %lli,
  // BatchConfig::MAX_NUM_REQUESTS: %i, bc->num_active_requests(): %i\n",
  //   m->kProjSize, BatchConfig::MAX_NUM_TOKENS, bc->num_active_tokens(),
  //   num_heads, BatchConfig::MAX_NUM_REQUESTS, bc->num_active_requests());
  // for (int t=0; t<bc->num_active_tokens(); t++) {
  //   printf("token %i has request_index: %li and token_position: %li\n", t,
  //   bc->token2ids.token_indexes[t].request_index,
  //   bc->token2ids.token_indexes[t].token_position);
  // }

  for (size_t h = 0; h < num_heads; h++) {
    for (size_t t = 0; t < bc->num_active_tokens(); t++) {
      for (size_t d = 0; d < m->kProjSize; d++) {
        kcache[d][bc->token2ids.token_indexes[t].token_position][h]
              [bc->token2ids.token_indexes[t].request_index] =
                  qkv_projs.index({(int64_t)d, (int64_t)t, 1, (int64_t)h})
                      .item<float>();
        assert(fabs(kcache[d][bc->token2ids.token_indexes[t].token_position][h]
                          [bc->token2ids.token_indexes[t].request_index] -
                    keyCache_cpu[bc->token2ids.token_indexes[t].request_index *
                                     num_heads * m->kProjSize * MAX_SEQ_LEN +
                                 h * m->kProjSize * MAX_SEQ_LEN +
                                 m->kProjSize * bc->token2ids.token_indexes[t]
                                                    .token_position +
                                 d]) <= FLT_EPSILON);
      }
      for (size_t d = 0; d < m->vProjSize; d++) {
        vcache[d][bc->token2ids.token_indexes[t].token_position][h]
              [bc->token2ids.token_indexes[t].request_index] =
                  qkv_projs.index({(int64_t)d, (int64_t)t, 2, (int64_t)h})
                      .item<float>();
        assert(
            fabs(vcache[d][bc->token2ids.token_indexes[t].token_position][h]
                       [bc->token2ids.token_indexes[t].request_index] -
                 valueCache_cpu[bc->token2ids.token_indexes[t].request_index *
                                    num_heads * m->vProjSize * MAX_SEQ_LEN +
                                h * m->vProjSize * MAX_SEQ_LEN +
                                m->vProjSize * bc->token2ids.token_indexes[t]
                                                   .token_position +
                                d]) <= FLT_EPSILON);
      }
    }
  }
  /* std::cout << "keyCache from CUDA:" << std::endl;
  for (int i=0; i<bc->num_active_requests()+1; i++) {
    for (int j=0; j<num_heads; j++) {
        for (int l=0; l<m->kProjSize; l++) {
            for (int k=0; k< MAX_SEQ_LEN; k++) {
                printf("%f ", keyCache_cpu[i*m->kProjSize*MAX_SEQ_LEN*num_heads
  + j*m->kProjSize*MAX_SEQ_LEN + k*m->kProjSize + l]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
  }
  std::cout << "valueCache from CUDA:" << std::endl;
  for (int i=0; i<bc->num_active_requests()+1; i++) {
    for (int j=0; j<num_heads; j++) {
        for (int l=0; l<m->vProjSize; l++) {
            for (int k=0; k< MAX_SEQ_LEN; k++) {
                printf("%f ",
  valueCache_cpu[i*m->vProjSize*MAX_SEQ_LEN*num_heads +
  j*m->vProjSize*MAX_SEQ_LEN + k*m->vProjSize + l]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
  }

  printf("\n");

  std::cout << "C++ kcache:" << std::endl;
  for (int i=0; i<bc->num_active_requests()+1; i++) {
    for (int j=0; j<num_heads; j++) {
        for (int l=0; l<m->kProjSize; l++) {
            for (int k=0; k< MAX_SEQ_LEN; k++) {
                printf("%f ", kcache[l][k][j][i]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
  }
  std::cout << "C++ vcache:" << std::endl;
  for (int i=0; i<bc->num_active_requests()+1; i++) {
    for (int j=0; j<num_heads; j++) {
        for (int l=0; l<m->vProjSize; l++) {
            for (int k=0; k< MAX_SEQ_LEN; k++) {
                printf("%f ", vcache[l][k][j][i]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
  } */

  checkCUDA(cudaFreeHost(input_cpu));
  checkCUDA(cudaFreeHost(weight_cpu));
  checkCUDA(cudaFreeHost(output_cpu));
  checkCUDA(cudaFreeHost(QKVProjArray_cpu));
  checkCUDA(cudaFreeHost(keyCache_cpu));
  checkCUDA(cudaFreeHost(valueCache_cpu));
  assert(false && "All good if you see this assert failure! :)");
}

void IncMultiHeadSelfAttention::backward(FFModel const &ff) {
  // IncMultiHeadSelfAttention does not support backward
  assert(false);
}

bool IncMultiHeadSelfAttention::get_int_parameter(PMParameter para,
                                                  int *value) const {
  switch (para) {
    case PM_NUM_HEADS:
      *value = num_heads;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

bool IncMultiHeadSelfAttention::measure_operator_cost(
    Simulator *sim, MachineView const &mv, CostMetrics &cost_metrics) const {
  return false;
}

using PCG::Node;

bool operator==(IncMultiHeadSelfAttentionParams const &lhs,
                IncMultiHeadSelfAttentionParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid && lhs.embed_dim == rhs.embed_dim &&
         lhs.num_heads == rhs.num_heads && lhs.kdim == rhs.kdim &&
         lhs.vdim == rhs.vdim && lhs.dropout == rhs.dropout &&
         lhs.bias == rhs.bias && lhs.add_bias_kv == rhs.add_bias_kv &&
         lhs.add_zero_attn == rhs.add_zero_attn;
}

IncMultiHeadSelfAttentionParams IncMultiHeadSelfAttention::get_params() const {
  IncMultiHeadSelfAttentionParams params;
  params.layer_guid = this->layer_guid;
  params.embed_dim = this->oProjSize;
  params.num_heads = this->num_heads;
  params.kdim = this->kProjSize;
  params.vdim = this->vProjSize;
  params.dropout = this->dropout;
  params.bias = this->bias;
  params.add_bias_kv = this->add_bias_kv;
  params.add_zero_attn = this->add_zero_attn;
  return params;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::IncMultiHeadSelfAttentionParams>::operator()(
    FlexFlow::IncMultiHeadSelfAttentionParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.embed_dim);
  hash_combine(key, params.num_heads);
  hash_combine(key, params.kdim);
  hash_combine(key, params.vdim);
  hash_combine(key, params.dropout);
  hash_combine(key, params.bias);
  hash_combine(key, params.add_bias_kv);
  hash_combine(key, params.add_zero_attn);
  return key;
}
}; // namespace std

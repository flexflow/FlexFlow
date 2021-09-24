/* Copyright 2021 Facebook
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

#include "flexflow/ops/attention.h"

namespace FlexFlow {

// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;

Tensor FFModel::multihead_attention(const Tensor query,
                                    const Tensor key,
                                    const Tensor value,
                                    int embed_dim,
                                    int num_heads,
                                    int kdim,
                                    int vdim,
                                    float dropout,
                                    bool bias,
                                    bool add_bias_kv,
                                    bool add_zero_attn,
                                    Initializer* kernel_initializer,
                                    const char* name)
{
#ifdef DEADCODE
  {
    MultiHeadAttention* attn = new MultiHeadAttention(*this, query, key, value,
                                                      embed_dim, num_heads,
                                                      kdim, vdim, dropout, bias,
                                                      add_bias_kv, add_zero_attn,
                                                      name);
    layers.push_back(attn);
    return attn->outputs[0];
  }
#endif
}

MultiHeadAttention::MultiHeadAttention(FFModel& model,
                                       const ParallelTensor _query,
                                       const ParallelTensor _key,
                                       const ParallelTensor _value,
                                       int _embed_dim, int _num_heads,
                                       int _kdim, int _vdim,
                                       float _dropout, bool _bias,
                                       bool _add_bias_kv, bool _add_zero_attn,
                                       const char* name)
                                       //Initializer* _bias_initializer)
: Op(model, OP_MULTIHEAD_ATTENTION, name, 3/*inputs*/, 0/*weights*/, 1/*outputs*/,
     _query, _key, _value),
  num_heads(_num_heads), dropout(_dropout), bias(_bias),
  add_bias_kv(_add_bias_kv), add_zero_attn(_add_zero_attn),
  qSize(_query->dims[0].size), kSize(_key->dims[0].size), vSize(_value->dims[0].size),
  qProjSize(_kdim), kProjSize(_kdim), vProjSize(_vdim), oProjSize(_embed_dim),
  qoSeqLength(_query->dims[1].size), kvSeqLength(_key->dims[1].size)
  //bias_initializer(_bias_initializer)
{
  // assert key and value have the same sequence length
  assert(_key->dims[1] == _value->dims[1]);
  numOutputs = 1;
  int numdim = _query->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++)
    dims[i] = _query->dims[i];
  dims[0].size = _embed_dim;
  // Currently require no parallelism along this dim
  assert(dims[0].degree == 1);

  outputs[0] = model.create_parallel_tensor_legion_ordering(_query->num_dims,
                                                   dims, DT_FLOAT, this);
  /* for (int i = 0; i < numdim; i++) { */
  /*   register_output_input_parallel_dims(outputs[0], i, inputs[0], i); */
  /* } */
  /* // Check correctness */
  /* assert(check_output_input_weight_parallel_dims()); */
}

MultiHeadAttention::MultiHeadAttention(FFModel& model,
                                       const ParallelTensor _query,
                                       const ParallelTensor _key,
                                       const ParallelTensor _value,
                                       const ParallelTensor _weight,
                                       int _embed_dim, int _num_heads,
                                       int _kdim, int _vdim,
                                       float _dropout, bool _bias,
                                       bool _add_bias_kv, bool _add_zero_attn,
                                       const char* name)
                                       //Initializer* _bias_initializer)
: Op(model, OP_MULTIHEAD_ATTENTION, name, 3/*inputs*/, 1/*weights*/, 1/*outputs*/,
     _query, _key, _value, _weight),
  num_heads(_num_heads), dropout(_dropout), bias(_bias),
  add_bias_kv(_add_bias_kv), add_zero_attn(_add_zero_attn),
  qSize(_query->dims[0].size), kSize(_key->dims[0].size), vSize(_value->dims[0].size),
  qProjSize(_kdim), kProjSize(_kdim), vProjSize(_vdim), oProjSize(_embed_dim),
  qoSeqLength(_query->dims[1].size), kvSeqLength(_key->dims[1].size)
  //bias_initializer(_bias_initializer)
{
  // assert key and value have the same sequence length
  assert(_key->dims[1] == _value->dims[1]);
  numOutputs = 1;
  int numdim = _query->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++)
    dims[i] = _query->dims[i];
  // assert key and value have the same sequence length
  assert(_key->dims[1] == _value->dims[1]);
  dims[0].size = _embed_dim;
  // Currently require no parallelism along this dim
  assert(dims[0].degree == 1);

  outputs[0] = model.create_parallel_tensor_legion_ordering(_query->num_dims,
                                                   dims, DT_FLOAT, this);
  /* for (int i = 0; i < numdim; i++) { */
  /*   register_output_input_parallel_dims(outputs[0], i, inputs[0], i); */
  /* } */
  assert(_weight->num_dims == 3);
  /* register_output_weight_parallel_dims(outputs[0], numdim-1, _weight, 1); */
  /* register_output_weight_parallel_dims(outputs[0], numdim-2, _weight, 2); */
  // Check correctness
  /* assert(check_output_input_weight_parallel_dims()); */
}

void MultiHeadAttention::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(ATTENTION_INIT_TASK_ID, parallel_is,
      TaskArgument(this, sizeof(MultiHeadAttention)), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(inputs[1]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(inputs[2]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[2]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(3, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
          WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(4, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void MultiHeadAttention::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(ATTENTION_FWD_TASK_ID, parallel_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(inputs[1]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(inputs[2]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[2]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(3, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
          WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(4, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void MultiHeadAttention::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(ATTENTION_BWD_TASK_ID, parallel_is,
      TaskArgument(NULL, 0), argmap,
      Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
      outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(inputs[1]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(inputs[2]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, inputs[2]->region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(3, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
          READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(4, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part_grad, 0/*projection id*/,
          READ_WRITE, EXCLUSIVE, weights[0]->region_grad));
  launcher.add_field(5, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
          READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(6, FID_DATA);
  int num_regions = 7;
  if (inputs[1]->region != inputs[0]->region) {
    // when key != query
    launcher.add_region_requirement(
        RegionRequirement(inputs[1]->part_grad, 0/*projection id*/,
            READ_WRITE, EXCLUSIVE, inputs[1]->region_grad));
    launcher.add_field(num_regions++, FID_DATA);
  }
  if ((inputs[2]->region != inputs[0]->region)
  && (inputs[2]->region != inputs[1]->region)) {
    // when value != key and value != query
    launcher.add_region_requirement(
        RegionRequirement(inputs[2]->part_grad, 0/*projection id*/,
            READ_WRITE, EXCLUSIVE, inputs[2]->region_grad));
    launcher.add_field(num_regions++, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

bool MultiHeadAttention::get_int_parameter(PMParameter para, int* value) const
{
  switch (para) {
    case PM_NUM_HEADS:
      *value = num_heads;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

using PCG::Node;

Node FFModel::get_or_create_multihead_attn_node(const ParallelTensor query,
                                                const ParallelTensor key,
                                                const ParallelTensor value,
                                                int embed_dim,
                                                int num_heads,
                                                int kdim,
                                                int vdim,
                                                float dropout,
                                                bool bias,
                                                bool add_bias_kv,
                                                bool add_zero_attn)
{
  size_t hash = query->get_owner_independent_hash();
  hash = hash * 31 + key->get_owner_independent_hash();
  hash = hash * 31 + value->get_owner_independent_hash();
  hash = hash * 31 + std::hash<int>()(embed_dim);
  hash = hash * 31 + std::hash<int>()(num_heads);
  hash = hash * 31 + std::hash<int>()(kdim);
  hash = hash * 31 + std::hash<int>()(vdim);
  hash = hash * 31 + std::hash<int>()((int)bias);
  hash = hash * 31 + std::hash<int>()((int)add_bias_kv);
  hash = hash * 31 + std::hash<int>()((int)add_zero_attn);
  const auto& it = cached_multihead_attn_ops.find(hash);
  MultiHeadAttention* attn = NULL;
  if (it != cached_multihead_attn_ops.end()) {
    attn = it->second;
  } else {
    attn = new MultiHeadAttention(*this, query, key, value, embed_dim, num_heads,
                                  kdim, vdim, dropout, bias,
                                  add_bias_kv, add_zero_attn, NULL);
    cached_multihead_attn_ops[hash] = attn;
  }
  Node ret;
  ret.guid = node_global_guid ++;
  ret.ptr = attn;
  return ret;
}

}; // namespace FlexFlow

/* Copyright 2019 Stanford
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

#include "flexflow/ops/embedding.h"

namespace FlexFlow {

// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::Rect;
using Legion::PhysicalRegion;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;
using Legion::coord_t;
using Legion::InlineLauncher;

Tensor FFModel::embedding(const Tensor input,
                          int num_entries,
                          int out_dim,
                          AggrMode aggr,
                          const Layer* shared_op,
                          Initializer* kernel_initializer,
                          const char* name)
{
  assert(false);
#ifdef DEADCODE
  Embedding* embed = new Embedding(*this, input, num_entries, out_dim,
                                   aggr, false/*allocate_weights*/, name);
  layers.push_back(embed);
  return embed->outputs[0];
#endif
}

int Embedding::input_vocab_size_replica_dim() const {
  return this->inputs[0]->num_dims - 1;
}

int Embedding::input_channel_out_replica_dim() const {
  return this->inputs[0]->num_dims - 2;
}

int Embedding::output_vocab_size_replica_dim() const {
  return this->inputs[0]->num_dims - 1;
}

int Embedding::output_size(ParallelDim output_dims[MAX_TENSOR_DIM]) {
  ParallelTensor const &input = this->inputs[0];

  const int REPLICA = this->output_vocab_size_replica_dim();
  const int OUT_CHANNELS = Output::OUT_CHANNELS;

  output_dims[OUT_CHANNELS].size = this->out_channels;
  for (int i = 1; i < input->num_dims; i++) {
    output_dims[i] = input->dims[i]; 
  }
  output_dims[REPLICA].is_replica_dim = true;

  return input->num_dims;
}

int Embedding::weight_size(ParallelDim weight_dims[MAX_TENSOR_DIM]) {
  ParallelTensor const &input = this->inputs[0];

  weight_dims[Weight::OUT_CHANNELS].size = this->out_channels;
  weight_dims[Weight::VOCAB_SIZE].size = this->num_entries;
  for (int i = 2; i < input->num_dims; i++) {
    weight_dims[i].is_replica_dim = true;     
  }

  return input->num_dims;
}

void Embedding::register_output_mappings() {
  this->register_output_parallel_dims({
    { this->input_vocab_size_replica_dim(), this->output_vocab_size_replica_dim() },
    { this->input_channel_out_replica_dim(), Output::OUT_CHANNELS },
  });

  for (int i = 1; i < this->inputs[0]->num_dims - 1; i++) {
    this->register_output_parallel_dims(i - 1, i);
  }
}

void Embedding::register_weight_mappings() {
  this->register_weight_parallel_dims({
    { this->input_vocab_size_replica_dim(), Weight::VOCAB_SIZE },
    { this->input_channel_out_replica_dim(), Weight::OUT_CHANNELS },
  });

  for (int i = 2; i < this->inputs[0]->num_dims; i++) {
    this->register_weight_parallel_dims(i - 2, i);
  }
}

void Embedding::register_mappings() {
  this->register_output_mappings();
  this->register_weight_mappings();
}

Embedding::Embedding(FFModel& model,
                     Embedding const &other,
                     const ParallelTensor input,
                     bool allocate_weights) 
: Embedding(model, input, other.num_entries, other.out_channels, other.aggr, allocate_weights, other.name) 
{ }

Embedding::Embedding(FFModel& model,
                     const ParallelTensor _input,
                     int _num_entries,
                     int _out_channels,
                     AggrMode _aggr,
                     bool allocate_weights,
                     const char* name)
: Op(model, OP_EMBEDDING, name, 1/*inputs*/, 1/*weights*/, allocate_weights, 1/*outputs*/, _input),
  num_entries(_num_entries), out_channels(_out_channels), aggr(_aggr)
{
  this->register_mappings();

  std::vector<ParallelDim *> weight_dim_sets;

  int weight_ndim;
  ParallelDim weight_dims[MAX_TENSOR_DIM];
  if (allocate_weights) {
    weight_ndim = this->weight_size(weight_dims);
    weight_dim_sets.push_back(weight_dims);
  }

  ParallelDim output_dims[MAX_TENSOR_DIM];
  int output_ndim = this->output_size(output_dims);

  this->solve_parallel_dim_mappings(
    { _input->dims },
    weight_dim_sets,
    { output_dims }
  );

  if (allocate_weights) {
    Initializer *weight_initializer = new GlorotUniform(std::rand()/*seed*/);

    weights[0] = model.create_parallel_weight_legion_ordering(
        weight_ndim, weight_dims, DT_FLOAT, nullptr/*owner_op*/, true/*create_grad*/, weight_initializer, CHOSEN_SYNC_TYPE);
  }

  outputs[0] = model.create_parallel_tensor_legion_ordering(output_ndim, output_dims, DT_FLOAT, this);

  assert (check_output_input_weight_parallel_dims(allocate_weights));
}

void Embedding::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(EMBED_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Embedding)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0]: input
  //launcher.add_region_requirement(
  //  RegionRequirement(input_lps[0], 0/*projection*/,
  //    READ_ONLY, EXCLUSIVE, inputs[0]->region));
  //launcher.add_field(0, FID_DATA);
  // regions[1]: output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // regions[2]: weight
  launcher.add_region_requirement(
    RegionRequirement(weights[0]->part, 0/*projection*/,
      READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(1, FID_DATA);
  // regions[3]: input_grad
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part_grad, 0/*projection*/,
      WRITE_ONLY, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void Embedding::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(EMBED_FWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0]: input
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // regions[1]: output
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0]->region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[2]: weight
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Embedding::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(EMBED_BWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0]: input
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // regions[1]: output_grad
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part_grad, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  // regions[2]: weight_grad
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part_grad, 0/*projection*/,
                        READ_WRITE, EXCLUSIVE, weights[0]->region_grad));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

using PCG::Node;
Node FFModel::get_or_create_embedding_node(const ParallelTensor input,
                                           int num_entries,
                                           int out_channels,
                                           AggrMode aggr)
{
  size_t hash = input->get_owner_independent_hash();
  hash = hash * 31 + std::hash<int>()(num_entries);
  hash = hash * 31 + std::hash<int>()(out_channels);
  hash = hash * 31 + std::hash<int>()(aggr);
  const auto& it = cached_embedding_ops.find(hash);
  Embedding* embed = NULL;
  if (it != cached_embedding_ops.end()) {
    embed = it->second;
  } else {
    embed = new Embedding(*this, input, num_entries, out_channels,
                          aggr, false/*allocate_weights*/, NULL);
    cached_embedding_ops[hash] = embed;
  }
  Node ret;
  ret.guid = node_global_guid ++;
  ret.ptr = embed;
  return ret;
}

void EmbeddingLookup_int64_t_float_float__avx2_fma(
    const int block_size,
    const int output_size,
    const int index_size,
    const int data_size,
    const float* input,
    const int64_t* indices,
    const int* lengths,
    const float* weight,
    bool normalize_by_lengths,
    float* out) 
{
#ifdef FF_USE_AVX2
  const int64_t prefdist_T0 = 16;
  if (block_size == 128) {
    // unrolling 16 times
    int64_t dataInd = 0;
    for (int64_t rangeIndex = 0; rangeIndex < output_size; ++rangeIndex) {
      float* op = &out[rangeIndex * block_size];
      __m256 vop0 = _mm256_setzero_ps();
      __m256 vop8 = _mm256_setzero_ps();
      __m256 vop16 = _mm256_setzero_ps();
      __m256 vop24 = _mm256_setzero_ps();
      __m256 vop32 = _mm256_setzero_ps();
      __m256 vop40 = _mm256_setzero_ps();
      __m256 vop48 = _mm256_setzero_ps();
      __m256 vop56 = _mm256_setzero_ps();
      __m256 vop64 = _mm256_setzero_ps();
      __m256 vop72 = _mm256_setzero_ps();
      __m256 vop80 = _mm256_setzero_ps();
      __m256 vop88 = _mm256_setzero_ps();
      __m256 vop96 = _mm256_setzero_ps();
      __m256 vop104 = _mm256_setzero_ps();
      __m256 vop112 = _mm256_setzero_ps();
      __m256 vop120 = _mm256_setzero_ps();
      for (int64_t start = dataInd; dataInd < start + lengths[rangeIndex];
           ++dataInd) {
        const int64_t idx = indices[dataInd];
        float wgt = 1.f;
        if (weight) {
          wgt = weight[dataInd];
        }
        __m256 vwgt = _mm256_set1_ps(wgt);
        const float* ip = &input[idx * block_size];
        const int64_t next_T0 = (dataInd < index_size - prefdist_T0)
            ? (dataInd + prefdist_T0)
            : dataInd;
        const int64_t idx_pref_T0 = indices[next_T0];
        assert(
            idx >= 0 && idx_pref_T0 >= 0 && idx < data_size &&
            idx_pref_T0 < data_size);
        const float* ip_next_T0 = &input[idx_pref_T0 * block_size];
        vop0 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (0)), vop0);
        _mm_prefetch((&ip_next_T0[0]), _MM_HINT_T0);
        vop8 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (8)), vop8);
        _mm_prefetch((&ip_next_T0[8]), _MM_HINT_T0);
        vop16 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (16)), vop16);
        _mm_prefetch((&ip_next_T0[16]), _MM_HINT_T0);
        vop24 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (24)), vop24);
        _mm_prefetch((&ip_next_T0[24]), _MM_HINT_T0);
        vop32 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (32)), vop32);
        _mm_prefetch((&ip_next_T0[32]), _MM_HINT_T0);
        vop40 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (40)), vop40);
        _mm_prefetch((&ip_next_T0[40]), _MM_HINT_T0);
        vop48 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (48)), vop48);
        _mm_prefetch((&ip_next_T0[48]), _MM_HINT_T0);
        vop56 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (56)), vop56);
        _mm_prefetch((&ip_next_T0[56]), _MM_HINT_T0);
        vop64 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (64)), vop64);
        _mm_prefetch((&ip_next_T0[64]), _MM_HINT_T0);
        vop72 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (72)), vop72);
        _mm_prefetch((&ip_next_T0[72]), _MM_HINT_T0);
        vop80 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (80)), vop80);
        _mm_prefetch((&ip_next_T0[80]), _MM_HINT_T0);
        vop88 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (88)), vop88);
        _mm_prefetch((&ip_next_T0[88]), _MM_HINT_T0);
        vop96 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (96)), vop96);
        _mm_prefetch((&ip_next_T0[96]), _MM_HINT_T0);
        vop104 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (104)), vop104);
        _mm_prefetch((&ip_next_T0[104]), _MM_HINT_T0);
        vop112 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (112)), vop112);
        _mm_prefetch((&ip_next_T0[112]), _MM_HINT_T0);
        vop120 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (120)), vop120);
        _mm_prefetch((&ip_next_T0[120]), _MM_HINT_T0);
      }
      if (normalize_by_lengths == false) {
        _mm256_storeu_ps(&op[0], vop0);
        _mm256_storeu_ps(&op[8], vop8);
        _mm256_storeu_ps(&op[16], vop16);
        _mm256_storeu_ps(&op[24], vop24);
        _mm256_storeu_ps(&op[32], vop32);
        _mm256_storeu_ps(&op[40], vop40);
        _mm256_storeu_ps(&op[48], vop48);
        _mm256_storeu_ps(&op[56], vop56);
        _mm256_storeu_ps(&op[64], vop64);
        _mm256_storeu_ps(&op[72], vop72);
        _mm256_storeu_ps(&op[80], vop80);
        _mm256_storeu_ps(&op[88], vop88);
        _mm256_storeu_ps(&op[96], vop96);
        _mm256_storeu_ps(&op[104], vop104);
        _mm256_storeu_ps(&op[112], vop112);
        _mm256_storeu_ps(&op[120], vop120);
      } else if (lengths[rangeIndex]) {
        __m256 vlen_inv = _mm256_set1_ps(1.0f / lengths[rangeIndex]);
        _mm256_storeu_ps(&op[0], _mm256_mul_ps(vop0, vlen_inv));
        _mm256_storeu_ps(&op[8], _mm256_mul_ps(vop8, vlen_inv));
        _mm256_storeu_ps(&op[16], _mm256_mul_ps(vop16, vlen_inv));
        _mm256_storeu_ps(&op[24], _mm256_mul_ps(vop24, vlen_inv));
        _mm256_storeu_ps(&op[32], _mm256_mul_ps(vop32, vlen_inv));
        _mm256_storeu_ps(&op[40], _mm256_mul_ps(vop40, vlen_inv));
        _mm256_storeu_ps(&op[48], _mm256_mul_ps(vop48, vlen_inv));
        _mm256_storeu_ps(&op[56], _mm256_mul_ps(vop56, vlen_inv));
        _mm256_storeu_ps(&op[64], _mm256_mul_ps(vop64, vlen_inv));
        _mm256_storeu_ps(&op[72], _mm256_mul_ps(vop72, vlen_inv));
        _mm256_storeu_ps(&op[80], _mm256_mul_ps(vop80, vlen_inv));
        _mm256_storeu_ps(&op[88], _mm256_mul_ps(vop88, vlen_inv));
        _mm256_storeu_ps(&op[96], _mm256_mul_ps(vop96, vlen_inv));
        _mm256_storeu_ps(&op[104], _mm256_mul_ps(vop104, vlen_inv));
        _mm256_storeu_ps(&op[112], _mm256_mul_ps(vop112, vlen_inv));
        _mm256_storeu_ps(&op[120], _mm256_mul_ps(vop120, vlen_inv));
      }
    }
  } else if (block_size == 64) {
    // unrolling 8 times
	  int64_t dataInd = 0;
    for (int64_t rangeIndex = 0; rangeIndex < output_size; ++rangeIndex) {
      float* op = &out[rangeIndex * block_size];
      __m256 vop0 = _mm256_setzero_ps();
      __m256 vop8 = _mm256_setzero_ps();
      __m256 vop16 = _mm256_setzero_ps();
      __m256 vop24 = _mm256_setzero_ps();
      __m256 vop32 = _mm256_setzero_ps();
      __m256 vop40 = _mm256_setzero_ps();
      __m256 vop48 = _mm256_setzero_ps();
      __m256 vop56 = _mm256_setzero_ps();
      for (int64_t start = dataInd; dataInd < start + lengths[rangeIndex];
           ++dataInd) {
        const int64_t idx = indices[dataInd];
        float wgt = 1.f;
        if (weight) {
          wgt = weight[dataInd];
        }
        __m256 vwgt = _mm256_set1_ps(wgt);
        const float* ip = &input[idx * block_size];
        const int64_t next_T0 = (dataInd < index_size - prefdist_T0)
            ? (dataInd + prefdist_T0)
            : dataInd;
        const int64_t idx_pref_T0 = indices[next_T0];
        assert(
            idx >= 0 && idx_pref_T0 >= 0 && idx < data_size &&
            idx_pref_T0 < data_size);
        const float* ip_next_T0 = &input[idx_pref_T0 * block_size];
        vop0 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (0)), vop0);
        _mm_prefetch((&ip_next_T0[0]), _MM_HINT_T0);
        vop8 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (8)), vop8);
        _mm_prefetch((&ip_next_T0[8]), _MM_HINT_T0);
        vop16 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (16)), vop16);
        _mm_prefetch((&ip_next_T0[16]), _MM_HINT_T0);
        vop24 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (24)), vop24);
        _mm_prefetch((&ip_next_T0[24]), _MM_HINT_T0);
        vop32 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (32)), vop32);
        _mm_prefetch((&ip_next_T0[32]), _MM_HINT_T0);
        vop40 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (40)), vop40);
        _mm_prefetch((&ip_next_T0[40]), _MM_HINT_T0);
        vop48 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (48)), vop48);
        _mm_prefetch((&ip_next_T0[48]), _MM_HINT_T0);
        vop56 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (56)), vop56);
        _mm_prefetch((&ip_next_T0[56]), _MM_HINT_T0);
      }
      if (normalize_by_lengths == false) {
        _mm256_storeu_ps(&op[0], vop0);
        _mm256_storeu_ps(&op[8], vop8);
        _mm256_storeu_ps(&op[16], vop16);
        _mm256_storeu_ps(&op[24], vop24);
        _mm256_storeu_ps(&op[32], vop32);
        _mm256_storeu_ps(&op[40], vop40);
        _mm256_storeu_ps(&op[48], vop48);
        _mm256_storeu_ps(&op[56], vop56);
      } else if (lengths[rangeIndex]) {
        __m256 vlen_inv = _mm256_set1_ps(1.0f / lengths[rangeIndex]);
        _mm256_storeu_ps(&op[0], _mm256_mul_ps(vop0, vlen_inv));
        _mm256_storeu_ps(&op[8], _mm256_mul_ps(vop8, vlen_inv));
        _mm256_storeu_ps(&op[16], _mm256_mul_ps(vop16, vlen_inv));
        _mm256_storeu_ps(&op[24], _mm256_mul_ps(vop24, vlen_inv));
        _mm256_storeu_ps(&op[32], _mm256_mul_ps(vop32, vlen_inv));
        _mm256_storeu_ps(&op[40], _mm256_mul_ps(vop40, vlen_inv));
        _mm256_storeu_ps(&op[48], _mm256_mul_ps(vop48, vlen_inv));
        _mm256_storeu_ps(&op[56], _mm256_mul_ps(vop56, vlen_inv));
      }
    }
  } else if (block_size == 32) {
    // unrolling 4 times
    int64_t dataInd = 0;
    for (int64_t rangeIndex = 0; rangeIndex < output_size; ++rangeIndex) {
      float* op = &out[rangeIndex * block_size];
      __m256 vop0 = _mm256_setzero_ps();
      __m256 vop8 = _mm256_setzero_ps();
      __m256 vop16 = _mm256_setzero_ps();
      __m256 vop24 = _mm256_setzero_ps();
      for (int64_t start = dataInd; dataInd < start + lengths[rangeIndex];
           ++dataInd) {
        const int64_t idx = indices[dataInd];
        float wgt = 1.f;
        if (weight) {
          wgt = weight[dataInd];
        }
        __m256 vwgt = _mm256_set1_ps(wgt);
        const float* ip = &input[idx * block_size];
        const int64_t next_T0 = (dataInd < index_size - prefdist_T0)
            ? (dataInd + prefdist_T0)
            : dataInd;
        const int64_t idx_pref_T0 = indices[next_T0];
        assert(
            idx >= 0 && idx_pref_T0 >= 0 && idx < data_size &&
            idx_pref_T0 < data_size);
        const float* ip_next_T0 = &input[idx_pref_T0 * block_size];
        vop0 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (0)), vop0);
        _mm_prefetch((&ip_next_T0[0]), _MM_HINT_T0);
        vop8 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (8)), vop8);
        _mm_prefetch((&ip_next_T0[8]), _MM_HINT_T0);
        vop16 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (16)), vop16);
        _mm_prefetch((&ip_next_T0[16]), _MM_HINT_T0);
        vop24 = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (24)), vop24);
        _mm_prefetch((&ip_next_T0[24]), _MM_HINT_T0);
      }
      if (normalize_by_lengths == false) {
        _mm256_storeu_ps(&op[0], vop0);
        _mm256_storeu_ps(&op[8], vop8);
        _mm256_storeu_ps(&op[16], vop16);
        _mm256_storeu_ps(&op[24], vop24);
      } else if (lengths[rangeIndex]) {
        __m256 vlen_inv = _mm256_set1_ps(1.0f / lengths[rangeIndex]);
        _mm256_storeu_ps(&op[0], _mm256_mul_ps(vop0, vlen_inv));
        _mm256_storeu_ps(&op[8], _mm256_mul_ps(vop8, vlen_inv));
        _mm256_storeu_ps(&op[16], _mm256_mul_ps(vop16, vlen_inv));
        _mm256_storeu_ps(&op[24], _mm256_mul_ps(vop24, vlen_inv));
      }
    }
  } else {
    // generic code
    int64_t dataInd = 0;
    for (int64_t rangeIndex = 0; rangeIndex < output_size; ++rangeIndex) {
      float* op = &out[rangeIndex * block_size];
      int j = 0;
      for (; j + 8 <= block_size; j += 8) {
        _mm256_storeu_ps(op + j, _mm256_setzero_ps());
      }
      for (; j < block_size; j++) {
        op[j] = 0.0f;
      }
      for (int64_t start = dataInd; dataInd < start + lengths[rangeIndex];
           ++dataInd) {
        const int64_t idx = indices[dataInd];
        float wgt = 1.f;
        if (weight) {
          wgt = weight[dataInd];
        }
        __m256 vwgt = _mm256_set1_ps(wgt);
        const float* ip = &input[idx * block_size];
        const int64_t next_T0 = (dataInd < index_size - prefdist_T0)
            ? (dataInd + prefdist_T0)
            : dataInd;
        const int64_t idx_pref_T0 = indices[next_T0];
        assert(
            idx >= 0 && idx_pref_T0 >= 0 && idx < data_size &&
            idx_pref_T0 < data_size);
        const float* ip_next_T0 = &input[idx_pref_T0 * block_size];
        j = 0;
        for (; j + 8 <= block_size; j += 8) {
          _mm256_storeu_ps(
              &op[j],
              _mm256_fmadd_ps(
                  vwgt, _mm256_loadu_ps(&ip[j]), _mm256_loadu_ps(&op[j])));
          _mm_prefetch((&ip_next_T0[j]), _MM_HINT_T0);
        }
        for (; j < block_size; j++) {
          op[j] += wgt * ip[j];
        }
      }
      if (normalize_by_lengths && lengths[rangeIndex]) {
        float len_inv = 1.0f / lengths[rangeIndex];
        __m256 vlen_inv = _mm256_set1_ps(len_inv);
        j = 0;
        for (; j + 8 <= block_size; j += 8) {
          _mm256_storeu_ps(
              &op[j], _mm256_mul_ps(_mm256_loadu_ps(&op[j]), vlen_inv));
        }
        for (; j < block_size; j++) {
          op[j] = len_inv * op[j];
        }
      }
    }
  }
#else
  assert(0);
#endif
}

void embed_forward(const int64_t* input,
		const int* lengths,
		float* output,
		const float* embed,
		int block_size,
		int output_size,
		int index_size,
		int data_size)
{
  EmbeddingLookup_int64_t_float_float__avx2_fma(
      block_size,
      output_size,
      index_size,
      data_size,
      embed,
      input,
      lengths,
      nullptr,
      false,
      output
      );
}

void embed_backward_generic(const int64_t* input,
                            const int* lengths,
                            const float* output,
                            float* embed,
                            int block_size,
                            int output_size,
                            int index_size,
                            int data_size)
{
  //FIXME: Not functionaly correct.
  for (int i=0; i<output_size * block_size; i++)
  {
    int idx = i / block_size;
    int off = i % block_size;
    int64_t wordIdx = input[idx];
    // FIXME: Need to be atomic depending on the strategy
    embed[wordIdx * block_size + off] +=  output[i];;
  }
}

void embed_backward(const int64_t* input,
		                const int* lengths,
                    const float* output,
                    float* embed,
                    int block_size,
                    int output_size,
                    int index_size,
                    int data_size)
{
  embed_backward_generic(input, lengths, output, embed, block_size, output_size, index_size, data_size);
}


void Embedding::forward_task_cpu(const Task *task,
                                 const std::vector<PhysicalRegion>& regions,
                                 Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  //const Embedding* embed = (Embedding*) task->args;
  const AccessorRO<int64_t, 2> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[1], FID_DATA);
  const AccessorRO<float, 2> acc_weight(regions[2], FID_DATA);
  Rect<2> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_output = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_weight = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  coord_t batch_size = rect_input.hi[1] - rect_input.lo[1] + 1;
  // Input and output have same batch size
  assert(batch_size == rect_output.hi[1] - rect_output.lo[1] + 1);
  coord_t out_dim = rect_output.hi[0] - rect_output.lo[0] + 1;
  // Weight and output have same out dim
  assert(out_dim == rect_weight.hi[1] - rect_weight.lo[1] + 1);
  //const int64_t* input = acc_input.ptr(rect_input);
  //float* output = acc_output.ptr(rect_output);
  //const float* weight = acc_weight.ptr(rect_weight);
  int block_size = out_dim;
  int output_size = batch_size;
  int data_size = 1000000; //FIXME
  // For now we are assuming the length is always 1
  int index_size = rect_input.hi[1] - rect_input.lo[1] + 1;
  coord_t in_dim = rect_input.hi[0] - rect_input.lo[0] + 1;
  assert(in_dim == 1);
  std::vector<int> lengths(output_size, 1);
  embed_forward(
      acc_input.ptr(rect_input), lengths.data(), acc_output.ptr(rect_output),
      acc_weight.ptr(rect_weight),
      block_size, output_size, index_size, data_size
  );
}

void Embedding::backward_task_cpu(const Task *task,
                                  const std::vector<PhysicalRegion>& regions,
                                  Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  //const Embedding* embed = (Embedding*) task->args;
  const AccessorRO<int64_t, 2> acc_input(regions[0], FID_DATA);
  const AccessorRO<float, 2> acc_output(regions[1], FID_DATA);
  const AccessorRW<float, 2> acc_weight(regions[2], FID_DATA);
  Rect<2> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_output = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_weight = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  coord_t batch_size = rect_input.hi[1] - rect_input.lo[1] + 1;
  // Input and output have same batch size
  assert(batch_size == rect_output.hi[1] - rect_output.lo[1] + 1);
  //coord_t in_dim = rect_input.hi[0] - rect_input.lo[0] + 1;
  coord_t out_dim = rect_output.hi[0] - rect_output.lo[0] + 1;
  // Weight and output have same out dim
  assert(out_dim == rect_weight.hi[1] - rect_weight.lo[1] + 1);
  //const int64_t* input = acc_input.ptr(rect_input);
  //const float* output = acc_output.ptr(rect_output);
  //float* weight = acc_weight.ptr(rect_weight);
  int block_size = out_dim;
  int output_size = batch_size;
  int index_size = rect_input.hi[1] - rect_input.lo[0] + 1;
  int data_size = 1000000; //FIXME
  std::vector<int> lengths(output_size, 1);
  embed_backward(
      acc_input.ptr(rect_input), lengths.data(), acc_output.ptr(rect_output),
      acc_weight.ptr(rect_weight),
      block_size, output_size, index_size, data_size
  );
}

}; // namespace FlexFlow

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
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

/*static*/
void IncMultiHeadSelfAttention::inference_kernel1(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    float const *input_ptr,
    float const *weight_ptr,
    float *output_ptr,
    cudaStream_t stream) {

  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  int out_dim = (m->qProjSize + m->kProjSize + m->vProjSize) * m->num_heads;
  int in_dim = m->qSize;
  assert(in_dim == m->vSize && in_dim == m->kSize);
  cudaDataType_t data_type = ff_to_cuda_datatype(DT_FLOAT);
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = CUDA_R_32F;
#endif
  checkCUDA(cublasGemmEx(m->handle.blas,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         out_dim,
                         bc->num_tokens,
                         in_dim,
                         &alpha,
                         weight_ptr,
                         data_type,
                         in_dim,
                         input_ptr,
                         data_type,
                         in_dim,
                         &beta,
                         output_ptr,
                         data_type,
                         out_dim,
                         compute_type,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

__global__ void store_kv_cache(float const *devQKVProjArray,
                               float *cache_ptr,
                               BatchConfig::token_ids const *id_map,
                               int qProjSize,
                               int kProjSize,
                               int vProjSize,
                               int num_tokens,
                               int num_heads,
                               int max_seq_len,
                               bool k_cache) {
  CUDA_KERNEL_LOOP(i, num_tokens * (k_cache ? kProjSize : vProjSize) * num_heads) {
    int proj_size = k_cache ? kProjSize : vProjSize;
    int head_idx = i / (num_tokens * proj_size);
    int token_idx = (i - head_idx * (num_tokens * proj_size)) / proj_size;
    int data_idx = i % proj_size;
    
    int qkv_block_size = (qProjSize + kProjSize + vProjSize) * num_tokens;
    int current_head_block_size = num_tokens * (k_cache ? qProjSize : qProjSize + kProjSize);
    float val = devQKVProjArray[head_idx * qkv_block_size + current_head_block_size + token_idx*proj_size + data_idx];
    
    int const req_id = id_map[token_idx].request_id;
    int const tok_id = id_map[token_idx].token_id;
    
    cache_ptr[req_id * (num_heads * max_seq_len * proj_size) + head_idx * (max_seq_len * proj_size) + tok_id * proj_size + data_idx ] = val;
  }
}


/*static*/
void IncMultiHeadSelfAttention::inference_kernel2(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    cudaStream_t stream) {
  int num_tokens = bc->num_active_tokens();
  if (num_tokens > 0) {
    int parallelism = m->kProjSize * num_tokens * m->num_heads;
    store_kv_cache<<<GET_BLOCKS(parallelism),
                    min(CUDA_NUM_THREADS, parallelism),
                    0,
                    stream>>>(
        m->devQKVProjArray,
        m->keyCache,
        m->dev_token2ids,
        m->qProjSize,
        m->kProjSize,
        m->vProjSize,
        num_tokens,
        m->num_heads,
        bc->MAX_SEQUENCE_LENGTH,
        /* k_cache = */ true);
    parallelism = m->vProjSize * num_tokens * m->num_heads;
    store_kv_cache<<<GET_BLOCKS(parallelism),
                    min(CUDA_NUM_THREADS, parallelism),
                    0,
                    stream>>>(
        m->devQKVProjArray,
        m->valueCache,
        m->dev_token2ids,
        m->qProjSize,
        m->kProjSize,
        m->vProjSize,
        num_tokens,
        m->num_heads,
        bc->MAX_SEQUENCE_LENGTH,
        /* k_cache = */ false);
  }
}

/*static*/
void IncMultiHeadSelfAttention::inference_kernel_wrapper(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    float const *input_ptr,
    float const *weight_ptr,
    float *output_ptr) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  // phase 1: Implement kernel to compute KQV for input tokens
  IncMultiHeadSelfAttention::inference_kernel1(m, bc, input_ptr, weight_ptr, m->devQKVProjArray, stream);

  // phase 2: Update key/val cache
  cudaMemcpyAsync(m->dev_token2ids, bc->token2ids, bc->MAX_NUM_TOKENS * sizeof(BatchConfig::token_ids), cudaMemcpyHostToDevice, stream);
  IncMultiHeadSelfAttention::inference_kernel2(m, bc, stream);

  // phase 3: Compute attention score
  // 3 kernels for pahse 3: matmul1 - softmax - matmal2

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("IncMultiHeadSelfAttention forward time = %.2fms\n", elapsed);
    // print_tensor<3, float>(acc_query.ptr, acc_query.rect,
    // "[Attention:forward:query]"); print_tensor<3, float>(acc_output.ptr,
    // acc_output.rect, "[Attention:forward:output]");
  }
}

IncMultiHeadSelfAttentionMeta::IncMultiHeadSelfAttentionMeta(
    FFHandler handler,
    IncMultiHeadSelfAttention const *attn,
    BatchConfig const *bc,
    Memory gpu_mem,
    int num_samples,
    int _num_heads)
    : OpMeta(handler, attn) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  // checkCUDNN(cudnnSetStream(handler.dnn, stream));

  qSize = attn->qSize;
  kSize = attn->kSize;
  vSize = attn->vSize;
  // assume dimensions match for now
  assert(qSize == kSize);
  assert(kSize == vSize);
  qProjSize = attn->qProjSize;
  kProjSize = attn->kProjSize;
  vProjSize = attn->vProjSize;
  oProjSize = attn->oProjSize;
  num_heads = _num_heads;
  weightSize = (qSize * qProjSize + kSize * kProjSize + vSize * vProjSize +
                oProjSize * (vProjSize > 0 ? vProjSize : vSize)) *
               num_heads * sizeof(float);

  // Currently do not support adding bias to key/value projection
  assert(!attn->add_bias_kv);

  // allocate memory for the seqArray and reserve space
  {
    // size_t totalSize = reserveSpaceSize + sizeof(int) * num_samples * 2 +
    // bc->MAX_NUM_REQUESTS *bc-> MAX_SEQUENCE_LENGTH * sizeof(int); size_t
    // max_num_tokens = bc->MAX_NUM_REQUESTS * bc->MAX_SEQUENCE_LENGTH;
    size_t qkv_proj_dim = qProjSize + kProjSize + vProjSize;
    size_t qkv_max_proj_size = bc->MAX_NUM_TOKENS * qkv_proj_dim * num_heads;
    size_t key_cache_size =
        num_heads * kProjSize * bc->MAX_NUM_REQUESTS * bc->MAX_SEQUENCE_LENGTH;
    size_t value_cache_size =
        num_heads * vProjSize * bc->MAX_NUM_REQUESTS * bc->MAX_SEQUENCE_LENGTH;
    size_t token2ids_size = bc->MAX_NUM_TOKENS * sizeof(BatchConfig::token_ids);
    size_t totalSize =
        (qkv_max_proj_size + key_cache_size + value_cache_size) *
        sizeof(float) +
        token2ids_size; // more components will be added here later

    Realm::Rect<1, coord_t> bounds(Realm::Point<1, coord_t>(0),
                                   Realm::Point<1, coord_t>(totalSize - 1));
    std::vector<size_t> field_sizes;
    field_sizes.push_back(sizeof(char));
    Realm::RegionInstance::create_instance(reserveInst,
                                           gpu_mem,
                                           bounds,
                                           field_sizes,
                                           0,
                                           Realm::ProfilingRequestSet())
        .wait();
    devQKVProjArray = (float *)reserveInst.pointer_untyped(0, sizeof(char));
    keyCache = (float *)devQKVProjArray + qkv_max_proj_size;
    valueCache = (float *)keyCache + key_cache_size;
    dev_token2ids = (BatchConfig::token_ids *) (valueCache + value_cache_size);
    // checkCUDA(cudaMemcpy(devQoSeqArray,
    //                      qoSeqArray,
    //                      sizeof(int) * num_samples,
    //                      cudaMemcpyHostToDevice));
    // devKvSeqArray = (int *)devQoSeqArray + num_samples;
    // checkCUDA(cudaMemcpy(devKvSeqArray,
    //                      kvSeqArray,
    //                      sizeof(int) * num_samples,
    //                      cudaMemcpyHostToDevice));
    // kvCache = (int *)devKvSeqArray + num_samples;
    // reserveSpace = (int *)kvCache + bc->MAX_NUM_REQUESTS * bc->
    // MAX_SEQUENCE_LENGTH;
  }

  //input_token_ids = new request_token_id[bc->MAX_NUM_TOKENS];
}

IncMultiHeadSelfAttentionMeta::~IncMultiHeadSelfAttentionMeta(void) {
  reserveInst.destroy();
}

//__global__ void store_kv_cache(
//    float const *input_ptr, float const *cache_ptr, request_token_id const
//    *id_map, int max_seq_len, int hid_dim) {
//  int const token_idx = blockIdx.x;
//  int const element_idx = threadIdx.x;
//  int const req_id = id_map[token_idx].request_id;
//  int const tok_id = id_map[token_idx].token_id;
//  cache_ptr[(req_id * max_seq_len + tok_id) * hid_dim + element_idx] =
//  input_ptr[token_idx * hid_dim + element_idx];
//}

}; // namespace FlexFlow

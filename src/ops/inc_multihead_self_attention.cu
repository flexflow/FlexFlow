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

  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  checkCUDA(cublasSetStream(m->handle.blas, stream));

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

__global__ void store_kv_cache(float const *input_ptr,
                               float const *cache_ptr,
                               request_token_id const *id_map,
                               int max_seq_len,
                               int hid_dim) {
  int const token_idx = blockIdx.x;
  int const element_idx = threadIdx.x;
  int const req_id = id_map[token_idx].request_id;
  int const tok_id = id_map[token_idx].token_id;
  memcpy((float *)input_ptr + token_idx * hid_dim + element_idx,
         (float *)cache_ptr + (req_id * max_seq_len + tok_id) * hid_dim +
             element_idx,
         sizeof(float));
}

/*static*/
void IncMultiHeadSelfAttention::inference_kernel2(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    float const *input_ptr,
    request_token_id const *id_map,
    cudaStream_t stream) {
  store_kv_cache<<<bc->num_tokens, m->kProjSize>>>(
      (float *)input_ptr + bc->MAX_NUM_TOKENS * m->qProjSize,
      m->keyCache,
      id_map,
      bc->MAX_SEQUENCE_LENGTH,
      m->kProjSize);
  store_kv_cache<<<bc->num_tokens, m->vProjSize>>>(
      (float *)input_ptr + bc->MAX_NUM_TOKENS * (m->qProjSize + m->kProjSize),
      m->valueCache,
      id_map,
      bc->MAX_SEQUENCE_LENGTH,
      m->vProjSize);
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

  // phase 0: convert BatchConfig representation to {rid, tid} struct
  int curr_token_idx = 0;
  int processed_requests = 0;
  printf("Start phase 0: num_tokens: %d, num_requests: %d\n",
         bc->num_tokens,
         bc->num_requests);
  printf("num_active_tokens: %d, num_active_requests: %d\n",
         bc->num_active_tokens(),
         bc->num_active_requests());

  for (uint16_t curr_request_idx = 0; curr_request_idx < bc->MAX_NUM_REQUESTS;
       curr_request_idx++) {
    // printf("request %u: num_tokens: %d, start_idx: %d\n",
    //        curr_request_idx,
    //        bc->num_processing_tokens[curr_request_idx],
    //        bc->token_start_idx[curr_request_idx]);

    if (bc->num_processing_tokens[curr_request_idx] == 0) {
      continue;
    }

    for (int i = 0; i < bc->num_processing_tokens[curr_request_idx]; i++) {
      m->input_token_ids[curr_token_idx].request_id = curr_request_idx;
      m->input_token_ids[curr_token_idx].token_id =
          bc->token_start_idx[curr_request_idx] + i;

      // printf("token %d: request_id: %u, token_id: %u\n",
      //        curr_token_idx,
      //        m->input_token_ids[curr_token_idx].request_id,
      //        m->input_token_ids[curr_token_idx].token_id);
      if (curr_token_idx >= bc->num_active_tokens()) {
        printf("curr_token_idx: %d, curr_request_idx: %d\n",
               curr_token_idx,
               curr_request_idx);
        assert(false); // total number of tokens should matches the batch config
      }
      curr_token_idx += 1;
    }
    processed_requests += 1;
  }
  printf("End of phase 0: curr_token_idx: %d, processed_requests: %d\n",
         curr_token_idx,
         processed_requests);

  assert(processed_requests == bc->num_active_requests());

  // phase 1: Implement kernel to compute KQV for input tokens
  IncMultiHeadSelfAttention::inference_kernel1(
      m, bc, input_ptr, weight_ptr, m->devQKVProjArray, stream);

  // phase 2: Update key/val cache
  IncMultiHeadSelfAttention::inference_kernel2(
      m, bc, m->devQKVProjArray, m->input_token_ids, stream);

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
    size_t qkv_max_proj_size = num_samples * qkv_proj_dim * num_heads;
    size_t key_cache_size =
        kProjSize * bc->MAX_NUM_REQUESTS * bc->MAX_SEQUENCE_LENGTH;
    size_t value_cache_size =
        vProjSize * bc->MAX_NUM_REQUESTS * bc->MAX_SEQUENCE_LENGTH;

    size_t totalSize =
        (qkv_max_proj_size + key_cache_size + value_cache_size) *
        sizeof(float); // more components will be added here later

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

  input_token_ids = new request_token_id[bc->MAX_NUM_TOKENS];
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

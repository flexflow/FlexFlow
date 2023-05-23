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
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include "cuComplex.h"
#endif
#include "flexflow/ops/spec_inc_multihead_self_attention.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

__global__ void spec_build_w_out_tensor(float const *weight_ptr,
                                        float *contiguous_weight_ptr,
                                        int vProjSize,
                                        int oProjSize,
                                        int num_heads,
                                        int qkv_weight_block_size) {
  CUDA_KERNEL_LOOP(i, vProjSize * oProjSize * num_heads) {
    int row_idx = i % vProjSize;
    int col_idx = (i / vProjSize) % oProjSize;
    int head_idx = i / (vProjSize * oProjSize);
    contiguous_weight_ptr[i] =
        weight_ptr[head_idx * (qkv_weight_block_size + vProjSize * oProjSize) +
                   qkv_weight_block_size + col_idx * vProjSize + row_idx];
  }
}

__global__ void spec_apply_proj_bias_w(float *input_ptr,
                                       float const *bias_ptr,
                                       int num_tokens,
                                       int oProjSize) {
  CUDA_KERNEL_LOOP(i, num_tokens * oProjSize) {
    int bias_idx = 3 * oProjSize + i % oProjSize;
    input_ptr[i] += bias_ptr[bias_idx];
  }
}

__global__ void spec_apply_proj_bias_qkv(float *input_ptr,
                                         float const *bias_ptr,
                                         int num_tokens,
                                         int qProjSize,
                                         int kProjSize,
                                         int vProjSize,
                                         int num_heads,
                                         bool scaling_query,
                                         float scaling_factor) {
  CUDA_KERNEL_LOOP(
      i, num_tokens * (qProjSize + kProjSize + vProjSize) * num_heads) {
    // for simplicity, assume q, k, v is in same shape
    // 0->q, 1->k, 2->v
    int qkv_index = i / (num_tokens * qProjSize) % 3;

    int head_idx = i / (num_tokens * (qProjSize + kProjSize + vProjSize));
    int qkv_block_size = (qProjSize + kProjSize + vProjSize) * num_tokens;
    int q_block_size = qProjSize * num_tokens;

    int idx = i % (num_tokens * (qProjSize));

    int real_part_index =
        head_idx * qkv_block_size + qkv_index * q_block_size + idx;
    int bias_idx = qkv_index * qProjSize * num_heads + head_idx * qProjSize +
                   (idx % qProjSize);
    input_ptr[real_part_index] += bias_ptr[bias_idx];

    if (scaling_query && qkv_index == 0) {
      input_ptr[real_part_index] *= scaling_factor;
    }
  }
}

__global__ void
    spec_apply_rotary_embedding(float *input_ptr,
                                cuFloatComplex *complex_input,
                                BatchConfig::PerTokenInfo *tokenInfos,
                                int qProjSize,
                                int kProjSize,
                                int num_heads,
                                int num_tokens,
                                int q_block_size,
                                int k_block_size,
                                int v_block_size,
                                bool q_tensor) {
  int proj_size = q_tensor ? qProjSize : kProjSize;
  CUDA_KERNEL_LOOP(i, num_tokens * proj_size * num_heads / 2) {
    // create complex number
    int head_idx = i / (num_tokens * proj_size / 2);
    int idx = i % (num_tokens * proj_size / 2);
    int token_idx =
        (i - head_idx * (num_tokens * proj_size / 2)) / (proj_size / 2);

    int real_part_index =
        idx + token_idx * (proj_size / 2) +
        head_idx * (q_block_size + k_block_size + v_block_size) +
        (q_tensor ? 0 : q_block_size);
    int complex_part_index = real_part_index + (proj_size / 2);

    complex_input[i] = {input_ptr[real_part_index],
                        input_ptr[complex_part_index]};

    // get the freq_cis: shape 1 * (qProjSize/2) = 1 * 64
    // apply a Cartesian coordinate transformation
    // multiple with input & /copy back to q/k

    // get position of token
    //  int head_idx = i / (num_tokens * proj_size);

    // size_t pos = id_map[token_idx].token_position;
    size_t pos = tokenInfos[token_idx].abs_depth_in_request;

    // float before_real = complex_input[i].x, before_complex =
    // complex_input[i].y;

    int pos_i = i % (proj_size / 2);
    float freq = pos * (1.0 / pow(10000.0, (float)2 * pos_i / proj_size));
    cuFloatComplex complex_pos = {cos(freq), sin(freq)};

    complex_input[i] = cuCmulf(complex_input[i], complex_pos);

    input_ptr[real_part_index] = complex_input[i].x;
    input_ptr[complex_part_index] = complex_input[i].y;
  }
}

void compute_qkv_kernel(SpecIncMultiHeadSelfAttentionMeta const *m,
                        BeamSearchBatchConfig const *bc,
                        float const *input_ptr,
                        float const *weight_ptr,
                        float *output_ptr,
                        float const *bias_ptr,
                        cudaStream_t stream) {

  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  float alpha = 1.0f, beta = 0.0f;
  assert(m->qSize == m->vSize && m->qSize == m->kSize);
  cudaDataType_t data_type = ff_to_cuda_datatype(DT_FLOAT);
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = CUDA_R_32F;
#endif
  // Compute (W^T)x matmul: einsum(ijkl,im->jmkl)
  // Weights: qSize x qProjSize x 3 x num_heads
  // Input: qSize x num_tokens
  // Output >>> qProjSize x num_tokens x 3 x num_heads
  int m_q = m->qProjSize;
  int m_k = m->kProjSize;
  int m_v = m->vProjSize;
  assert(m_q == m_k && m_k == m_v); // keep things simple for now
  int n = bc->num_active_tokens();
  int k = m->qSize;
  int lda = k, ldb = k, ldc_q = m_q, ldc_k = m_k, ldc_v = m_v;
  size_t strideA =
      m->weights_params; // need to also skip over all the parameters for each
                         // head, plus the unused W_o weights
  size_t strideB = 0;    // input stays the same for all heads.
  size_t strideC =
      (m_q + m_k + m_v) * n; // size of the output block for each head.
  // Q
  checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       m_q,
                                       n,
                                       k,
                                       &alpha,
                                       weight_ptr,
                                       data_type,
                                       lda,
                                       strideA,
                                       input_ptr,
                                       data_type,
                                       ldb,
                                       strideB,
                                       &beta,
                                       output_ptr,
                                       data_type,
                                       ldc_q,
                                       strideC,
                                       m->num_heads,
                                       compute_type,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       m_k,
                                       n,
                                       k,
                                       &alpha,
                                       weight_ptr + m_q * k,
                                       data_type,
                                       lda,
                                       strideA,
                                       input_ptr,
                                       data_type,
                                       ldb,
                                       strideB,
                                       &beta,
                                       output_ptr + m_q * n,
                                       data_type,
                                       ldc_k,
                                       strideC,
                                       m->num_heads,
                                       compute_type,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // V
  checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       m_v,
                                       n,
                                       k,
                                       &alpha,
                                       weight_ptr + (m_q + m_k) * k,
                                       data_type,
                                       lda,
                                       strideA,
                                       input_ptr,
                                       data_type,
                                       ldb,
                                       strideB,
                                       &beta,
                                       output_ptr + (m_q + m_k) * n,
                                       data_type,
                                       ldc_v,
                                       strideC,
                                       m->num_heads,
                                       compute_type,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // apply rotary emmmbedding for k and v
  // step1 change the k, v to complex tensor
  int num_tokens = bc->num_active_tokens();

  int parallelism = m->kProjSize * num_tokens * m->num_heads;
  int q_block_size = m->qProjSize * num_tokens;
  int k_block_size = m->kProjSize * num_tokens;
  int v_block_size = m->vProjSize * num_tokens;
  // apply bias for q, k, v
  if (*m->bias) {
    spec_apply_proj_bias_qkv<<<GET_BLOCKS(parallelism),
                               min(CUDA_NUM_THREADS, parallelism),
                               0,
                               stream>>>(output_ptr,
                                         bias_ptr,
                                         num_tokens,
                                         m->qProjSize,
                                         m->kProjSize,
                                         m->vProjSize,
                                         m->num_heads,
                                         *m->scaling_query,
                                         m->scaling_factor);
  }

  if (*m->apply_rotary_embedding) {
    /*q*/
    spec_apply_rotary_embedding<<<GET_BLOCKS(parallelism),
                                  min(CUDA_NUM_THREADS, parallelism),
                                  0,
                                  stream>>>(output_ptr,
                                            m->complex_input,
                                            m->token_infos,
                                            m->qProjSize,
                                            m->kProjSize,
                                            m->num_heads,
                                            num_tokens,
                                            q_block_size,
                                            k_block_size,
                                            v_block_size,
                                            true);
    /*k*/
    spec_apply_rotary_embedding<<<GET_BLOCKS(parallelism),
                                  min(CUDA_NUM_THREADS, parallelism),
                                  0,
                                  stream>>>(output_ptr,
                                            m->complex_input,
                                            m->token_infos,
                                            m->qProjSize,
                                            m->kProjSize,
                                            m->num_heads,
                                            num_tokens,
                                            q_block_size,
                                            k_block_size,
                                            v_block_size,
                                            false);
  }
  checkCUDA(cudaDeviceSynchronize());
}

__global__ void spec_store_kv_cache(
    float const *devQKVProjArray,
    float *cache_ptr,
    BatchConfig::PerTokenInfo *tokenInfos,
    BatchConfig::PerRequestInfo *requestInfo,
    BeamSearchBatchConfig::BeamSearchPerTokenInfo *beamTokenInfos,
    BeamSearchBatchConfig::BeamSearchPerRequestInfo *beamRequestInfos,
    int qProjSize,
    int kProjSize,
    int vProjSize,
    int num_tokens,
    int num_heads,
    int max_seq_len,
    int max_beam_width,
    bool k_cache,
    bool is_root) {
  CUDA_KERNEL_LOOP(i,
                   num_tokens * (k_cache ? kProjSize : vProjSize) * num_heads) {
    int proj_size = k_cache ? kProjSize : vProjSize;
    int head_idx = i / (num_tokens * proj_size);
    int token_idx = (i - head_idx * (num_tokens * proj_size)) / proj_size;
    int data_idx = i % proj_size;

    int qkv_block_size = (qProjSize + kProjSize + vProjSize) * num_tokens;
    int current_head_block_size =
        num_tokens * (k_cache ? qProjSize : qProjSize + kProjSize);
    float val =
        devQKVProjArray[head_idx * qkv_block_size + current_head_block_size +
                        token_idx * proj_size + data_idx];

    // above no need to be changed
    // int const req_id = id_map[token_idx].request_index;
    // int const tok_id = id_map[token_idx].token_position;
    // int const sub_req_id = id_map[token_idx].sub_request_index;
    // int const parent_id = id_map[token_idx].parent_id;
    // int const beam_depth = id_map[token_idx].beam_depth;
    // int const beam_width = id_map[token_idx].beam_width;

    int const req_id = tokenInfos[token_idx].request_index;
    int const tok_id = tokenInfos[token_idx].abs_depth_in_request;
    int const sub_req_id = beamTokenInfos[token_idx].sub_request_index;
    int const parent_id = beamRequestInfos[req_id].parent_id[sub_req_id];
    int const beam_depth = beamRequestInfos[req_id].current_depth;
    int const beam_width = beamRequestInfos[req_id].beam_size;

    // new token
    int new_token_cache_idx = (req_id * max_beam_width + sub_req_id) *
                                  (num_heads * max_seq_len * proj_size) +
                              head_idx * (max_seq_len * proj_size) +
                              tok_id * proj_size + data_idx;
    cache_ptr[new_token_cache_idx] = val;

    // replica in the root iteration
    if (beam_depth == 1) {
      for (int i = 1; i < beam_width; i++) {
        cache_ptr[(req_id * max_beam_width + i) *
                      (num_heads * max_seq_len * proj_size) +
                  head_idx * (max_seq_len * proj_size) + tok_id * proj_size +
                  data_idx] = val;
      }
    }

    // if (head_idx == 0 && beam_depth == 0 && token_idx == 8 && k_cache) {
    //   // printf("token idx %d\n", token_idx);
    //   printf("data idx: %d, tok_id %d, new_token_cache_idx %d, parent_id %d,
    //   "
    //          "sub_req_id %d, num_tokens %d, kProjSize %d, num_heads %d, val "
    //          "%f, beam_width %d\n",
    //          data_idx,
    //          tok_id,
    //          new_token_cache_idx,
    //          parent_id,
    //          sub_req_id,
    //          num_tokens,
    //          kProjSize,
    //          num_heads,
    //          val,
    //          beam_width);
    // }

    // naive cache stealing
    if (sub_req_id != parent_id) {
      if (data_idx == 0 && head_idx == 0 && k_cache) {
        printf("cache stealing!, depth %d req_id %d sub_req_id %d, parentid "
               "%d, tok_id %d\n",
               beam_depth,
               req_id,
               sub_req_id,
               parent_id,
               tok_id);
      }

      for (int depth = 0; depth < beam_depth; depth++) {
        int steal_token_idx = tok_id - beam_depth + depth;
        int steal_from_idx = (req_id * max_beam_width + parent_id) *
                                 (num_heads * max_seq_len * proj_size) +
                             head_idx * (max_seq_len * proj_size) +
                             steal_token_idx * proj_size + data_idx;
        int steal_to_idx = (req_id * max_beam_width + sub_req_id) *
                               (num_heads * max_seq_len * proj_size) +
                           head_idx * (max_seq_len * proj_size) +
                           steal_token_idx * proj_size + data_idx;
        cache_ptr[steal_to_idx] = cache_ptr[steal_from_idx];

        //   if(data_idx == 0 && head_idx == 0 && k_cache && req_id == 1){
        //     printf("cache stealing kernel!, steal_token_idx %d\n",
        //     steal_token_idx);
        // }
      }
    }

    // parallel cache stealing not yet implemented
    // logic shld be
    // launch spec_store_kv_cache with parallelism * current depth
    // from the i here, get depth index
    // if depth index not the current one, check if we need to steal
    // steal if needed

    // cache stealing theory
    // identify which sub request does this token come from
    // for initial token, 0
    // for other, may 0,0,1/ 0,1,2/ 1,1,1 to get which cache to be reuse and
    // which to be delete copy beam_size bunch of blocks when sub_req_id ==
    // parent_id : like 0 -> 0, 1->1, 2->2, do nothing, just append the new k/v
  }
}

void update_kv_cache_kernel(SpecIncMultiHeadSelfAttentionMeta const *m,
                            BeamSearchBatchConfig const *bc,
                            cudaStream_t stream) {
  int num_tokens = bc->num_active_tokens();
  int curr_depth = bc->beamRequestsInfo[0].current_depth;
  // printf("curr depth: %d\n", curr_depth);
  // assert(curr_depth < 3);
  if (num_tokens > 0) {
    int parallelism = m->kProjSize * num_tokens * m->num_heads;
    spec_store_kv_cache<<<GET_BLOCKS(parallelism),
                          min(CUDA_NUM_THREADS, parallelism),
                          0,
                          stream>>>(m->devQKVProjArray,
                                    m->keyCache,
                                    m->token_infos,
                                    m->request_infos,
                                    m->beam_token_infos,
                                    m->beam_request_infos,
                                    m->qProjSize,
                                    m->kProjSize,
                                    m->vProjSize,
                                    num_tokens,
                                    m->num_heads,
                                    BatchConfig::MAX_SEQ_LENGTH,
                                    BeamSearchBatchConfig::MAX_BEAM_WIDTH,
                                    /* k_cache = */ true,
                                    /*root*/ curr_depth == 0);

    parallelism = m->vProjSize * num_tokens * m->num_heads;
    spec_store_kv_cache<<<GET_BLOCKS(parallelism),
                          min(CUDA_NUM_THREADS, parallelism),
                          0,
                          stream>>>(m->devQKVProjArray,
                                    m->valueCache,
                                    m->token_infos,
                                    m->request_infos,
                                    m->beam_token_infos,
                                    m->beam_request_infos,
                                    m->qProjSize,
                                    m->kProjSize,
                                    m->vProjSize,
                                    num_tokens,
                                    m->num_heads,
                                    BatchConfig::MAX_SEQ_LENGTH,
                                    BeamSearchBatchConfig::MAX_BEAM_WIDTH,
                                    /* k_cache = */ false,
                                    /*root*/ curr_depth == 0);
  }
}

__global__ void spec_fill_entries_above_diagonal(float *matrix,
                                                 size_t new_tokens,
                                                 size_t total_tokens_in_request,
                                                 size_t num_heads,
                                                 float value) {
  CUDA_KERNEL_LOOP(i, new_tokens * total_tokens_in_request * num_heads) {
    // size_t head_idx = i / (new_tokens * total_tokens_in_request);
    size_t src_idx = (i / new_tokens) % total_tokens_in_request;
    size_t dst_idx = i % new_tokens + total_tokens_in_request - new_tokens;
    // Casual Mask
    if (src_idx > dst_idx) {
      matrix[i] = value;
    }
  }
}

void compute_attention_kernel(SpecIncMultiHeadSelfAttentionMeta const *m,
                              BeamSearchBatchConfig const *bc,
                              float *output_ptr,
                              float const *bias_ptr,
                              cudaStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(DT_FLOAT);
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = CUDA_R_32F;
#endif
  // int num_requests = bc->num_active_requests();
  int num_tokens = bc->num_active_tokens();
  int tokens_previous_requests = 0;
  int tokens_prev_requests_squares = 0;
  int qkv_block_size =
      (m->qProjSize + m->kProjSize + m->vProjSize) * num_tokens;
  int kt_block_size = m->kProjSize * BatchConfig::MAX_SEQ_LENGTH;
  int kt_req_block_size = kt_block_size * m->num_heads;
  int vt_block_size = m->vProjSize * BatchConfig::MAX_SEQ_LENGTH;
  int vt_req_block_size = vt_block_size * m->num_heads;
  assert(m->qProjSize == m->kProjSize);

  for (int i = 0; i < bc->MAX_NUM_REQUESTS; i++) {
    if (bc->request_completed[i]) {
      continue;
    }
    for (int sub_req_id = 0; sub_req_id < bc->sub_requests[i]; sub_req_id++) {

      // int num_new_tokens = bc->num_processing_tokens[i];
      // int total_tokens = bc->token_last_available_idx[i] + 1;

      int num_new_tokens = bc->requestsInfo[i].num_tokens_in_batch;
      int total_tokens = bc->requestsInfo[i].token_start_offset +
                         bc->requestsInfo[i].num_tokens_in_batch;
      // Compute (QK^T/sqrt(d_k))
      int m_ = num_new_tokens;
      int n = total_tokens;
      int k = m->qProjSize;
      int lda = k, ldb = k, ldc = m_;
      int strideA = qkv_block_size;
      int strideB = kt_block_size;
      int strideC = num_new_tokens * total_tokens;

      // a flag of using this scaling alpha
      float alpha = 1.0f, beta = 0.0f;
      if (*m->qk_prod_scaling) {
        alpha = 1.0f / (float)sqrt(m->kProjSize), beta = 0.0f;
      }
      // To get A, skip over Q entries from previous requests (same head)
      void const *A = (void const *)(m->devQKVProjArray +
                                     tokens_previous_requests * m->qProjSize);
      // To get B, skip over K entries from previous requests (all heads +
      // padding)
      void const *B =
          (void const *)(m->keyCache + (i * bc->MAX_BEAM_WIDTH + sub_req_id) *
                                           kt_req_block_size);

      // if (i == 0 && sub_req_id == 0 &&
      //     bc->beam_slots.at(0).current_depth == 1) {
      //   int offset = (float *)B - m->keyCache;
      //   printf("key cache offset %d\n", kt_req_block_size);
      // }
      // To get C, skip over QK^T products from previous requests
      void *C =
          (void *)(m->qk_prods + m->num_heads * tokens_prev_requests_squares);

      checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_N,
                                           m_,
                                           n,
                                           k,
                                           &alpha,
                                           A,
                                           cublas_data_type,
                                           lda,
                                           strideA,
                                           B,
                                           cublas_data_type,
                                           ldb,
                                           strideB,
                                           &beta,
                                           C,
                                           cublas_data_type,
                                           ldc,
                                           strideC,
                                           m->num_heads,
                                           compute_type,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      // Fill all elements above diagonal in qk prods with -inf to force
      // causal attention.
      assert(num_new_tokens <= total_tokens);
      if (num_new_tokens > 1) {
        size_t parallelism = m->num_heads * num_new_tokens * total_tokens;
        spec_fill_entries_above_diagonal<<<GET_BLOCKS(parallelism),
                                           min((size_t)CUDA_NUM_THREADS,
                                               parallelism),
                                           0,
                                           stream>>>(
            (float *)C, num_new_tokens, total_tokens, m->num_heads, -INFINITY);
      }
      // Compute Softmax(QK^T/sqrt(d_k))
      cudnnTensorDescriptor_t qk_tensor;
      checkCUDNN(cudnnCreateTensorDescriptor(&qk_tensor));
      // Before modifying the parameters below, make sure to read the following
      // description of the CUDNN_TENSOR_NCHW tensor layout, from
      // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnTensorFormat_t:
      // This tensor format specifies that the data is laid out in the following
      // order: batch size, feature maps, rows, columns. The strides are
      // implicitly defined in such a way that the data are contiguous in memory
      // with no padding between images, feature maps, rows, and columns; the
      // columns are the inner dimension and the images are the outermost
      // dimension.
      int n_param = m->num_heads;
      int c_param = total_tokens;
      int h_param = 1;
      int w_param = num_new_tokens;
      checkCUDNN(cudnnSetTensor4dDescriptor(qk_tensor,
                                            CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT,
                                            n_param,
                                            c_param,
                                            h_param,
                                            w_param));
      alpha = 1.0f, beta = 0.0f;
      void *C_softmax = (void *)(m->qk_prods_softmax +
                                 m->num_heads * tokens_prev_requests_squares);
      // The softmax operation below is executed according to the
      // CUDNN_SOFTMAX_MODE_CHANNEL, which is also described in the docs: The
      // softmax operation is computed per spatial location (H,W) per image (N)
      // across dimension C.
      checkCUDNN(cudnnSoftmaxForward(m->handle.dnn,
                                     CUDNN_SOFTMAX_ACCURATE,
                                     CUDNN_SOFTMAX_MODE_CHANNEL,
                                     &alpha,
                                     qk_tensor,
                                     (void *)((float *)C),
                                     &beta,
                                     qk_tensor,
                                     (void *)((float *)C_softmax)));
      // Matmul softmax(QK^T/sqrt(d_k)) by V
      alpha = 1.0f, beta = 0.0f;
      m_ = num_new_tokens;
      n = m->vProjSize;
      k = total_tokens;
      lda = m_, ldb = n, ldc = m_;
      strideA = num_new_tokens * total_tokens;
      strideB = vt_block_size;
      strideC = num_new_tokens * m->vProjSize;
      // To get A, skip over softmax(QK^T/sqrt(d_k)) entries from previous
      // requests (all heads)
      A = (void const *)C_softmax;
      // To get B, skip over V^T entries from previous requests (all heads +
      // padding)
      B = (void const *)(m->valueCache + (i * bc->MAX_BEAM_WIDTH + sub_req_id) *
                                             vt_req_block_size);
      // To get C, skip over softmax(QK^T/sqrt(d_k))V products from previous
      // requests
      C = (void *)(m->attn_heads +
                   tokens_previous_requests * m->num_heads * m->vProjSize);

      checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                           CUBLAS_OP_N,
                                           CUBLAS_OP_T,
                                           m_,
                                           n,
                                           k,
                                           &alpha,
                                           A,
                                           cublas_data_type,
                                           lda,
                                           strideA,
                                           B,
                                           cublas_data_type,
                                           ldb,
                                           strideB,
                                           &beta,
                                           C,
                                           cublas_data_type,
                                           ldc,
                                           strideC,
                                           m->num_heads,
                                           compute_type,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      // Project to output, save result directly on output tensor
      alpha = 1.0f, beta = 0.0f;
      m_ = m->oProjSize;
      k = m->vProjSize * m->num_heads;
      n = num_new_tokens;
      lda = k, ldb = n, ldc = m_;
      A = (void const *)m->W_out_contiguous;
      B = (void const *)C;
      C = (void *)(output_ptr + tokens_previous_requests * m->oProjSize);

      checkCUDA(cublasGemmEx(m->handle.blas,
                             CUBLAS_OP_T,
                             CUBLAS_OP_T,
                             m_,
                             n,
                             k,
                             &alpha,
                             A,
                             cublas_data_type,
                             lda,
                             B,
                             cublas_data_type,
                             ldb,
                             &beta,
                             C,
                             cublas_data_type,
                             ldc,
                             compute_type,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      tokens_previous_requests += num_new_tokens;
      tokens_prev_requests_squares += num_new_tokens * total_tokens;
    }
    if (*m->bias) {
      int parallelism = m->oProjSize * num_tokens;
      spec_apply_proj_bias_w<<<GET_BLOCKS(parallelism),
                               min(CUDA_NUM_THREADS, parallelism),
                               0,
                               stream>>>(
          output_ptr, bias_ptr, num_tokens, m->oProjSize);
    }
  }

  assert(tokens_previous_requests == num_tokens);
}

/*static*/
void SpecIncMultiHeadSelfAttention::inference_kernel_wrapper(
    SpecIncMultiHeadSelfAttentionMeta const *m,
    BeamSearchBatchConfig const *bc,
    float const *input_ptr,
    float const *weight_ptr,
    float *output_ptr,
    float const *bias_ptr) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  // reload the weight_o

  if (!(*m->has_load_weights)) {
    int parallelism = m->vProjSize * m->oProjSize * m->num_heads;
    spec_build_w_out_tensor<<<GET_BLOCKS(parallelism),
                              min(CUDA_NUM_THREADS, parallelism),
                              0,
                              stream>>>(weight_ptr,
                                        m->W_out_contiguous,
                                        m->vProjSize,
                                        m->oProjSize,
                                        m->num_heads,
                                        (m->qSize * m->qProjSize +
                                         m->kSize * m->kProjSize +
                                         m->vSize * m->vProjSize));
    *m->has_load_weights = true;
  }

  // here because we need postion info in infernece 1
  cudaMemcpyAsync(m->token_infos,
                  &(bc->tokensInfo),
                  bc->MAX_NUM_TOKENS * sizeof(BatchConfig::PerTokenInfo),
                  cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(m->request_infos,
                  &(bc->requestsInfo),
                  bc->MAX_NUM_REQUESTS * sizeof(BatchConfig::PerRequestInfo),
                  cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(m->beam_token_infos,
                  &(bc->beamTokenInfo),
                  bc->MAX_NUM_TOKENS * bc->MAX_BEAM_WIDTH *
                      sizeof(BeamSearchBatchConfig::BeamSearchPerTokenInfo),
                  cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(m->beam_request_infos,
                  &(bc->beamRequestsInfo),
                  bc->MAX_NUM_REQUESTS *
                      sizeof(BeamSearchBatchConfig::BeamSearchPerRequestInfo),
                  cudaMemcpyHostToDevice,
                  stream);
  // phase 1: Implement kernel to compute KQV for input tokens
  compute_qkv_kernel(
      m, bc, input_ptr, weight_ptr, m->devQKVProjArray, bias_ptr, stream);
  // phase 2: Update key/val cache
  update_kv_cache_kernel(m, bc, stream);

  // phase 3: Compute attention score
  // 3 kernels for pahse 3: matmul1 - softmax - matmal2
  compute_attention_kernel(m, bc, output_ptr, bias_ptr, stream);

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("SpecIncMultiHeadSelfAttention forward time = %.2fms\n", elapsed);
    // print_tensor<3, float>(acc_query.ptr, acc_query.rect,
    // "[Attention:forward:query]"); print_tensor<3, float>(acc_output.ptr,
    // acc_output.rect, "[Attention:forward:output]");
  }
}

SpecIncMultiHeadSelfAttentionMeta::SpecIncMultiHeadSelfAttentionMeta(
    FFHandler handler,
    SpecIncMultiHeadSelfAttention const *attn,
    float const *weight_ptr,
    Memory gpu_mem,
    int num_samples,
    int _num_heads)
    : IncMultiHeadSelfAttentionMeta(handler,
                                    BEAM_SEARCH_MODE,
                                    attn,
                                    attn->qSize,
                                    attn->kSize,
                                    attn->vSize,
                                    attn->qProjSize,
                                    attn->kProjSize,
                                    attn->vProjSize,
                                    attn->oProjSize,
                                    attn->apply_rotary_embedding,
                                    attn->bias,
                                    attn->scaling_query,
                                    attn->qk_prod_scaling,
                                    attn->add_bias_kv,
                                    attn->scaling_factor,
                                    weight_ptr,
                                    gpu_mem,
                                    num_samples,
                                    _num_heads) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(cudnnSetStream(handler.dnn, stream));

  // allocate memory for the seqArray and reserve space
  {
    size_t beam_tokeninfo_size = BeamSearchBatchConfig::MAX_NUM_TOKENS *
                                 BeamSearchBatchConfig::MAX_BEAM_WIDTH;
    size_t requestinfo_size = BeamSearchBatchConfig::MAX_NUM_REQUESTS;
    size_t beam_requestinfo_size = BeamSearchBatchConfig::MAX_NUM_REQUESTS;
    size_t totalSize =
        requestinfo_size * sizeof(BatchConfig::PerRequestInfo) +
        beam_tokeninfo_size *
            sizeof(BeamSearchBatchConfig::BeamSearchPerTokenInfo) +
        beam_requestinfo_size *
            sizeof(BeamSearchBatchConfig::
                       BeamSearchPerRequestInfo); // more components will
                                                  // be added here later

    Realm::Rect<1, coord_t> bounds(Realm::Point<1, coord_t>(0),
                                   Realm::Point<1, coord_t>(totalSize - 1));
    std::vector<size_t> field_sizes;
    field_sizes.push_back(sizeof(char));
    Realm::RegionInstance::create_instance(beam_search_reserve_inst,
                                           gpu_mem,
                                           bounds,
                                           field_sizes,
                                           0,
                                           Realm::ProfilingRequestSet())
        .wait();
    beam_token_infos =
        (BeamSearchBatchConfig::BeamSearchPerTokenInfo *)
            beam_search_reserve_inst.pointer_untyped(0, sizeof(char));
    request_infos =
        (BatchConfig::PerRequestInfo *)(beam_token_infos + beam_tokeninfo_size);
    beam_request_infos =
        (BeamSearchBatchConfig::BeamSearchPerRequestInfo *)(request_infos +
                                                            requestinfo_size);
  }

  cudaStreamSynchronize(stream);
}

SpecIncMultiHeadSelfAttentionMeta::~SpecIncMultiHeadSelfAttentionMeta(void) {
  beam_search_reserve_inst.destroy();
}

}; // namespace FlexFlow

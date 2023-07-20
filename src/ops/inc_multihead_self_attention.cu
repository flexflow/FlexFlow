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
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/inc_multihead_self_attention.h"
#include "flexflow/ops/kernels/decompress_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_kernels.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

namespace Kernels {
namespace IncMultiHeadAttention {

// template <typename DT>
// __global__ void build_w_out_tensor(DT const *weight_ptr,
//                                    DT *contiguous_weight_ptr,
//                                    int vProjSize,
//                                    int oProjSize,
//                                    int num_heads,
//                                    int qkv_weight_block_size) {
//   CUDA_KERNEL_LOOP(i, vProjSize * oProjSize * num_heads) {
//     // Each slice (one per head) in the weight_ptr has shape (oProjSize,
//     // vProjSize)
//     int row_idx = i % oProjSize;
//     int col_idx = (i / oProjSize) % vProjSize;
//     int head_idx = i / (oProjSize * vProjSize);
//     // The contiguous_weight_ptr has shape (vProjSize * num_heads, oProjSize)
//     int idx = row_idx * vProjSize * num_heads + vProjSize * head_idx +
//     col_idx; contiguous_weight_ptr[idx] =
//         weight_ptr[(qkv_weight_block_size + vProjSize * oProjSize) * head_idx
//         +
//                    qkv_weight_block_size + col_idx * oProjSize + row_idx];
//   }
// }

template <typename DT>
__global__ void apply_proj_bias_w(DT *input_ptr,
                                  DT const *bias_ptr,
                                  int num_tokens,
                                  int oProjSize) {
  CUDA_KERNEL_LOOP(i, num_tokens * oProjSize) {
    int bias_idx = 3 * oProjSize + i % oProjSize;
    input_ptr[i] += bias_ptr[bias_idx];
  }
}

// template <typename DT>
// __global__ void apply_proj_bias_qkv(DT *input_ptr,
//                                     DT const *bias_ptr,
//                                     int shard_id,
//                                     int num_tokens,
//                                     int qProjSize,
//                                     int kProjSize,
//                                     int vProjSize,
//                                     int global_num_heads,
//                                     int num_heads,
//                                     bool scaling_query,
//                                     float scaling_factor) {
//   CUDA_KERNEL_LOOP(
//       i, num_tokens * (qProjSize + kProjSize + vProjSize) * num_heads) {
//     // for simplicity, assume q, k, v is in same shape
//     // 0->q, 1->k, 2->v
//     int qkv_index = i / (num_tokens * qProjSize) % 3;

//     int head_idx = i / (num_tokens * (qProjSize + kProjSize + vProjSize));
//     int qkv_block_size = (qProjSize + kProjSize + vProjSize) * num_tokens;
//     int q_block_size = qProjSize * num_tokens;

//     int idx = i % (num_tokens * (qProjSize));

//     int real_part_index =
//         head_idx * qkv_block_size + qkv_index * q_block_size + idx;

//     int global_head_idx = head_idx + shard_id * num_heads;
//     int bias_idx = qkv_index * qProjSize * global_num_heads +
//                    global_head_idx * qProjSize + (idx % qProjSize);
//     input_ptr[real_part_index] += bias_ptr[bias_idx];

//     if (scaling_query && qkv_index == 0) {
//       input_ptr[real_part_index] *= scaling_factor;
//     }
//   }
// }

template <typename DT>
__global__ void apply_proj_bias_qkv(DT *input_ptr,
                                    DT const *bias_ptr,
                                    int shard_id,
                                    int num_tokens,
                                    int qProjSize,
                                    int kProjSize,
                                    int vProjSize,
                                    int global_num_heads,
                                    int num_heads,
                                    int num_kv_heads,
                                    bool scaling_query,
                                    float scaling_factor) {
  CUDA_KERNEL_LOOP(i,
                   num_tokens *
                       (qProjSize * num_heads + kProjSize * num_kv_heads +
                        vProjSize * num_kv_heads)) {
    // for simplicity, assume q, k, v is in same shape
    // 0->q, 1->k, 2->v
    int qkv_index = i / (num_tokens * qProjSize) % 3;

    int head_idx = i / (num_tokens * (qProjSize + kProjSize + vProjSize));
    int qkv_block_size = (qProjSize + kProjSize + vProjSize) * num_tokens;
    int q_block_size = qProjSize * num_tokens;

    int idx = i % (num_tokens * (qProjSize));

    int real_part_index =
        head_idx * qkv_block_size + qkv_index * q_block_size + idx;

    int global_head_idx = head_idx + shard_id * num_heads;
    int bias_idx = qkv_index * qProjSize * global_num_heads +
                   global_head_idx * qProjSize + (idx % qProjSize);
    input_ptr[real_part_index] += bias_ptr[bias_idx];

    if (scaling_query && qkv_index == 0) {
      input_ptr[real_part_index] *= scaling_factor;
    }
  }
}

template <typename DT>
__global__ void
    apply_rotary_embedding(DT *input_ptr,
                           cuFloatComplex *complex_input,
                           BatchConfig::PerTokenInfo const *tokenInfos,
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

template <typename DT>
void compute_qkv_kernel(IncMultiHeadSelfAttentionMeta const *m,
                        BatchConfig const *bc,
                        int shard_id,
                        DT const *input_ptr,
                        DT const *weight_ptr,
                        DT *output_ptr,
                        DT const *bias_ptr,
                        cudaStream_t stream) {

  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  DT alpha = 1.0f, beta = 0.0f;
  assert(m->qSize == m->vSize && m->qSize == m->kSize);
  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(m->output_type[0]);
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = cublas_data_type;
#endif
  // Compute (W^T)x matmul: einsum(ijkl,im->jmkl)
  // Weights: qSize x qProjSize x 3 x num_heads
  // Input: qSize x num_tokens
  // Output >>> qProjSize x num_tokens x 3 x num_heads
  int m_q = m->qProjSize * m->embed_dim;
  int m_k = m->kProjSize;
  int m_v = m->vProjSize;
  assert(m_q == m_k && m_k == m_v); // keep things simple for now
  int n = bc->num_active_tokens();
  int k_q = m->qSize, k_k = m->kProjSize, k_v = m->vProjSize;
  int lda = k, ldb = k, ldc_q = m_q, ldc_k = m_k, ldc_v = m_v;
  // Q
  checkCUDA(cublasGemmEx(m->handle.blas,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         m_q,
                         n,
                         k_q,
                         &alpha,
                         weight_ptr,
                         cublas_data_type,
                         lda,
                         input_ptr,
                         cublas_data_type,
                         ldb,
                         &beta,
                         output_ptr,
                         cublas_data_type,
                         ldc_q,
                         compute_type,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // key
  checkCUDA(cublasGemmEx(m->handle.blas,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         m_k,
                         n,
                         k_k,
                         &alpha,
                         weight_ptr + m->embed_dim * m->embed_dim,
                         cublas_data_type,
                         lda,
                         input_ptr,
                         cublas_data_type,
                         ldb,
                         &beta,
                         output_ptr + num_tokens * m->embed_dim,
                         cublas_data_type,
                         ldc_k,
                         compute_type,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // value
  checkCUDA(cublasGemmEx(
      m->handle.blas,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      m_v,
      n,
      k_v,
      &alpha,
      weight_ptr +
          m->embed_dim * (m->embed_dim + m->kProjSize * m->num_kv_heads),
      cublas_data_type,
      lda,
      input_ptr,
      cublas_data_type,
      ldb,
      &beta,
      output_ptr + num_tokens * (m->embed_dim + m->kProjSize * m->num_kv_heads),
      cublas_data_type,
      ldc_v,
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
    apply_proj_bias_qkv<<<GET_BLOCKS(parallelism),
                          min(CUDA_NUM_THREADS, parallelism),
                          0,
                          stream>>>(output_ptr,
                                    bias_ptr,
                                    shard_id,
                                    num_tokens,
                                    m->qProjSize,
                                    m->kProjSize,
                                    m->vProjSize,
                                    m->global_num_heads,
                                    m->num_heads,
                                    *m->scaling_query,
                                    m->scaling_factor);
  }

  if (*m->apply_rotary_embedding) {
    /*q*/
    apply_rotary_embedding<<<GET_BLOCKS(parallelism),
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
    apply_rotary_embedding<<<GET_BLOCKS(parallelism),
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
}

template <typename DT>
void update_kv_cache_kernel(IncMultiHeadSelfAttentionMeta const *m,
                            BatchConfig const *bc,
                            cudaStream_t stream) {
  int num_tokens = bc->num_active_tokens();
  if (num_tokens > 0) {
    int parallelism = m->kProjSize * num_tokens * m->num_heads;
    store_kv_cache<<<GET_BLOCKS(parallelism),
                     min(CUDA_NUM_THREADS, parallelism),
                     0,
                     stream>>>(static_cast<DT *>(m->devQKVProjArray),
                               static_cast<DT *>(m->keyCache),
                               m->token_infos,
                               m->qProjSize,
                               m->kProjSize,
                               m->vProjSize,
                               num_tokens,
                               m->num_heads,
                               BatchConfig::MAX_SEQ_LENGTH,
                               /* k_cache = */ true);

    parallelism = m->vProjSize * num_tokens * m->num_heads;
    store_kv_cache<<<GET_BLOCKS(parallelism),
                     min(CUDA_NUM_THREADS, parallelism),
                     0,
                     stream>>>(static_cast<DT *>(m->devQKVProjArray),
                               static_cast<DT *>(m->valueCache),
                               m->token_infos,
                               m->qProjSize,
                               m->kProjSize,
                               m->vProjSize,
                               num_tokens,
                               m->num_heads,
                               BatchConfig::MAX_SEQ_LENGTH,
                               /* k_cache = */ false);
  }
}

template <typename DT>
void pre_build_weight_kernel(IncMultiHeadSelfAttentionMeta const *m,
                             GenericTensorAccessorR const weight,
                             DataType data_type,
                             cudaStream_t stream) {
  // additional processing for weight uploading
  // Note that we update weight_ptr and bias_ptr when uploading weight and
  // bias
  if (m->quantization_type != DT_NONE) {
    // copy weight_ptr to quantized_weight_ptr, do compression and store in
    // m->weight_ptr
    cudaMemcpyAsync(m->quantized_weight_ptr,
                    weight.get_byte_ptr(),
                    m->quantized_weightSize,
                    cudaMemcpyHostToDevice,
                    stream);

    if (m->quantization_type == DT_INT4) {
      int parallelism = m->qProjSize * m->qSize * m->num_heads / 2;
      decompress_int4_attention_weights<<<GET_BLOCKS(parallelism),
                                          min(CUDA_NUM_THREADS, parallelism),
                                          0,
                                          stream>>>(
          m->quantized_weight_ptr,
          static_cast<DT *>(m->weight_ptr),
          m->qProjSize,
          m->qSize,
          m->num_heads);
    } else {
      assert(m->quantization_type == DT_INT8);
      int parallelism = m->qProjSize * m->qSize * m->num_heads;
      decompress_int8_attention_weights<<<GET_BLOCKS(parallelism),
                                          min(CUDA_NUM_THREADS, parallelism),
                                          0,
                                          stream>>>(
          m->quantized_weight_ptr,
          static_cast<DT *>(m->weight_ptr),
          m->qProjSize,
          m->qSize,
          m->num_heads);
    }
  } else {
    if (data_type == DT_FLOAT) {
      cudaMemcpyAsync(m->weight_ptr,
                      weight.get_float_ptr(),
                      m->weightSize,
                      cudaMemcpyHostToDevice,
                      stream);
    } else if (data_type == DT_HALF) {
      cudaMemcpyAsync(m->weight_ptr,
                      weight.get_half_ptr(),
                      m->weightSize,
                      cudaMemcpyHostToDevice,
                      stream);
    } else {
      assert(false);
    }
  }
  // // reload weight_o for offloading case
  // int parallelism = m->vProjSize * m->oProjSize * m->num_heads;
  // build_w_out_tensor<<<GET_BLOCKS(parallelism),
  //                      min(CUDA_NUM_THREADS, parallelism),
  //                      0,
  //                      stream>>>(static_cast<DT *>(m->weight_ptr),
  //                                static_cast<DT *>(m->W_out_contiguous),
  //                                m->vProjSize,
  //                                m->oProjSize,
  //                                m->num_heads,
  //                                (m->qSize * m->qProjSize +
  //                                 m->kSize * m->kProjSize +
  //                                 m->vSize * m->vProjSize));
}

template <typename DT>
void inference_kernel(IncMultiHeadSelfAttentionMeta const *m,
                      BatchConfig const *bc,
                      int shard_id,
                      DT const *input_ptr,
                      DT const *weight_ptr,
                      DT *output_ptr,
                      DT const *bias_ptr,
                      cudaStream_t stream) {
  // here because we need postion info in infernece 1

  if (m->offload && m->biasSize > 0) {
    cudaMemcpyAsync(
        m->bias_ptr, bias_ptr, m->biasSize, cudaMemcpyHostToDevice, stream);
    bias_ptr = static_cast<DT *>(m->bias_ptr);
  }
  cudaMemcpyAsync(m->token_infos,
                  &(bc->tokensInfo),
                  bc->num_active_tokens() * sizeof(BatchConfig::PerTokenInfo),
                  cudaMemcpyHostToDevice,
                  stream);
  // phase 1: Implement kernel to compute KQV for input tokens
  compute_qkv_kernel(m,
                     bc,
                     shard_id,
                     input_ptr,
                     weight_ptr,
                     static_cast<DT *>(m->devQKVProjArray),
                     bias_ptr,
                     stream);

  // phase 2: Update key/val cache
  update_kv_cache_kernel<DT>(m, bc, stream);

  // phase 3: Compute attention score
  // 3 kernels for pahse 3: matmul1 - softmax - matmal2
  compute_attention_kernel(m, bc, shard_id, output_ptr, bias_ptr, stream);
}

} // namespace IncMultiHeadAttention
} // namespace Kernels

using namespace Kernels::IncMultiHeadAttention;

template <typename DT>
__global__ void store_kv_cache(DT const *devQKVProjArray,
                               DT *cache_ptr,
                               BatchConfig::PerTokenInfo const *tokenInfos,
                               int qProjSize,
                               int kProjSize,
                               int vProjSize,
                               int num_tokens,
                               int num_heads,
                               int max_seq_len,
                               bool k_cache) {
  CUDA_KERNEL_LOOP(i,
                   num_tokens * (k_cache ? kProjSize : vProjSize) * num_heads) {
    int proj_size = k_cache ? kProjSize : vProjSize;
    int head_idx = i / (num_tokens * proj_size);
    int token_idx = (i - head_idx * (num_tokens * proj_size)) / proj_size;
    int data_idx = i % proj_size;

    int qkv_block_size = (qProjSize + kProjSize + vProjSize) * num_tokens;
    int current_head_block_size =
        num_tokens * (k_cache ? qProjSize : qProjSize + kProjSize);
    DT val =
        devQKVProjArray[head_idx * qkv_block_size + current_head_block_size +
                        token_idx * proj_size + data_idx];
    // int const req_id = id_map[token_idx].request_index;
    // int const tok_id = id_map[token_idx].token_position;
    int const req_id = tokenInfos[token_idx].request_index;
    int const tok_id = tokenInfos[token_idx].abs_depth_in_request;

    cache_ptr[req_id * (num_heads * max_seq_len * proj_size) +
              head_idx * (max_seq_len * proj_size) + tok_id * proj_size +
              data_idx] = val;
  }
}

template <typename DT>
__global__ void fill_entries_above_diagonal(DT *matrix,
                                            size_t num_rows,
                                            size_t num_cols,
                                            size_t num_heads,
                                            size_t entries_above_diagonal,
                                            DT value) {
  CUDA_KERNEL_LOOP(i, entries_above_diagonal * num_heads) {
    size_t head_idx = i / entries_above_diagonal;
    size_t entry_idx = i % entries_above_diagonal;
    size_t y = (-1 + sqrt(8 * (float)entry_idx + 1)) / 2;
    size_t x = entry_idx - y * (y + 1) / 2;
    y += (num_cols - num_rows) + 1;
    matrix[head_idx * num_rows * num_cols + num_cols * y + x] = value;
  }
}

template <typename DT>
void compute_attention_kernel(IncMultiHeadSelfAttentionMeta const *m,
                              BatchConfig const *bc,
                              int shard_id,
                              DT *output_ptr,
                              DT const *bias_ptr,
                              cudaStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  assert(data_type_size(m->output_type[0]) == sizeof(DT));
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  cudaDataType_t compute_type = cublas_data_type;
#endif
  // int num_requests = bc->num_active_requests();
  int num_tokens = bc->num_active_tokens();
  int tokens_previous_requests = 0;
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
    int num_new_tokens = bc->requestsInfo[i].num_tokens_in_batch;
    int total_tokens = bc->requestsInfo[i].token_start_offset +
                       bc->requestsInfo[i].num_tokens_in_batch;
    // bc->token_last_available_idx[i] + 1;
    // Compute (QK^T/sqrt(d_k))
    int m_ = num_new_tokens;
    int n = total_tokens;
    int k = m->qProjSize;
    int lda = k, ldb = k, ldc = m_;
    int strideA = qkv_block_size;
    int strideB = kt_block_size;
    int strideC = num_new_tokens * total_tokens;

    // a flag of using this scaling alpha
    DT alpha = 1.0f, beta = 0.0f;
    if (*m->qk_prod_scaling) {
      alpha = static_cast<DT>(1.0f / sqrt(m->kProjSize));
    }
    // To get A, skip over Q entries from previous requests (same head)
    void const *A = static_cast<DT *>(m->devQKVProjArray) +
                    tokens_previous_requests * m->qProjSize;
    // To get B, skip over K entries from previous requests (all heads +
    // padding)
    void const *B = static_cast<DT *>(m->keyCache) + i * kt_req_block_size;
    // To get C, skip over QK^T products from previous requests
    void *C = (void *)(m->qk_prods);

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
    size_t entries_above_diagonal = num_new_tokens * (num_new_tokens - 1) / 2;
    if (entries_above_diagonal > 0) {
      size_t parallelism = m->num_heads * entries_above_diagonal;
      fill_entries_above_diagonal<<<GET_BLOCKS(parallelism),
                                    min((size_t)CUDA_NUM_THREADS, parallelism),
                                    0,
                                    stream>>>(static_cast<DT *>(C),
                                              num_new_tokens,
                                              total_tokens,
                                              m->num_heads,
                                              entries_above_diagonal,
                                              static_cast<DT>(-INFINITY));
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
                                          cudnn_data_type,
                                          n_param,
                                          c_param,
                                          h_param,
                                          w_param));
    float softmax_alpha = 1.0f, softmax_beta = 0.0f;
    void *C_softmax = (void *)(m->qk_prods_softmax);
    // The softmax operation below is executed according to the
    // CUDNN_SOFTMAX_MODE_CHANNEL, which is also described in the docs: The
    // softmax operation is computed per spatial location (H,W) per image (N)
    // across dimension C.
    checkCUDNN(cudnnSoftmaxForward(m->handle.dnn,
                                   CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_CHANNEL,
                                   &softmax_alpha,
                                   qk_tensor,
                                   C,
                                   &softmax_beta,
                                   qk_tensor,
                                   C_softmax));
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
    A = static_cast<DT *>(C_softmax);
    // To get B, skip over V^T entries from previous requests (all heads +
    // padding)
    B = static_cast<DT *>(m->valueCache) + i * vt_req_block_size;
    // To get C, skip over softmax(QK^T/sqrt(d_k))V products from previous
    // requests
    C = static_cast<DT *>(m->attn_heads) +
        tokens_previous_requests * m->num_heads * m->vProjSize;

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
    A = static_cast<DT *>(m->W_out_contiguous);
    B = C;
    C = static_cast<DT *>(output_ptr) + tokens_previous_requests * m->oProjSize;

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
  }

  if (*m->bias && shard_id == 0) {
    int parallelism = m->oProjSize * num_tokens;
    apply_proj_bias_w<<<GET_BLOCKS(parallelism),
                        min(CUDA_NUM_THREADS, parallelism),
                        0,
                        stream>>>(
        output_ptr, bias_ptr, num_tokens, m->oProjSize);
  }

  assert(tokens_previous_requests == num_tokens);
}

/*static*/
void IncMultiHeadSelfAttention::inference_kernel_wrapper(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    int shard_id,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &weight,
    GenericTensorAccessorW const &output,
    GenericTensorAccessorR const &bias) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  bool use_bias = *m->bias;

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  // assert(input.data_type == weight.data_type);
  assert(input.data_type == output.data_type);
  if (use_bias) {
    assert(input.data_type == bias.data_type);
  }

  if (input.data_type == DT_HALF) {
    if (m->offload) {
      pre_build_weight_kernel<half>(m, weight, input.data_type, stream);
    }
    half const *bias_ptr =
        use_bias ? bias.get_half_ptr() : static_cast<half const *>(nullptr);
    Kernels::IncMultiHeadAttention::inference_kernel(
        m,
        bc,
        shard_id,
        input.get_half_ptr(),
        m->offload ? static_cast<half *>(m->weight_ptr) : weight.get_half_ptr(),
        output.get_half_ptr(),
        bias_ptr,
        stream);
  } else if (input.data_type == DT_FLOAT) {
    if (m->offload) {
      pre_build_weight_kernel<float>(m, weight, input.data_type, stream);
    }
    float const *bias_ptr =
        use_bias ? bias.get_float_ptr() : static_cast<float const *>(nullptr);
    Kernels::IncMultiHeadAttention::inference_kernel(
        m,
        bc,
        shard_id,
        input.get_float_ptr(),
        m->offload ? static_cast<float *>(m->weight_ptr)
                   : weight.get_float_ptr(),
        output.get_float_ptr(),
        bias_ptr,
        stream);
  } else {
    assert(false && "Unspported data type");
  }
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
    GenericTensorAccessorR const &weight,
    MemoryAllocator &gpu_mem_allocator,
    int num_samples,
    int _num_heads)
    : IncMultiHeadSelfAttentionMeta(handler,
                                    INC_DECODING_MODE,
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
                                    weight,
                                    gpu_mem_allocator,
                                    num_samples,
                                    attn->num_heads,
                                    _num_heads,
                                    attn->num_kv_heads,
                                    attn->quantization_type,
                                    attn->offload) {}

IncMultiHeadSelfAttentionMeta::IncMultiHeadSelfAttentionMeta(
    FFHandler handler,
    InferenceMode infer_mode,
    Op const *attn,
    int _qSize,
    int _kSize,
    int _vSize,
    int _qProjSize,
    int _kProjSize,
    int _vProjSize,
    int _oProjSize,
    bool _apply_rotary_embedding,
    bool _bias,
    bool _scaling_query,
    bool _qk_prod_scaling,
    bool _add_bias_kv,
    float _scaling_factor,
    GenericTensorAccessorR const &weight,
    MemoryAllocator &gpu_mem_allocator,
    int num_samples,
    int _global_num_heads,
    int _num_heads,
    int _num_kv_heads,
    DataType _quantization_type,
    bool _offload)
    : OpMeta(handler, attn), weight_ptr(nullptr), bias_ptr(nullptr) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(cudnnSetStream(handler.dnn, stream));
  qSize = _qSize;
  kSize = _kSize;
  vSize = _vSize;
  // assume dimensions match for now
  assert(qSize == kSize);
  assert(kSize == vSize);
  qProjSize = _qProjSize;
  kProjSize = _kProjSize;
  assert(qProjSize == kProjSize); // required for attention QK^T matmul
  vProjSize = _vProjSize;
  oProjSize = _oProjSize;
  size_t size_of_dt = data_type_size(attn->data_type);
  quantization_type = _quantization_type;
  offload = _offload;

  global_num_heads = _global_num_heads;
  num_heads = _num_heads;
  num_kv_heads = _num_kv_heads;
  weights_params = (qSize * qProjSize + kSize * kProjSize + vSize * vProjSize +
                    oProjSize * (vProjSize > 0 ? vProjSize : vSize));
  weightSize = weights_params * num_heads * size_of_dt;
  if (quantization_type != DT_NONE) {
    quantized_weightSize = get_quantization_to_byte_size(
        attn->data_type, quantization_type, weightSize);
  }
  biasSize = _bias ? oProjSize * size_of_dt * 4 : 0;
  // has_load_weights = (bool *)calloc(1, sizeof(bool));
  //*has_load_weights = false;
  apply_rotary_embedding = (bool *)calloc(1, sizeof(bool));
  *apply_rotary_embedding = _apply_rotary_embedding;
  bias = (bool *)calloc(1, sizeof(bool));
  *bias = _bias;
  scaling_query = (bool *)calloc(1, sizeof(bool));
  *scaling_query = _scaling_query;
  scaling_factor = _scaling_factor;
  qk_prod_scaling = (bool *)calloc(1, sizeof(bool));
  *qk_prod_scaling = _qk_prod_scaling;
  // Currently do not support adding bias to key/value projection
  assert(!_add_bias_kv);

  // allocate weight and bias in the reserve space for cpu offloading
  if (offload) {
    weight_ptr = gpu_mem_allocator.allocate_reserved_untyped(weightSize);
    bias_ptr = gpu_mem_allocator.allocate_reserved_untyped(biasSize);
  }

#ifdef INFERENCE_TESTS
  kcache = (float *)calloc(kProjSize * BatchConfig::MAX_SEQ_LENGTH * num_heads *
                               BatchConfig::MAX_NUM_REQUESTS,
                           sizeof(float));
  vcache = (float *)calloc(vProjSize * BatchConfig::MAX_SEQ_LENGTH * num_heads *
                               BatchConfig::MAX_NUM_REQUESTS,
                           sizeof(float));
#endif

  // allocate memory for the seqArray and reserve space
  {
    // size_t qkv_proj_dim = qProjSize + kProjSize + vProjSize;
    // size_t qkv_max_proj_size =
    //     BatchConfig::MAX_NUM_TOKENS * qkv_proj_dim * num_heads;

    size_t qkv_max_proj_size =
        BatchConfig::MAX_NUM_TOKENS *
        (qProjSize * num_heads + kProjSize * num_kv_heads,
         vProjSize * num_kv_heads);
    size_t key_cache_size = 0, value_cache_size = 0;
    switch (infer_mode) {
      case INC_DECODING_MODE:
      case TREE_VERIFY_MODE: {
        key_cache_size = num_kv_heads * kProjSize *
                         BatchConfig::MAX_NUM_REQUESTS *
                         BatchConfig::MAX_SEQ_LENGTH;
        value_cache_size = num_kv_heads * vProjSize *
                           BatchConfig::MAX_NUM_REQUESTS *
                           BatchConfig::MAX_SEQ_LENGTH;
        break;
      }
      case BEAM_SEARCH_MODE: {
        key_cache_size =
            num_kv_heads * kProjSize * BeamSearchBatchConfig::MAX_NUM_REQUESTS *
            BatchConfig::MAX_SEQ_LENGTH * BeamSearchBatchConfig::MAX_BEAM_WIDTH;
        value_cache_size =
            num_kv_heads * vProjSize * BeamSearchBatchConfig::MAX_NUM_REQUESTS *
            BatchConfig::MAX_SEQ_LENGTH * BeamSearchBatchConfig::MAX_BEAM_WIDTH;
        break;
      }
      default:
        assert(false && "Unkown inference mode");
    }
    size_t tokeninfo_size = BatchConfig::MAX_NUM_TOKENS;
    size_t qk_prod_size =
        BatchConfig::MAX_NUM_TOKENS * BatchConfig::MAX_SEQ_LENGTH * num_heads;
    size_t attn_heads_size =
        BatchConfig::MAX_NUM_TOKENS * num_heads * vProjSize;
    size_t W_out_block_size = oProjSize * (vProjSize > 0 ? vProjSize : vSize);
    size_t W_out_contiguous_size = W_out_block_size * num_heads;
    size_t complex_size =
        (BatchConfig::MAX_NUM_TOKENS * qProjSize * num_heads) / 2;
    size_t totalSize =
        (qkv_max_proj_size + key_cache_size + value_cache_size +
         2 * qk_prod_size + attn_heads_size + W_out_contiguous_size) *
            size_of_dt +
        tokeninfo_size * sizeof(BatchConfig::PerTokenInfo) +
        complex_size * sizeof(cuFloatComplex); // more components will
                                               // be added here later
    if (offload) {
      // assert that we have enough reserved work space left
      size_t totalSharedSize =
          infer_mode == TREE_VERIFY_MODE
              ? totalSize -
                    (key_cache_size + value_cache_size + qkv_max_proj_size) *
                        size_of_dt
              : totalSize - (key_cache_size + value_cache_size) * size_of_dt;

      size_t instance_size =
          size_of_dt *
          (infer_mode == TREE_VERIFY_MODE
               ? key_cache_size + value_cache_size + qkv_max_proj_size
               : key_cache_size + value_cache_size);

      if (quantization_type != DT_NONE) {
        totalSharedSize += quantized_weightSize;
      }
      assert(gpu_mem_allocator.reserved_total_size -
                 gpu_mem_allocator.reserved_allocated_size >=
             totalSharedSize);
      gpu_mem_allocator.create_legion_instance(reserveInst, instance_size);
    } else {
      gpu_mem_allocator.create_legion_instance(reserveInst, totalSize);
    }

    // in tree_verify, enable devQKVProjArray;
    if (!offload || infer_mode == TREE_VERIFY_MODE) {
      devQKVProjArray = gpu_mem_allocator.allocate_instance_untyped(
          qkv_max_proj_size * size_of_dt);
    } else {
      devQKVProjArray = gpu_mem_allocator.allocate_reserved_untyped(
          qkv_max_proj_size * size_of_dt);
      // offset += qkv_max_proj_size * size_of_dt;
    }

    // use key value cache in all mode.
    keyCache = gpu_mem_allocator.allocate_instance_untyped(key_cache_size *
                                                           size_of_dt);
    valueCache = gpu_mem_allocator.allocate_instance_untyped(value_cache_size *
                                                             size_of_dt);

    if (offload) {
      token_infos =
          gpu_mem_allocator.allocate_reserved<BatchConfig::PerTokenInfo>(
              tokeninfo_size);
      // offset += sizeof(BatchConfig::PerTokenInfo) * tokeninfo_size;
      qk_prods = gpu_mem_allocator.allocate_reserved_untyped(qk_prod_size *
                                                             size_of_dt);
      // offset += qk_prod_size * size_of_dt;
      qk_prods_softmax = gpu_mem_allocator.allocate_reserved_untyped(
          qk_prod_size * size_of_dt);
      // offset += qk_prod_size * size_of_dt;
      attn_heads = gpu_mem_allocator.allocate_reserved_untyped(attn_heads_size *
                                                               size_of_dt);
      // offset += attn_heads_size * size_of_dt;
      W_out_contiguous = gpu_mem_allocator.allocate_reserved_untyped(
          W_out_contiguous_size * size_of_dt);
      // offset += W_out_contiguous_size * size_of_dt;
      complex_input =
          gpu_mem_allocator.allocate_reserved<cuFloatComplex>(complex_size);
      // offset += complex_size * sizeof(cuFloatComplex);
    } else {
      token_infos =
          gpu_mem_allocator.allocate_instance<BatchConfig::PerTokenInfo>(
              tokeninfo_size);
      qk_prods = gpu_mem_allocator.allocate_instance_untyped(qk_prod_size *
                                                             size_of_dt);
      qk_prods_softmax = gpu_mem_allocator.allocate_instance_untyped(
          qk_prod_size * size_of_dt);
      attn_heads = gpu_mem_allocator.allocate_instance_untyped(attn_heads_size *
                                                               size_of_dt);
      W_out_contiguous = gpu_mem_allocator.allocate_instance_untyped(
          W_out_contiguous_size * size_of_dt);
      complex_input =
          gpu_mem_allocator.allocate_instance<cuFloatComplex>(complex_size);
    }

    // allocate more size for quantization data
    if (quantization_type != DT_NONE) {
      assert(offload);
      quantized_weight_ptr =
          gpu_mem_allocator.allocate_reserved<char>(quantized_weightSize);
    }

    // if (weight.data_type == DT_FLOAT) {
    //   int parallelism = vProjSize * oProjSize * num_heads;
    //   build_w_out_tensor<<<GET_BLOCKS(parallelism),
    //                        min(CUDA_NUM_THREADS, parallelism),
    //                        0,
    //                        stream>>>(
    //       weight.get_float_ptr(),
    //       (float *)W_out_contiguous,
    //       vProjSize,
    //       oProjSize,
    //       num_heads,
    //       (qSize * qProjSize + kSize * kProjSize + vSize * vProjSize));
    // } else if (weight.data_type == DT_HALF) {
    //   int parallelism = vProjSize * oProjSize * num_heads;
    //   build_w_out_tensor<<<GET_BLOCKS(parallelism),
    //                        min(CUDA_NUM_THREADS, parallelism),
    //                        0,
    //                        stream>>>(
    //       weight.get_half_ptr(),
    //       (half *)W_out_contiguous,
    //       vProjSize,
    //       oProjSize,
    //       num_heads,
    //       (qSize * qProjSize + kSize * kProjSize + vSize * vProjSize));
    // } else {
    //   assert(weight.data_type == DT_INT4 || weight.data_type == DT_INT8);
    // }
    if (!offload) {
      assert(gpu_mem_allocator.reserved_total_size ==
             gpu_mem_allocator.reserved_allocated_size);
    }
  }
  cudaStreamSynchronize(stream);
}

IncMultiHeadSelfAttentionMeta::~IncMultiHeadSelfAttentionMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
#ifdef INFERENCE_TESTS
  free(kcache);
  free(vcache);
#endif
}

template void Kernels::IncMultiHeadAttention::pre_build_weight_kernel<float>(
    IncMultiHeadSelfAttentionMeta const *m,
    GenericTensorAccessorR const weight,
    DataType data_type,
    cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::pre_build_weight_kernel<half>(
    IncMultiHeadSelfAttentionMeta const *m,
    GenericTensorAccessorR const weight,
    DataType data_type,
    cudaStream_t stream);

}; // namespace FlexFlow

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
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/kernels/decompress_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_kernels.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

namespace Kernels {
namespace IncMultiHeadAttention {

// only used by MPT model. https://arxiv.org/abs/2108.12409
template <typename DT>
__global__ void apply_position_bias_qkprd(DT *input_ptr,
                                          int num_tokens,
                                          int num_total_tokens,
                                          int num_heads,
                                          int global_num_q_heads,
                                          int shard_id) {
  CUDA_KERNEL_LOOP(i, num_tokens * num_total_tokens * num_heads) {
    // get head_idx,
    int head_idx = i / (num_tokens * num_total_tokens) + (num_heads * shard_id);
    int position_idx = (i / num_tokens) % num_total_tokens;
    position_idx = position_idx + 1 - num_total_tokens;
    // 8 is alibi_bias_max in
    // https://huggingface.co/mosaicml/mpt-30b/blob/main/config.json
    float base = (float)(head_idx + 1) * 8 / global_num_q_heads;
    float slopes = 1.0 / pow(2, base);
    // if(i == 0){
    //   printf("see position: %d, %f, %f, %f\n", position_idx, base, slopes,
    //   position_idx * slopes);
    // }
    input_ptr[i] += static_cast<DT>(position_idx * slopes);
  }
}

template <typename DT>
__global__ void apply_proj_bias_w(DT *input_ptr,
                                  DT const *bias_ptr,
                                  int num_tokens,
                                  int qkv_weight_size,
                                  int oProjSize) {
  CUDA_KERNEL_LOOP(i, num_tokens * oProjSize) {
    int bias_idx = qkv_weight_size + i % oProjSize;
    input_ptr[i] += bias_ptr[bias_idx];
  }
}

template <typename DT>
__global__ void apply_proj_bias_qkv(DT *input_ptr,
                                    DT const *bias_ptr,
                                    int shard_id,
                                    int num_tokens,
                                    int qProjSize,
                                    int kProjSize,
                                    int vProjSize,
                                    int global_num_q_heads,
                                    int num_q_heads,
                                    bool scaling_query,
                                    float scaling_factor,
                                    int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size * QKV_WEIGHT_NUM) {
    // for simplicity, assume q, k, v is in same shape
    // 0->q, 1->k, 2->v
    // int qkv_index = i / (num_tokens * qProjSize) % 3;

    int token_idx = i / (hidden_size * QKV_WEIGHT_NUM);
    size_t in_token_idx = i - token_idx * hidden_size * 3;
    int qkv_index = in_token_idx / hidden_size;
    int proj_size = qkv_index == 0 ? qProjSize : kProjSize;

    int head_idx =
        (in_token_idx - qkv_index * num_q_heads * proj_size) / proj_size;
    int global_head_idx = head_idx + shard_id * num_q_heads;

    size_t pre_length =
        qkv_index == 0
            ? 0
            : (qkv_index == 1 ? qProjSize * global_num_q_heads
                              : qProjSize * global_num_q_heads * KV_WEIGHT_NUM);

    size_t bias_idx = pre_length + global_head_idx * proj_size + i % proj_size;

    input_ptr[i] += bias_ptr[bias_idx];

    if (scaling_query && qkv_index == 0) {
      input_ptr[i] *= scaling_factor;
    }
  }
}
template <typename DT>
__global__ void scaling_query_kernel(DT *input_ptr,
                                     int qProjSize,
                                     int num_tokens,
                                     int num_q_heads,
                                     float scaling_factor,
                                     int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size) {
    int token_idx = i / hidden_size;
    input_ptr[i % hidden_size + token_idx * hidden_size * QKV_WEIGHT_NUM] *=
        scaling_factor;
  }
}

template <typename DT>
__global__ void
    apply_rotary_embedding_native(DT *input_ptr,
                                  hipFloatComplex *complex_input,
                                  BatchConfig::PerTokenInfo const *tokenInfos,
                                  int qProjSize,
                                  int kProjSize,
                                  int num_q_heads,
                                  int num_tokens,
                                  int num_kv_heads,
                                  int q_block_size,
                                  int k_block_size,
                                  int q_array_size) {
  CUDA_KERNEL_LOOP(
      i,
      num_tokens * (qProjSize * num_q_heads + kProjSize * num_kv_heads) / 2) {
    // create complex number
    bool q_tensor = i < (q_array_size / 2);
    int proj_size = q_tensor ? qProjSize : kProjSize;
    int real_i = q_tensor ? i : i - q_array_size / 2;

    int head_idx = real_i / (num_tokens * proj_size / 2);
    int idx = real_i % (num_tokens * proj_size / 2);
    int real_part_index = idx * 2 +
                          head_idx * (q_tensor ? q_block_size : k_block_size) +
                          (q_tensor ? 0 : q_array_size);

    int complex_part_index = real_part_index + 1;

    complex_input[i] = {input_ptr[real_part_index],
                        input_ptr[complex_part_index]};

    int token_idx =
        (real_i - head_idx * (num_tokens * proj_size / 2)) / (proj_size / 2);
    size_t pos = tokenInfos[token_idx].abs_depth_in_request;
    int pos_i = real_i % (proj_size / 2);
    float freq = pos * (1.0 / pow(10000.0, (float)2 * pos_i / proj_size));
    hipFloatComplex complex_pos = {cos(freq), sin(freq)};

    complex_input[i] = hipCmulf(complex_input[i], complex_pos);
    input_ptr[real_part_index] = complex_input[i].x;
    input_ptr[complex_part_index] = complex_input[i].y;
  }
}

template <typename DT>
__global__ void
    apply_rotary_embedding_hf(DT *input_ptr,
                              hipFloatComplex *complex_input,
                              BatchConfig::PerTokenInfo const *tokenInfos,
                              int qProjSize,
                              int kProjSize,
                              int num_tokens,
                              size_t q_array_size,
                              int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size) {
    // create complex number
    bool q_tensor = i < (q_array_size / 2);
    int proj_size = q_tensor ? qProjSize : kProjSize;
    int real_i = q_tensor ? i : i - q_array_size / 2;

    int token_idx = real_i / (hidden_size / 2);
    int idx = real_i % (proj_size / 2);
    int head_idx = (real_i - (token_idx * (hidden_size / 2))) / (proj_size / 2);

    int real_part_index = idx + head_idx * proj_size +
                          token_idx * hidden_size * 3 +
                          hidden_size * (q_tensor ? 0 : 1);
    int complex_part_index = real_part_index + (proj_size / 2);

    complex_input[i] = {input_ptr[real_part_index],
                        input_ptr[complex_part_index]};

    // get the freq_cis: shape 1 * (qProjSize/2) = 1 * 64
    // apply a Cartesian coordinate transformation
    // multiple with input & /copy back to q/k

    // get position of token

    // size_t pos = id_map[token_idx].token_position;
    size_t pos = tokenInfos[token_idx].abs_depth_in_request;

    // float before_real = complex_input[i].x, before_complex =
    int pos_i = real_i % (proj_size / 2);
    float freq = pos * (1.0 / pow(10000.0, (float)2 * pos_i / proj_size));
    hipFloatComplex complex_pos = {cos(freq), sin(freq)};

    complex_input[i] = hipCmulf(complex_input[i], complex_pos);
    input_ptr[real_part_index] = complex_input[i].x;
    input_ptr[complex_part_index] = complex_input[i].y;
  }
}

template <typename DT>
__global__ void store_kv_cache(DT const *devQKVProjArray,
                               DT *kCache_ptr,
                               DT *vCache_ptr,
                               BatchConfig::PerTokenInfo const *tokenInfos,
                               int num_tokens,
                               int max_seq_len,
                               int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size) {
    int token_idx = i / hidden_size;
    int offset = i % hidden_size;
    size_t val_idx = token_idx * 3 * hidden_size + hidden_size + offset;
    DT kVal = devQKVProjArray[val_idx];
    DT vVal = devQKVProjArray[val_idx + hidden_size];

    int const req_id = tokenInfos[token_idx].request_index;
    int const tok_id = tokenInfos[token_idx].abs_depth_in_request;

    // key cache
    kCache_ptr[req_id * (hidden_size * max_seq_len) + tok_id * hidden_size +
               offset] = kVal;
    vCache_ptr[req_id * (hidden_size * max_seq_len) + tok_id * hidden_size +
               offset] = vVal;
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
                        hipStream_t stream) {

  checkCUDA(hipblasSetStream(m->handle.blas, stream));
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));
  DT alpha = 1.0f, beta = 0.0f;
  assert(m->qSize == m->vSize && m->qSize == m->kSize);
  hipblasDatatype_t hipblas_data_type = ff_to_cuda_datatype(m->output_type[0]);
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to HIPBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = HIPBLAS_COMPUTE_16F;
#else
  hipblasDatatype_t compute_type = hipblas_data_type;
#endif
  // Compute (W^T)x matmul: einsum(ijkl,im->jmkl)
  // Weights: qSize x qProjSize x 3 x num_q_heads
  // Input: qSize x num_tokens
  // Output >>> qProjSize x num_tokens x 3 x num_q_heads
  int m_q = m->qProjSize * m->num_q_heads;
  int m_k = m->kProjSize * m->num_q_heads;
  int m_v = m->vProjSize * m->num_q_heads;
  assert(m_q == m_k && m_k == m_v); // keep things simple for now
  int n = bc->num_active_tokens();
  int k = m->qSize;
  int m_ = m_q * QKV_WEIGHT_NUM;
  int lda = k, ldb = k, ldc = m_;
  checkCUDA(hipblasGemmEx(m->handle.blas,
                          HIPBLAS_OP_T,
                          HIPBLAS_OP_N,
                          m_,
                          n,
                          k,
                          &alpha,
                          weight_ptr,
                          hipblas_data_type,
                          lda,
                          input_ptr,
                          hipblas_data_type,
                          ldb,
                          &beta,
                          output_ptr,
                          hipblas_data_type,
                          ldc,
                          compute_type,
                          HIPBLAS_GEMM_DEFAULT));

  // apply rotary emmmbedding for q and k
  // step1 change the k, v to complex tensor
  int num_tokens = bc->num_active_tokens();
  int parallelism = m->kProjSize * num_tokens * m->num_q_heads;
  size_t q_array_size = m->qProjSize * num_tokens * m->num_q_heads;
  // apply bias for q, k, v
  if (*m->qkv_bias) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(apply_proj_bias_qkv<DT>),
                       GET_BLOCKS(parallelism),
                       min(CUDA_NUM_THREADS, parallelism),
                       0,
                       stream,
                       output_ptr,
                       bias_ptr,
                       shard_id,
                       num_tokens,
                       m->qProjSize,
                       m->kProjSize,
                       m->vProjSize,
                       m->global_num_q_heads,
                       m->num_q_heads,
                       *m->scaling_query,
                       m->scaling_factor,
                       m->hidden_size);
  } else if (m->scaling_query) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(scaling_query_kernel<DT>),
                       GET_BLOCKS(parallelism),
                       min(CUDA_NUM_THREADS, parallelism),
                       0,
                       stream,
                       output_ptr,
                       num_tokens,
                       m->num_q_heads,
                       m->qProjSize,
                       m->scaling_factor,
                       m->hidden_size);
  }
  if (*m->apply_rotary_embedding) {
    /*q&k*/
    parallelism = num_tokens * m->hidden_size;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(apply_rotary_embedding_hf<DT>),
                       GET_BLOCKS(parallelism),
                       min(CUDA_NUM_THREADS, parallelism),
                       0,
                       stream,
                       output_ptr,
                       m->complex_input,
                       m->token_infos,
                       m->qProjSize,
                       m->kProjSize,
                       num_tokens,
                       q_array_size,
                       m->hidden_size);
  }
}

template <typename DT>
void update_kv_cache_kernel(IncMultiHeadSelfAttentionMeta const *m,
                            BatchConfig const *bc,
                            hipStream_t stream) {
  int num_tokens = bc->num_active_tokens();
  if (num_tokens > 0) {
    int parallelism = m->hidden_size * num_tokens;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(store_kv_cache<DT>),
                       GET_BLOCKS(parallelism),
                       min(CUDA_NUM_THREADS, parallelism),
                       0,
                       stream,
                       static_cast<DT *>(m->devQKVProjArray),
                       static_cast<DT *>(m->keyCache),
                       static_cast<DT *>(m->valueCache),
                       m->token_infos,
                       num_tokens,
                       BatchConfig::max_sequence_length(),
                       m->hidden_size);
  }
}

template <typename DT>
void pre_build_weight_kernel(IncMultiHeadSelfAttentionMeta const *m,
                             GenericTensorAccessorR const weight,
                             DataType data_type,
                             hipStream_t stream) {
  // additional processing for weight uploading
  // Note that we update weight_ptr and bias_ptr when uploading weight and
  // bias
  if (m->quantization_type != DT_NONE) {
    // copy weight_ptr to quantized_weight_ptr, do compression and store in
    // m->weight_ptr
    checkCUDA(hipMemcpyAsync(m->quantized_weight_ptr,
                             weight.get_byte_ptr(),
                             m->quantized_weightSize,
                             hipMemcpyHostToDevice,
                             stream));

    if (m->quantization_type == DT_INT4) {
      int parallelism = m->qProjSize * m->qSize * m->num_q_heads / 2;
      decompress_int4_attention_weights<<<GET_BLOCKS(parallelism),
                                          min(CUDA_NUM_THREADS, parallelism),
                                          0,
                                          stream>>>(
          m->quantized_weight_ptr,
          static_cast<DT *>(m->weight_ptr),
          m->qProjSize,
          m->qSize,
          m->num_q_heads);
    } else {
      assert(m->quantization_type == DT_INT8);
      int parallelism = m->qProjSize * m->qSize * m->num_q_heads;
      decompress_int8_attention_weights<<<GET_BLOCKS(parallelism),
                                          min(CUDA_NUM_THREADS, parallelism),
                                          0,
                                          stream>>>(
          m->quantized_weight_ptr,
          static_cast<DT *>(m->weight_ptr),
          m->qProjSize,
          m->qSize,
          m->num_q_heads);
    }
  } else {
    if (data_type == DT_FLOAT) {
      checkCUDA(hipMemcpyAsync(m->weight_ptr,
                               weight.get_float_ptr(),
                               m->weightSize,
                               hipMemcpyHostToDevice,
                               stream));
    } else if (data_type == DT_HALF) {
      checkCUDA(hipMemcpyAsync(m->weight_ptr,
                               weight.get_half_ptr(),
                               m->weightSize,
                               hipMemcpyHostToDevice,
                               stream));
    } else {
      assert(false);
    }
  }
}

template <typename DT>
void inference_kernel(IncMultiHeadSelfAttentionMeta const *m,
                      BatchConfig const *bc,
                      int shard_id,
                      DT const *input_ptr,
                      DT const *weight_ptr,
                      DT *output_ptr,
                      DT const *bias_ptr,
                      hipStream_t stream) {
  // here because we need postion info in infernece 1

  if (m->offload && m->biasSize > 0) {
    checkCUDA(hipMemcpyAsync(
        m->bias_ptr, bias_ptr, m->biasSize, hipMemcpyHostToDevice, stream));
    bias_ptr = static_cast<DT *>(m->bias_ptr);
  }
  checkCUDA(hipMemcpyAsync(m->token_infos,
                           &(bc->tokensInfo),
                           bc->num_active_tokens() *
                               sizeof(BatchConfig::PerTokenInfo),
                           hipMemcpyHostToDevice,
                           stream));
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
  compute_attention_kernel(
      m, bc, shard_id, output_ptr, bias_ptr, weight_ptr, stream);
}

} // namespace IncMultiHeadAttention
} // namespace Kernels

using namespace Kernels::IncMultiHeadAttention;

template <typename DT>
__global__ void fill_entries_above_diagonal(DT *matrix,
                                            size_t num_rows,
                                            size_t num_cols,
                                            size_t num_q_heads,
                                            size_t entries_above_diagonal,
                                            DT value) {
  CUDA_KERNEL_LOOP(i, entries_above_diagonal * num_q_heads) {
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
                              DT const *weight_ptr,
                              hipStream_t stream) {
  checkCUDA(hipblasSetStream(m->handle.blas, stream));
  checkCUDNN(miopenSetStream(m->handle.dnn, stream));
  hipblasDatatype_t hipblas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  miopenDataType_t miopen_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  assert(data_type_size(m->output_type[0]) == sizeof(DT));
#if CUDA_VERSION >= 11000
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
  hipblasDatatype_t compute_type = hipblas_data_type;
#endif
  // int num_requests = bc->num_active_requests();
  int num_tokens = bc->num_active_tokens();
  int tokens_previous_requests = 0;
  int q_block_size = m->qProjSize;
  int kt_block_size = m->kProjSize;
  int kt_req_block_size =
      kt_block_size * m->num_q_heads * BatchConfig::max_sequence_length();
  int vt_block_size = m->vProjSize;
  int vt_req_block_size =
      vt_block_size * m->num_q_heads * BatchConfig::max_sequence_length();
  assert(m->qProjSize == m->kProjSize);

  for (int i = 0; i < bc->max_requests_per_batch(); i++) {
    if (bc->request_completed[i]) {
      continue;
    }
    int num_new_tokens = bc->requestsInfo[i].num_tokens_in_batch;
    int total_tokens = bc->requestsInfo[i].first_token_depth_in_request +
                       bc->requestsInfo[i].num_tokens_in_batch;
    // bc->token_last_available_idx[i] + 1;
    // Compute (QK^T/sqrt(d_k))
    // a flag of using this scaling alpha
    int m_ = num_new_tokens;
    int n = total_tokens;
    int k = m->qProjSize;
    int lda = k * m->num_q_heads * QKV_WEIGHT_NUM, ldb = k * m->num_q_heads,
        ldc = m_;
    int strideA = q_block_size;
    int strideB = kt_block_size;
    int strideC = num_new_tokens * total_tokens;
    DT alpha = 1.0f, beta = 0.0f;
    if (*m->qk_prod_scaling) {
      alpha = static_cast<DT>(1.0f / sqrt(m->kProjSize));
    }
    // To get A, skip over Q entries from previous requests (same head)
    DT const *A = static_cast<DT *>(m->devQKVProjArray) +
                  tokens_previous_requests * m->qProjSize * m->num_q_heads *
                      QKV_WEIGHT_NUM;
    // To get B, skip over K entries from previous requests (all heads +
    // padding)
    DT const *B = static_cast<DT *>(m->keyCache) + i * kt_req_block_size;
    // To get C, skip over QK^T products from previous requests
    DT *C = static_cast<DT *>(m->qk_prods);
    checkCUDA(hipblasGemmStridedBatchedEx(m->handle.blas,
                                          HIPBLAS_OP_T,
                                          HIPBLAS_OP_N,
                                          m_,
                                          n,
                                          k,
                                          &alpha,
                                          A,
                                          hipblas_data_type,
                                          lda,
                                          strideA,
                                          B,
                                          hipblas_data_type,
                                          ldb,
                                          strideB,
                                          &beta,
                                          C,
                                          hipblas_data_type,
                                          ldc,
                                          strideC,
                                          m->num_q_heads,
                                          compute_type,
                                          HIPBLAS_GEMM_DEFAULT));

    // add alibi position bias to qk production
    if (*m->position_bias) {
      size_t parallelism = m->num_q_heads * total_tokens * num_new_tokens;
      hipLaunchKernelGGL(HIP_KERNEL_NAME(apply_position_bias_qkprd<DT>),
                         GET_BLOCKS(parallelism),
                         min((size_t)CUDA_NUM_THREADS, parallelism),
                         0,
                         stream,
                         C,
                         num_new_tokens,
                         total_tokens,
                         m->num_q_heads,
                         m->global_num_q_heads,
                         shard_id);
    }
    // Fill all elements above diagonal in qk prods with -inf to force
    // causal attention.
    assert(num_new_tokens <= total_tokens);
    size_t entries_above_diagonal = num_new_tokens * (num_new_tokens - 1) / 2;
    if (entries_above_diagonal > 0) {
      size_t parallelism = m->num_q_heads * entries_above_diagonal;
      hipLaunchKernelGGL(HIP_KERNEL_NAME(fill_entries_above_diagonal<DT>),
                         GET_BLOCKS(parallelism),
                         min((size_t)CUDA_NUM_THREADS, parallelism),
                         0,
                         stream,
                         C,
                         num_new_tokens,
                         total_tokens,
                         m->num_q_heads,
                         entries_above_diagonal,
                         static_cast<DT>(-INFINITY));
    }
    // Compute Softmax(QK^T/sqrt(d_k))
    // Before modifying the parameters below, make sure to read the following
    // description of the CUDNN_TENSOR_NCHW tensor layout, from
    // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnTensorFormat_t:
    // This tensor format specifies that the data is laid out in the following
    // order: batch size, feature maps, rows, columns. The strides are
    // implicitly defined in such a way that the data are contiguous in memory
    // with no padding between images, feature maps, rows, and columns; the
    // columns are the inner dimension and the images are the outermost
    // dimension.
    int n_param = m->num_q_heads;
    int c_param = total_tokens;
    int h_param = 1;
    int w_param = num_new_tokens;
    checkCUDNN(miopenSet4dTensorDescriptor(
        m->qk_tensor, miopen_data_type, n_param, c_param, h_param, w_param));
    float softmax_alpha = 1.0f, softmax_beta = 0.0f;
    DT *C_softmax = static_cast<DT *>(m->qk_prods_softmax);
    // The softmax operation below is executed according to the
    // CUDNN_SOFTMAX_MODE_CHANNEL, which is also described in the docs: The
    // softmax operation is computed per spatial location (H,W) per image (N)
    // across dimension C.
    checkCUDNN(miopenSoftmaxForward_V2(m->handle.dnn,
                                       &softmax_alpha,
                                       m->qk_tensor,
                                       C,
                                       &softmax_beta,
                                       m->qk_tensor,
                                       C_softmax,
                                       MIOPEN_SOFTMAX_ACCURATE,
                                       MIOPEN_SOFTMAX_MODE_CHANNEL));
    // Matmul softmax(QK^T/sqrt(d_k)) by V
    alpha = 1.0f, beta = 0.0f;
    m_ = num_new_tokens;
    n = m->vProjSize;
    k = total_tokens;
    lda = m_, ldb = n * m->num_q_heads, ldc = m_;
    strideA = num_new_tokens * total_tokens;
    strideB = vt_block_size;
    strideC = num_new_tokens * m->vProjSize;
    // To get A, skip over softmax(QK^T/sqrt(d_k)) entries from previous
    // requests (all heads)
    A = C_softmax;
    // To get B, skip over V^T entries from previous requests (all heads +
    // padding)
    B = static_cast<DT *>(m->valueCache) + i * vt_req_block_size;
    // To get C, skip over softmax(QK^T/sqrt(d_k))V products from previous
    // requests
    C = static_cast<DT *>(m->attn_heads) +
        tokens_previous_requests * m->num_q_heads * m->vProjSize;

    checkCUDA(hipblasGemmStridedBatchedEx(m->handle.blas,
                                          HIPBLAS_OP_N,
                                          HIPBLAS_OP_T,
                                          m_,
                                          n,
                                          k,
                                          &alpha,
                                          A,
                                          hipblas_data_type,
                                          lda,
                                          strideA,
                                          B,
                                          hipblas_data_type,
                                          ldb,
                                          strideB,
                                          &beta,
                                          C,
                                          hipblas_data_type,
                                          ldc,
                                          strideC,
                                          m->num_q_heads,
                                          compute_type,
                                          HIPBLAS_GEMM_DEFAULT));
    // Project to output, save result directly on output tensor
    alpha = 1.0f, beta = 0.0f;
    m_ = m->oProjSize;
    k = m->vProjSize * m->num_q_heads;
    n = num_new_tokens;
    lda = k, ldb = n, ldc = m_;
    A = weight_ptr + m->qSize * (m->qProjSize * m->num_q_heads +
                                 m->kProjSize * m->num_q_heads +
                                 m->vProjSize * m->num_q_heads);
    B = C;
    C = static_cast<DT *>(output_ptr) + tokens_previous_requests * m->oProjSize;

    checkCUDA(hipblasGemmEx(m->handle.blas,
                            HIPBLAS_OP_T,
                            HIPBLAS_OP_T,
                            m_,
                            n,
                            k,
                            &alpha,
                            A,
                            hipblas_data_type,
                            lda,
                            B,
                            hipblas_data_type,
                            ldb,
                            &beta,
                            C,
                            hipblas_data_type,
                            ldc,
                            compute_type,
                            HIPBLAS_GEMM_DEFAULT));
    tokens_previous_requests += num_new_tokens;
  }

  if (*m->final_bias && shard_id == 0) {
    int parallelism = m->oProjSize * num_tokens;
    int qkv_weight_size = m->qProjSize * m->global_num_q_heads +
                          m->kProjSize * m->global_num_q_heads +
                          m->vProjSize * m->global_num_q_heads;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(apply_proj_bias_w<DT>),
                       GET_BLOCKS(parallelism),
                       min(CUDA_NUM_THREADS, parallelism),
                       0,
                       stream,
                       output_ptr,
                       bias_ptr,
                       num_tokens,
                       qkv_weight_size,
                       m->oProjSize);
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
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  bool use_bias = *m->qkv_bias || *m->final_bias;

  hipEvent_t t_start, t_end;
  if (m->profiling) {
    checkCUDA(hipEventCreate(&t_start));
    checkCUDA(hipEventCreate(&t_end));
    checkCUDA(hipEventRecord(t_start, stream));
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
    checkCUDA(hipEventRecord(t_end, stream));
    checkCUDA(hipEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    checkCUDA(hipEventDestroy(t_start));
    checkCUDA(hipEventDestroy(t_end));
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
    int _num_q_heads,
    int _num_kv_heads)
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
                                    attn->qkv_bias,
                                    attn->scaling_query,
                                    attn->qk_prod_scaling,
                                    attn->position_bias,
                                    attn->final_bias,
                                    attn->scaling_factor,
                                    weight,
                                    gpu_mem_allocator,
                                    num_samples,
                                    attn->num_q_heads,
                                    attn->num_kv_heads,
                                    _num_q_heads,
                                    _num_kv_heads,
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
    bool _qkv_bias,
    bool _scaling_query,
    bool _qk_prod_scaling,
    bool _position_bias,
    bool _final_bias,
    float _scaling_factor,
    GenericTensorAccessorR const &weight,
    MemoryAllocator &gpu_mem_allocator,
    int num_samples,
    int _global_num_q_heads,
    int _global_num_kv_heads,
    int _num_q_heads,
    int _num_kv_heads,
    DataType _quantization_type,
    bool _offload)
    : OpMeta(handler, attn), weight_ptr(nullptr), bias_ptr(nullptr) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(miopenSetStream(handler.dnn, stream));
  checkCUDNN(miopenCreateTensorDescriptor(&qk_tensor));
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

  global_num_q_heads = _global_num_q_heads;
  global_num_kv_heads = _global_num_kv_heads;
  num_q_heads = _num_q_heads;
  num_kv_heads = _num_kv_heads;
  hidden_size = num_q_heads * qProjSize;

  weightSize =
      ((qSize * qProjSize + oProjSize * (vProjSize > 0 ? vProjSize : vSize)) *
           num_q_heads +
       (kSize * kProjSize + vSize * vProjSize) * num_q_heads) *
      size_of_dt;
  if (quantization_type != DT_NONE) {
    quantized_weightSize = get_quantization_to_byte_size(
        attn->data_type, quantization_type, weightSize);
  }
  // biasSize = _bias ? oProjSize * size_of_dt * 4 : 0;

  int qkv_bias_size =
      qProjSize * num_q_heads + (kProjSize + vProjSize) * num_q_heads;
  int final_bias_size = oProjSize;
  biasSize =
      (_qkv_bias ? qkv_bias_size : 0) + (final_bias ? final_bias_size : 0);

  // has_load_weights = (bool *)calloc(1, sizeof(bool));
  //*has_load_weights = false;
  apply_rotary_embedding = (bool *)calloc(1, sizeof(bool));
  *apply_rotary_embedding = _apply_rotary_embedding;
  qkv_bias = (bool *)calloc(1, sizeof(bool));
  *qkv_bias = _qkv_bias;
  scaling_query = (bool *)calloc(1, sizeof(bool));
  *scaling_query = _scaling_query;
  scaling_factor = _scaling_factor;
  qk_prod_scaling = (bool *)calloc(1, sizeof(bool));
  *qk_prod_scaling = _qk_prod_scaling;
  position_bias = (bool *)calloc(1, sizeof(bool));
  *position_bias = _position_bias;
  final_bias = (bool *)calloc(1, sizeof(bool));
  *final_bias = _final_bias;

  // allocate weight and bias in the reserve space for cpu offloading
  if (offload) {
    weight_ptr = gpu_mem_allocator.allocate_reserved_untyped(weightSize);
    bias_ptr = gpu_mem_allocator.allocate_reserved_untyped(biasSize);
  }

  // allocate memory for the seqArray and reserve space
  {
    int max_tokens_per_batch = BatchConfig::max_tokens_per_batch();
    size_t qkv_max_proj_size = max_tokens_per_batch * (qProjSize * num_q_heads +
                                                       kProjSize * num_q_heads +
                                                       vProjSize * num_q_heads);
    size_t key_cache_size = 0, value_cache_size = 0;
    switch (infer_mode) {
      case INC_DECODING_MODE:
      case TREE_VERIFY_MODE: {
        key_cache_size = num_q_heads * kProjSize *
                         BatchConfig::max_requests_per_batch() *
                         BatchConfig::max_sequence_length();
        value_cache_size = num_q_heads * vProjSize *
                           BatchConfig::max_requests_per_batch() *
                           BatchConfig::max_sequence_length();
        break;
      }
      case BEAM_SEARCH_MODE: {
        key_cache_size = num_q_heads * kProjSize *
                         BeamSearchBatchConfig::max_requests_per_batch() *
                         BatchConfig::max_sequence_length() *
                         BeamSearchBatchConfig::MAX_BEAM_WIDTH;
        value_cache_size = num_q_heads * vProjSize *
                           BeamSearchBatchConfig::max_requests_per_batch() *
                           BatchConfig::max_sequence_length() *
                           BeamSearchBatchConfig::MAX_BEAM_WIDTH;
        break;
      }
      default:
        assert(false && "Unkown inference mode");
    }
    size_t tokeninfo_size = max_tokens_per_batch;
    size_t qk_prod_size =
        max_tokens_per_batch * BatchConfig::max_sequence_length() * num_q_heads;
    size_t attn_heads_size = max_tokens_per_batch * num_q_heads * vProjSize;
    size_t complex_size = (max_tokens_per_batch * (qProjSize * num_q_heads +
                                                   kProjSize * num_q_heads)) /
                          2;
    size_t totalSize =
        (qkv_max_proj_size + key_cache_size + value_cache_size +
         2 * qk_prod_size + attn_heads_size) *
            size_of_dt +
        tokeninfo_size * sizeof(BatchConfig::PerTokenInfo) +
        complex_size * sizeof(hipFloatComplex); // more components will
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
      complex_input =
          gpu_mem_allocator.allocate_reserved<hipFloatComplex>(complex_size);
      // offset += complex_size * sizeof(hipFloatComplex);
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
      complex_input =
          gpu_mem_allocator.allocate_instance<hipFloatComplex>(complex_size);
    }

    // allocate more size for quantization data
    if (quantization_type != DT_NONE) {
      assert(offload);
      quantized_weight_ptr =
          gpu_mem_allocator.allocate_reserved<char>(quantized_weightSize);
    }
    if (!offload) {
      assert(gpu_mem_allocator.reserved_total_size ==
             gpu_mem_allocator.reserved_allocated_size);
    }
  }
  checkCUDA(hipStreamSynchronize(stream));
}

IncMultiHeadSelfAttentionMeta::~IncMultiHeadSelfAttentionMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}

template void Kernels::IncMultiHeadAttention::pre_build_weight_kernel<float>(
    IncMultiHeadSelfAttentionMeta const *m,
    GenericTensorAccessorR const weight,
    DataType data_type,
    hipStream_t stream);

template void Kernels::IncMultiHeadAttention::pre_build_weight_kernel<half>(
    IncMultiHeadSelfAttentionMeta const *m,
    GenericTensorAccessorR const weight,
    DataType data_type,
    hipStream_t stream);

}; // namespace FlexFlow

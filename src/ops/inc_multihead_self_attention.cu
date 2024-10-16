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
#include "cuComplex.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/inc_multihead_self_attention.h"
#include "flexflow/ops/kernels/decompress_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_utils.cuh"
#include "flexflow/utils/cuda_helper.h"
#include <math_constants.h>

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

#define WARP_SIZE 32

namespace Kernels {
namespace IncMultiHeadAttention {

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

    size_t val_idx =
        token_idx * QKV_WEIGHT_NUM * hidden_size + hidden_size + offset;

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
__global__ void store_query_cache(DT const *devQKVProjArray,
                                  DT *qCache_ptr,
                                  int num_tokens,
                                  int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size) {
    int token_idx = i / hidden_size;
    int offset = i % hidden_size;

    size_t val_idx = token_idx * QKV_WEIGHT_NUM * hidden_size + offset;

    DT qVal = devQKVProjArray[val_idx];

    // query cache
    qCache_ptr[i] = qVal;
  }
}

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
void compute_attention_kernel_prompt(IncMultiHeadSelfAttentionMeta *m,
                                     BatchConfig const *bc,
                                     int shard_id,
                                     cudaStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  assert(data_type_size(m->output_type[0]) == sizeof(DT));
  cudaDataType_t compute_type = cublas_data_type;

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
    if (bc->request_completed[i] ||
        (!bc->requestsInfo[i].prompt_phase && !bc->requestsInfo[i].peft_bwd)) {
      continue;
    }
    int num_new_tokens = bc->requestsInfo[i].num_tokens_in_batch;
    int total_tokens = bc->requestsInfo[i].first_token_depth_in_request +
                       bc->requestsInfo[i].num_tokens_in_batch;
    int max_peft_tokens = bc->requestsInfo[i].max_length;
    // Copy query to m->query_activation_buffer if we need to compute
    // PEFT backward
    if (bc->requestsInfo[i].peft_bwd) {
      size_t activation_size_needed =
          sizeof(DT) * max_peft_tokens * m->num_q_heads * m->qProjSize;
      if (activation_size_needed > m->allocated_peft_buffer_size1) {
        MemoryAllocator *allocator = m->handle.peft_activation_allocator;
        m->query_activation_buffer =
            allocator->allocate_instance_untyped(activation_size_needed);
        m->allocated_peft_buffer_size1 = activation_size_needed;
      }
      int parallelism = m->hidden_size * num_tokens;
      store_query_cache<<<GET_BLOCKS(parallelism),
                          min(CUDA_NUM_THREADS, parallelism),
                          0,
                          stream>>>(
          static_cast<DT *>(m->devQKVProjArray),
          static_cast<DT *>(m->query_activation_buffer),
          num_tokens,
          m->hidden_size);
    }
    // Step 1: compute query-key product QK.T/sqrt(d_k)
    {
      // Scale by sqrt(d_k) as per the original attention paper
      DT alpha = 1.0f, beta = 0.0f;
      if (*m->qk_prod_scaling) {
        alpha = static_cast<DT>(1.0f / sqrt(m->kProjSize));
      }
      // after transpositions
      int m_ = num_new_tokens;
      int n = total_tokens;
      int k = m->qProjSize;
      // before transpositions
      int lda = k * m->num_q_heads * QKV_WEIGHT_NUM, ldb = k * m->num_q_heads,
          ldc = m_;
      // N.B. strides are applied before transpose operations
      int strideA = q_block_size;
      int strideB = kt_block_size;
      int strideC = num_new_tokens * total_tokens;

      // matrix A: devQKVProjArray
      // matrix A's layout: [qProjSize, num_heads, 3, num_new_tokens]
      // To get query projection, skip over Q entries from previous requests
      DT const *A = static_cast<DT *>(m->devQKVProjArray) +
                    bc->requestsInfo[i].first_token_offset_in_batch *
                        m->qProjSize * m->num_q_heads * QKV_WEIGHT_NUM;
      // matrix B: key cache
      // matrix B's layout: [kProjSize * num_heads, total_tokens]
      // To get B, skip over K entries from previous requests (all heads +
      // padding)
      DT const *B = static_cast<DT *>(m->keyCache) + i * kt_req_block_size;
      // matrix C: qk_prods
      // matrix C's layout: [num_new_tokens, total_tokens, num_heads]
      // To get C, skip over QK.T products from previous requests
      DT *C = static_cast<DT *>(m->qk_prods);
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
                                           m->num_q_heads,
                                           compute_type,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    // Step 2: Add alibi position bias to qk production
    // matrix C: qk_prods
    // matrix C's layout: [num_new_tokens, total_tokens, num_heads]
    // To get C, skip over QK.T products from previous requests
    DT *C = static_cast<DT *>(m->qk_prods);
    if (*m->position_bias) {
      size_t parallelism = m->num_q_heads * total_tokens * num_new_tokens;
      apply_position_bias_qkprd<<<GET_BLOCKS(parallelism),
                                  min((size_t)CUDA_NUM_THREADS, parallelism),
                                  0,
                                  stream>>>(C,
                                            num_new_tokens,
                                            total_tokens,
                                            m->num_q_heads,
                                            m->global_num_q_heads,
                                            shard_id);
    }

    // Step 3: Apply causal mask. Fill all elements above diagonal in qk prods
    // with -inf to force causal attention.
    assert(num_new_tokens <= total_tokens);
    size_t entries_above_diagonal = num_new_tokens * (num_new_tokens - 1) / 2;
    if (entries_above_diagonal > 0) {
      size_t parallelism = m->num_q_heads * entries_above_diagonal;
      fill_entries_above_diagonal<<<GET_BLOCKS(parallelism),
                                    min((size_t)CUDA_NUM_THREADS, parallelism),
                                    0,
                                    stream>>>(C,
                                              num_new_tokens,
                                              total_tokens,
                                              m->num_q_heads,
                                              entries_above_diagonal,
                                              static_cast<DT>(-INFINITY));
    }

    // Step 4: Compute Softmax(QK.T/sqrt(d_k))
    {
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
      checkCUDNN(cudnnSetTensor4dDescriptor(m->qk_tensor,
                                            CUDNN_TENSOR_NCHW,
                                            cudnn_data_type,
                                            n_param,
                                            c_param,
                                            h_param,
                                            w_param));
      float softmax_alpha = 1.0f, softmax_beta = 0.0f;
      DT *C_softmax = static_cast<DT *>(m->qk_prods_softmax);
      // The softmax operation below is executed according to the
      // CUDNN_SOFTMAX_MODE_CHANNEL, which is also described in the docs: The
      // softmax operation is computed per spatial location (H,W) per image (N)
      // across dimension C.
      checkCUDNN(cudnnSoftmaxForward(m->handle.dnn,
                                     CUDNN_SOFTMAX_ACCURATE,
                                     CUDNN_SOFTMAX_MODE_CHANNEL,
                                     &softmax_alpha,
                                     m->qk_tensor,
                                     C,
                                     &softmax_beta,
                                     m->qk_tensor,
                                     C_softmax));
    }
    // Copy C_softmax to m->softmax_activation_buffer if we need to compute
    // PEFT backward
    if (bc->requestsInfo[i].peft_bwd) {
      DT *C_softmax = static_cast<DT *>(m->qk_prods_softmax);
      size_t activation_size_needed =
          sizeof(DT) * max_peft_tokens * max_peft_tokens * m->num_q_heads;
      if (activation_size_needed > m->allocated_peft_buffer_size2) {
        MemoryAllocator *allocator = m->handle.peft_activation_allocator;
        m->softmax_activation_buffer =
            allocator->allocate_instance_untyped(activation_size_needed);
        m->allocated_peft_buffer_size2 = activation_size_needed;
      }
      checkCUDA(cudaMemcpyAsync(m->softmax_activation_buffer,
                                C_softmax,
                                sizeof(DT) * total_tokens * num_new_tokens *
                                    m->num_q_heads,
                                cudaMemcpyDeviceToDevice,
                                stream));
    }
    // Step 5: Matmul softmax(QK.T/sqrt(d_k)) by V. Implemented as V @
    // softmax(QK.T/sqrt(d_k)).T
    {
      DT alpha = 1.0f, beta = 0.0f;
      // after transpositions
      int m_ = m->vProjSize;
      int n = num_new_tokens;
      int k = total_tokens;
      // before transpositions
      int lda = m_ * m->num_q_heads, ldb = n, ldc = m_ * m->num_q_heads;
      // N.B. strides are applied before transpose operations
      int strideA = vt_block_size;
      int strideB = num_new_tokens * total_tokens;
      int strideC = m->vProjSize;
      // matrix A: value cache
      // matrix A's layout: [vProjSize, num_heads, total_tokens]
      // To get A, skip over V.T entries from previous requests (all heads +
      // padding)
      DT *A = static_cast<DT *>(m->valueCache) + i * vt_req_block_size;
      // matrix B: qk_prods_softmax
      // matrix B's layout: [num_new_tokens, total_tokens, num_heads]
      // To get B, skip over softmax(QK.T/sqrt(d_k)) entries from previous
      // requests (all heads)
      DT *B = static_cast<DT *>(m->qk_prods_softmax);
      // matrix C: attn heads
      // matrix C's layout: [vProjSize, num_heads, num_new_tokens]
      // To get C, skip over softmax(QK.T/sqrt(d_k))V products from previous
      // requests
      // store the result attn heads, also skip the genration tokens
      DT *C = static_cast<DT *>(m->attn_heads) +
              (bc->requestsInfo[i].first_token_offset_in_batch) *
                  m->num_q_heads * m->vProjSize;
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
                                           m->num_q_heads,
                                           compute_type,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    tokens_previous_requests += num_new_tokens;
  }
  if (tokens_previous_requests != (num_tokens - bc->num_generation_tokens)) {
    bc->print();
    printf("tokens_previous_requests: %i\n", tokens_previous_requests);
    printf("num_tokens: %i\n", num_tokens);
    printf("bc->num_generation_tokens: %i\n", bc->num_generation_tokens);
  }
  assert(tokens_previous_requests == (num_tokens - bc->num_generation_tokens));
}

// gridDim = num_heads
// blockDim = num_tokens/num_request * head_size
// QKV tensor layout: |QKV| * num_new_tokens. |Q=K=V=head_size * num_heads|
// one thread process one head_size
template <typename DT,
          int THREADS_PER_BLOCK,
          int Dh,
          int Dh_MAX,
          int THREADS_PER_KEY,
          int THREADS_PER_VALUE>
__global__ void compute_attention_kernel_generation_kernel(
    DT const *query,
    DT const *key_cache,
    DT const *value_cache,
    DT *output_ptr,
    float const scale,
    int max_seq_length,
    int per_head_size,
    int hidden_size,
    BatchConfig::PerRequestInfo *request_infos) {

  // q, k
  using Q_vec = typename VEC_K<DT, THREADS_PER_KEY>::Type;
  using K_vec = typename VEC_K<DT, THREADS_PER_KEY>::Type;
  using V_vec = typename VEC_V<DT>::Type;
  using Out_sum = typename Vec_fp32_<V_vec>::Type;

  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

  // eg.  if head_size = 128, thread_per_key = 4, with float32 precision
  // then K_VEC_SIZE = 1,  QK_VEC_SIZE = 4
  //  K_ELTS_PER_THREAD = 128 / 4 = 32
  //  K_VECS_PER_THREAD = 32 / 1 = 32
  constexpr int K_VEC_SIZE = sizeof(K_vec) / sizeof(DT);
  // constexpr int QK_VEC_SIZE = 16 / sizeof(DT);
  // // constexpr int QK_VEC_SIZE = sizeof(Qk_vec_k) / sizeof(DT);
  constexpr int K_ELTS_PER_THREAD = Dh / THREADS_PER_KEY;
  constexpr int K_VECS_PER_THREAD = K_ELTS_PER_THREAD / K_VEC_SIZE;
  // constexpr int QK_ELTS_IN_16B = 16 / sizeof(DT);

  // thread id
  int const tidx = threadIdx.x;
  // head id
  int const head_idx = blockIdx.x;
  // request idx
  int const request_idx = blockIdx.y;

  int const batch_config_request_id =
      request_infos[request_idx].batch_config_request_id;

  int const first_step = 0;

  int const tlength =
      request_infos[batch_config_request_id].first_token_depth_in_request +
      request_infos[batch_config_request_id].num_tokens_in_batch;

  // shared memory objects
  extern __shared__ char smem_[];

  float *qk_smem = reinterpret_cast<float *>(smem_);
  float *out_smem = reinterpret_cast<float *>(smem_);

  float qk_max = -FLT_MAX;

  // first WARPS_PER_BLOCK for store qk_max, second WARPS_PER_BLOCK for sum
  __shared__ float red_smem[WARPS_PER_BLOCK * 2];

  const DT *q_ptr = query + request_idx * hidden_size * QKV_WEIGHT_NUM +
                    head_idx * per_head_size;
  __shared__ Q_vec q_vecs[THREADS_PER_KEY][K_VECS_PER_THREAD];
  // DT const *q_ptr =
  //     query + request_idx * Dh * QKV_WEIGHT_NUM + head_idx * per_head_size;

  // q tensor in this thread
  // if THREADS_PER_KEY is 4, first thread load 0, 4, 8, 12..., total
  // K_VECS_PER_THREAD elements
  // QK_vec_k: 32->1, 64->2, 128->4... head_size
  // K_vec_k: 4->1, 2->2, 1->4 threads_per_key

  // the start offset of the element eg. (0, 1, 2, 3) * K_VEC_SIZE
  int ki = tidx % THREADS_PER_KEY * K_VEC_SIZE;
  int ki_o = tidx % THREADS_PER_KEY;
  // the first key's offset for this thread
  // ko = 0, 0, 0, 0, 1, 1, 1, 1, ....
  int ko = tidx / THREADS_PER_KEY;
  // load q tensor
  Q_vec q_vec[K_VECS_PER_THREAD];
#pragma unroll
  for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
    q_vecs[ki_o][ii] = *reinterpret_cast<Q_vec const *>(
        q_ptr + ki + ii * THREADS_PER_KEY * K_VEC_SIZE);
  }
  __syncthreads();
  // first iter = 128 / 4 = 32
  // K_VECS_PER_THREAD = 32
  //  K_PER_ITER how many keys in this loop
  //  The number of timesteps loaded per iteration.
  constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;
  //   // The number of keys per warp.
  constexpr int K_PER_WARP = WARP_SIZE / THREADS_PER_KEY;

  DT const *k_cache_batch =
      key_cache + batch_config_request_id * max_seq_length * hidden_size + ki;

  int ti_end =
      div_up(tlength - first_step, K_PER_WARP) * K_PER_WARP + first_step;
  // get k, perform qk proj

  for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
    K_vec k[K_VECS_PER_THREAD];
    int const ti_circ = ti % max_seq_length;
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
      int jj = ii * THREADS_PER_KEY * K_VEC_SIZE;
      if (ti < tlength) {
        k[ii] = *reinterpret_cast<K_vec const *>(k_cache_batch +
                                                 ti_circ * hidden_size +
                                                 head_idx * per_head_size + jj);
      }
      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
    }
    float qk = scale * Qk_dot<DT, THREADS_PER_KEY>::dot(q_vecs[ki_o], k);
    // // todo add positional embedding to the qk production
    // // Store the product to shared memory. There's one qk value per
    // timestep.
    // // Update the max.
    if (ti < tlength && tidx % THREADS_PER_KEY == 0) {
      // todo add alobi here
      bool const mask = ti_circ >= tlength;
      if (mask) {
        assert(false);
      }
      qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      qk_smem[ti - first_step] = mask ? 0.f : qk;
    }
  }

  __syncthreads();

#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }

  // Decompose the thread index into warp and lane.
  int const warp = tidx / WARP_SIZE;
  int const lane = tidx % WARP_SIZE;

  // The warp leader writes the max to shared memory.
  if (lane == 0) {
    red_smem[warp] = qk_max;
  }

  // Make sure the products are in shared memory.
  __syncthreads();

  // The warps finalize the reduction.
  qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }

  // Broadcast to all the threads in the warp.
  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

  float exp_sum = 0.f;
  for (int ti = first_step + tidx; ti < tlength; ti += THREADS_PER_BLOCK) {
    float logit = __expf(qk_smem[ti - first_step] - qk_max);
    exp_sum += logit;
    qk_smem[ti - first_step] = logit;
  }

  // Compute the sum.
  exp_sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], exp_sum);

  // softmax
  float inv_sum = __fdividef(1.f, exp_sum + 1.e-6);
  for (int ti = first_step + tidx; ti < tlength; ti += THREADS_PER_BLOCK) {
    qk_smem[ti - first_step] *= inv_sum;
  }

  __syncthreads();
  // if (blockIdx.y == 0 && blockIdx.x == 0 && tidx == 0) {
  //   printf("softmax %.10f\n", qk_smem[0]);
  // }

  // value projection
  constexpr int V_VEC_SIZE = 16 / sizeof(DT);
  // A vector of V elements for the current timestep.
  // using V_vec_k = typename V_vec_k_<DT, V_VEC_SIZE>::Type;
  // using V_vec_acum = typename V_vec_acum_fp32_<V_vec_k>::Type;

  // The value computed by this thread.
  int vo = tidx / THREADS_PER_VALUE;
  // The hidden dimensions computed by this particular thread.
  int vi = tidx % THREADS_PER_VALUE * V_VEC_SIZE;
  constexpr int V_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;

  Out_sum out;
  zero(out);

  // The base pointer for the value in the cache buffer.
  DT const *v_cache_batch =
      value_cache + batch_config_request_id * max_seq_length * hidden_size + vi;

  if (Dh == Dh_MAX || vi < Dh) {
    for (int ti = first_step + vo; ti < tlength; ti += V_PER_ITER) {
      // Load the values from the cache.
      int const ti_circ = ti % max_seq_length;

      V_vec v = *reinterpret_cast<V_vec const *>(
          v_cache_batch + ti_circ * hidden_size + head_idx * per_head_size);
      float logit = qk_smem[ti - first_step];
      out = FlexFlow::fma(logit, cast_to_float(v), out);
    }
  }

  //   // Make sure we can start writing to shared memory.
  __syncthreads();

  // Run the final reduction amongst the different groups computing different
  // partial outputs.
  if (Dh == Dh_MAX || vi < Dh) {
#pragma unroll
    for (int active_groups = V_PER_ITER; active_groups >= 2;
         active_groups /= 2) {

      // The midpoint in the number of active groups.
      int midpoint = active_groups / 2;

      // The upper part of active threads store to shared memory.
      if (vo >= midpoint && vo < active_groups && (Dh == Dh_MAX || vi < Dh)) {
        *reinterpret_cast<Out_sum *>(out_smem + (vo - midpoint) * Dh + vi) =
            out;
      }
      __syncthreads();

      // The bottom warps update their values.
      if (vo < midpoint && (Dh == Dh_MAX || vi < Dh)) {
        out = add(*reinterpret_cast<Out_sum const *>(out_smem + vo * Dh + vi),
                  out);
      }
      __syncthreads();
    }
  }

  // Output the final values.
  if (vo == 0 && (Dh == Dh_MAX || vi < Dh)) {
    convert_from_float(
        *reinterpret_cast<V_vec *>(output_ptr + request_idx * hidden_size +
                                   head_idx * per_head_size + vi),
        out);
  }
}

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
    apply_rotary_embedding_hf(DT *input_ptr,
                              cuFloatComplex *complex_input,
                              BatchConfig::PerTokenInfo const *tokenInfos,
                              float rope_theta,
                              bool llama3_rope,
                              float factor,
                              float low_freq_factor,
                              float high_freq_factor,
                              int original_max_position_embeddings,
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
                          token_idx * hidden_size * QKV_WEIGHT_NUM +
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

    float freq =
        pos * (1.0 / pow(rope_theta, (float)2 * pos_i / proj_size)); // θ_i

    if (llama3_rope) {
      float pi = CUDART_PI_F;
      float wavelen = 2 * pi / freq;
      float low_freq_wavelen =
          original_max_position_embeddings / low_freq_factor;
      float high_freq_wavelen =
          original_max_position_embeddings / high_freq_factor;
      if (wavelen < high_freq_wavelen) {
      } else if (wavelen > low_freq_wavelen) {
        freq = freq / factor;
      } else {
        assert(low_freq_wavelen != high_freq_wavelen);
        float smooth =
            (original_max_position_embeddings / wavelen - low_freq_factor) /
            (high_freq_factor - low_freq_factor);
        freq = ((1 - smooth) * freq / factor + smooth * freq);
      }
    }

    cuFloatComplex complex_pos = {cos(freq), sin(freq)};

    complex_input[i] = cuCmulf(complex_input[i], complex_pos);
    input_ptr[real_part_index] = complex_input[i].x;
    input_ptr[complex_part_index] = complex_input[i].y;
  }
}

template <typename DT>
__global__ void
    apply_rotary_embedding_bwd(DT *input_ptr,
                               cuFloatComplex *complex_input,
                               BatchConfig::PerTokenInfo const *tokenInfos,
                               float rope_theta,
                               bool llama3_rope,
                               float factor,
                               float low_freq_factor,
                               float high_freq_factor,
                               int original_max_position_embeddings,
                               int proj_size,
                               int num_tokens,
                               int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size) {
    // compute indexes to visit first half proj_size of each of q/k tensor.
    // devQKVProj has shape [num_tokens, qProjSize, num_heads, 3] in peft_bwd
    bool q_tensor = i < (num_tokens * hidden_size / 2);
    int real_i = q_tensor ? i : i - num_tokens * hidden_size / 2;
    assert(hidden_size % proj_size == 0);
    int num_heads = hidden_size / proj_size;

    int token_idx = real_i % num_tokens;
    int idx = (real_i / num_tokens) % (proj_size / 2);
    int head_idx = real_i / (num_tokens * proj_size / 2);
    assert(head_idx < num_heads);

    int complex_part_index = (q_tensor ? 0 : 1) * num_tokens * hidden_size +
                             head_idx * num_tokens * proj_size +
                             idx * num_tokens + token_idx;
    int real_part_index = complex_part_index + (proj_size / 2) * num_tokens;

    complex_input[i] = {input_ptr[real_part_index],
                        input_ptr[complex_part_index]};

    size_t pos = tokenInfos[token_idx].abs_depth_in_request;

    float freq =
        pos * (1.0 / pow(rope_theta, (float)2 * idx / proj_size)); // θ_i

    if (llama3_rope) {
      float pi = CUDART_PI_F;
      float wavelen = 2 * pi / freq;
      float low_freq_wavelen =
          original_max_position_embeddings / low_freq_factor;
      float high_freq_wavelen =
          original_max_position_embeddings / high_freq_factor;
      if (wavelen < high_freq_wavelen) {
      } else if (wavelen > low_freq_wavelen) {
        freq = freq / factor;
      } else {
        assert(low_freq_wavelen != high_freq_wavelen);
        float smooth =
            (original_max_position_embeddings / wavelen - low_freq_factor) /
            (high_freq_factor - low_freq_factor);
        freq = ((1 - smooth) * freq / factor + smooth * freq);
      }
    }

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
                        DT *output_ptr,
                        cudaStream_t stream) {

  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  assert(m->qSize == m->vSize && m->qSize == m->kSize);

  int num_tokens = bc->num_active_tokens();
  int parallelism = m->kProjSize * num_tokens * m->num_q_heads;
  size_t q_array_size = m->qProjSize * num_tokens * m->num_q_heads;

  if (m->scaling_query) {
    scaling_query_kernel<<<GET_BLOCKS(parallelism),
                           min(CUDA_NUM_THREADS, parallelism),
                           0,
                           stream>>>(output_ptr,
                                     m->qProjSize,
                                     num_tokens,
                                     m->num_q_heads,
                                     m->scaling_factor,
                                     m->hidden_size);
  }

  // Step 3: apply rotary embedding if needed
  if (m->rotary_embedding_meta->apply_rotary_embedding) {
    /*q&k*/
    parallelism = num_tokens * m->hidden_size;
    apply_rotary_embedding_hf<<<GET_BLOCKS(parallelism),
                                min(CUDA_NUM_THREADS, parallelism),
                                0,
                                stream>>>(
        output_ptr,
        m->complex_input,
        m->token_infos,
        m->rotary_embedding_meta->rope_theta,
        (m->rotary_embedding_meta->rope_type == "llama3"),
        m->rotary_embedding_meta->factor,
        m->rotary_embedding_meta->low_freq_factor,
        m->rotary_embedding_meta->high_freq_factor,
        m->rotary_embedding_meta->original_max_position_embeddings,
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
                            cudaStream_t stream) {
  int num_tokens = bc->num_active_infr_tokens();
  if (num_tokens > 0) {
    int parallelism = m->hidden_size * num_tokens;
    store_kv_cache<<<GET_BLOCKS(parallelism),
                     min(CUDA_NUM_THREADS, parallelism),
                     0,
                     stream>>>(static_cast<DT *>(m->devQKVProjArray),
                               static_cast<DT *>(m->keyCache),
                               static_cast<DT *>(m->valueCache),
                               m->token_infos,
                               num_tokens,
                               BatchConfig::max_sequence_length(),
                               m->hidden_size);
  }
}

#define LAUNCH_ATTENTION_SCORE_KERNEL(                                         \
    DT, Dh, Dh_MAX, THDS_PER_KEY, THREADS_PER_VALUE, THDS_PER_BLOCK, stream)   \
  smem_sz = smem_size_in_bytes<DT>(m->qProjSize,                               \
                                   BatchConfig::max_sequence_length(),         \
                                   THREADS_PER_VALUE,                          \
                                   THDS_PER_BLOCK);                            \
  compute_attention_kernel_generation_kernel<DT,                               \
                                             THDS_PER_BLOCK,                   \
                                             Dh,                               \
                                             Dh_MAX,                           \
                                             THDS_PER_KEY,                     \
                                             THREADS_PER_VALUE>                \
      <<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(                             \
          static_cast<DT *>(m->devQKVProjArray),                               \
          static_cast<DT *>(m->keyCache),                                      \
          static_cast<DT *>(m->valueCache),                                    \
          output_ptr,                                                          \
          scale,                                                               \
          BatchConfig::max_sequence_length(),                                  \
          m->qProjSize,                                                        \
          m->hidden_size,                                                      \
          m->request_infos)

template <typename DT>
void compute_attention_kernel_generation(IncMultiHeadSelfAttentionMeta const *m,
                                         BatchConfig const *bc,
                                         DT *output_ptr,
                                         cudaStream_t stream) {
  dim3 grid(m->num_q_heads, bc->num_generation_tokens);
  int const per_head_size = m->qProjSize;
  float scale = (*m->qk_prod_scaling) ? 1.0f / sqrt(m->kProjSize) : 1.0f;
  size_t smem_sz;
  if (per_head_size == 64) {
    constexpr int THREADS_PER_VALUE_64 = threads_per_value_t<DT, 64>::value;
    LAUNCH_ATTENTION_SCORE_KERNEL(
        DT, 64, 64, 4, THREADS_PER_VALUE_64, 128, stream);
  } else if (per_head_size == 128) {
    constexpr int THREADS_PER_VALUE_128 = threads_per_value_t<DT, 128>::value;
    LAUNCH_ATTENTION_SCORE_KERNEL(
        DT, 128, 128, 4, THREADS_PER_VALUE_128, 128, stream);
  } else {
    assert(false && "a unsupported head size");
  }
}

std::string get_fwd_dbg_folder(IncMultiHeadSelfAttentionMeta const *m,
                               int shard_id) {
  std::string op_name_without_uid =
      IncMultiHeadSelfAttention::get_op_name_without_uid(m);
  fs::path dst_filepath = get_dst_folder("fwd", m->decoding_step, shard_id);
  if (m->layer_guid.model_id > 0) {
    assert(false && "Model ID > 0 not supported yet");
  }
  std::string layername = "layers." +
                          std::to_string(m->layer_guid.transformer_layer_id) +
                          "." + op_name_without_uid;
  dst_filepath /= layername;
  return dst_filepath.string();
}

template <typename DT>
void inference_kernel(IncMultiHeadSelfAttentionMeta *m,
                      BatchConfig const *bc,
                      int shard_id,
                      DT const *qkv_ptr,
                      DT *output_ptr,
                      cudaStream_t stream) {

  // phase 0: copy calculated qkv into devQKVProjArray
  // [qProjSize, num_heads, 3, num_new_tokens]
  size_t qkv_proj_size =
      m->qProjSize * m->num_q_heads * QKV_WEIGHT_NUM * bc->num_active_tokens();

  cudaMemcpyAsync(m->devQKVProjArray,
                  qkv_ptr,
                  qkv_proj_size * sizeof(DT),
                  cudaMemcpyDeviceToDevice,
                  stream);

  // phase 1: Implement kernel to apply rotary embedding and scaling
  compute_qkv_kernel(
      m, bc, shard_id, static_cast<DT *>(m->devQKVProjArray), stream);
  update_kv_cache_kernel<DT>(m, bc, stream);

  if (bc->num_generation_tokens > 0) {
    // phase 3: Compute attention score for generation tokens
    compute_attention_kernel_generation<DT>(
        m, bc, static_cast<DT *>(m->attn_heads), stream);
  }

  if (bc->num_tokens > bc->num_generation_tokens) {
    // phase 4: Compute attention score for prompt tokens;
    compute_attention_kernel_prompt<DT>(m, bc, shard_id, stream);
  }

  int num_tokens = bc->num_active_tokens();
  cudaMemcpyAsync(output_ptr,
                  m->attn_heads,
                  m->oProjSize * num_tokens * sizeof(DT),
                  cudaMemcpyDeviceToDevice,
                  stream);
}

std::string get_peft_dbg_folder(IncMultiHeadSelfAttentionMeta const *m,
                                int shard_id) {
  std::string op_name_without_uid =
      IncMultiHeadSelfAttention::get_op_name_without_uid(m);
  fs::path dst_filepath = get_dst_folder("bwd", m->bwd_step, shard_id);
  if (m->layer_guid.model_id > 0) {
    assert(false && "Model ID > 0 not supported yet");
  }
  std::string layername = "layers." +
                          std::to_string(m->layer_guid.transformer_layer_id) +
                          "." + op_name_without_uid;
  dst_filepath /= layername;
  return dst_filepath.string();
}

__global__ void transposeAdd_half_kernel(
    half *out, half const *in, int width, int height, half alpha, half beta) {
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (int i = t_id; i < width * height; i += num_threads) {
    int row = i / width;
    int col = i % width;
    out[col * height + row] =
        alpha * in[row * width + col] + beta * out[col * height + row];
  }
}

__global__ void transposeAdd_float_kernel(float *out,
                                          float const *in,
                                          int width,
                                          int height,
                                          float alpha,
                                          float beta) {
  int t_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  for (int i = t_id; i < width * height; i += num_threads) {
    int row = i / width;
    int col = i % width;
    out[col * height + row] =
        alpha * in[row * width + col] + beta * out[col * height + row];
  }
}

template <typename DT>
void transposeAdd(DT *out,
                  const DT *in,
                  int width,
                  int height,
                  float alpha,
                  float beta,
                  cudaStream_t stream) {
  assert(false && "Unsupported data type");
}

template <>
void transposeAdd<float>(float *out,
                         float const *in,
                         int width,
                         int height,
                         float alpha,
                         float beta,
                         cudaStream_t stream) {
  transposeAdd_float_kernel<<<4, 1024, 0, stream>>>(
      out, in, width, height, alpha, beta);
}

template <>
void transposeAdd<half>(half *out,
                        half const *in,
                        int width,
                        int height,
                        float alpha,
                        float beta,
                        cudaStream_t stream) {
  transposeAdd_half_kernel<<<4, 1024, 0, stream>>>(
      out, in, width, height, __float2half(alpha), __float2half(beta));
}

template <typename DT>
void peft_bwd_kernel(IncMultiHeadSelfAttentionMeta const *m,
                     BatchConfig const *bc,
                     int shard_id,
                     DT *input_grad_ptr,
                     DT const *output_grad_ptr,
                     cudaStream_t stream) {
  assert(!m->offload);
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  assert(data_type_size(m->output_type[0]) == sizeof(DT));
  cudaDataType_t compute_type = cublas_data_type;

  for (int i = 0; i < bc->max_requests_per_batch(); i++) {
    if (bc->request_completed[i]) {
      continue;
    }
    if (!bc->requestsInfo[i].peft_bwd) {
      continue;
    }
    int num_tokens = bc->requestsInfo[i].num_tokens_in_batch;
    int num_total_tokens = bc->requestsInfo[i].first_token_depth_in_request +
                           bc->requestsInfo[i].num_tokens_in_batch;
    // Currently assume we are calculating gradients for all tokens
    // of a request
    assert(num_tokens == num_total_tokens);
    int kt_block_size = m->kProjSize;
    int kt_req_block_size =
        kt_block_size * m->num_q_heads * BatchConfig::max_sequence_length();
    int vt_block_size = m->vProjSize;
    int vt_req_block_size =
        vt_block_size * m->num_q_heads * BatchConfig::max_sequence_length();
    assert(m->qProjSize == m->kProjSize && m->kProjSize == m->vProjSize);
    // Step 1: copy gradient before final projection into workspace
    {
      int m_ = m->vProjSize * m->num_q_heads;
      int n_ = num_tokens;
      DT *C = static_cast<DT *>(m->handle.workSpace);
      cudaMemcpyAsync(C,
                      output_grad_ptr +
                          bc->requestsInfo[i].first_token_offset_in_batch *
                              m->oProjSize,
                      m_ * n_ * sizeof(DT),
                      cudaMemcpyDeviceToDevice,
                      stream);
      if (m->inference_debugging) {
        // save result to file for checking
        std::string filename =
            get_peft_dbg_folder(m, shard_id) + ".o_proj.input_gradient_0";
        save_tensor(C, m_ * n_, filename.c_str());
      }
    }
    // Step 2: compute gradients w.r.t. value
    {
      float alpha = 1.0f, beta = 0.0f;
      // matrix A: qk_prods_softmax
      // matrix A's layout: [num_new_tokens, total_tokens, num_heads]
      DT const *A = static_cast<DT *>(m->qk_prods_softmax);
      // matrix B: attn_heads gradients
      // matrix B's layout: [vProjSize * num_heads, num_new_tokens]
      DT const *B = static_cast<DT *>(m->handle.workSpace);
      // matrix C: gradients for value (saved as part of m->devQKVProjArray)
      // matrix C's layout: [num_tokens, qProjsize * num_heads, 3]
      DT *C = static_cast<DT *>(m->devQKVProjArray) +
              2 * num_tokens *
                  (m->qProjSize * m->num_q_heads); // skip over regions reserved
                                                   // for Q and K gradients
      // after transpositions
      int m_ = num_tokens;   // total_tokens
      int n_ = m->vProjSize; // num_new_tokens
      int k_ = num_tokens;   // num_new_tokens
      // before transpositions
      int lda = num_tokens; // num_new_tokens
      int ldb = m->vProjSize * m->num_q_heads;
      int ldc = num_tokens; // total_tokens
      // N.B. strides are applied before transpose operations
      int strideA = num_tokens * num_tokens; // num_new_tokens * total_tokens
      int strideB = m->vProjSize;
      int strideC = num_tokens * m->vProjSize;
      checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_T,
                                           m_,
                                           n_,
                                           k_,
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
                                           m->num_q_heads,
                                           compute_type,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      // save result to file for checking
      if (m->inference_debugging) {
        std::string filename =
            get_peft_dbg_folder(m, shard_id) + ".v_proj.input_gradient_0";
        save_tensor(C, m_ * n_ * m->num_q_heads, filename.c_str());
        std::string filename2 =
            get_peft_dbg_folder(m, shard_id) + ".qk_prods.softmax";
        save_tensor(A, m_ * k_ * m->num_q_heads, filename2.c_str());
      }
    }
    // Step 3: compute gradients w.r.t. the qk_prods_softmax tensor
    {
      float alpha = 1.0f, beta = 0.0f;
      // matrix A: attn_heads gradients
      // matrix A's layout: [vProjSize * num_heads, num_new_tokens]
      DT const *A = static_cast<DT *>(m->handle.workSpace);
      // matrix B: value cache
      // matrix B's layout: [vProjSize * num_heads, max_num_tokens, num_req]
      DT const *B = static_cast<DT *>(m->valueCache) + i * vt_req_block_size;
      // matrix C: qk_prods_softmax gradients
      // matrix C's layout: [num_new_tokens, total_tokens, num_heads]
      DT *C = static_cast<DT *>(m->qk_prods_softmax);
      // after transposition & striding
      int m_ = num_tokens; // num_new_tokens
      int n_ = num_tokens;
      int k_ = m->vProjSize;
      // before transposition and striding
      int lda = m->vProjSize * m->num_q_heads;
      int ldb = m->vProjSize * m->num_q_heads;
      int ldc = num_tokens; // num_new_tokens
      int strideA = m->vProjSize;
      int strideB = m->vProjSize;
      int strideC = num_tokens * num_tokens; // num_new_tokens * total_tokens

      checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_N,
                                           m_,
                                           n_,
                                           k_,
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
                                           m->num_q_heads,
                                           compute_type,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      if (m->inference_debugging) {
        std::string filename =
            get_peft_dbg_folder(m, shard_id) + ".qk_prods.softmax_grad";
        save_tensor(
            C, num_tokens * num_tokens * m->num_q_heads, filename.c_str());
        std::string filename2 = get_peft_dbg_folder(m, shard_id) + ".vcache";
        save_tensor(
            B, m->vProjSize * m->num_q_heads * num_tokens, filename2.c_str());
      }
    }
    // Step 4: softmax backpropagation
    {
      float alpha = 1.0f, beta = 0.0f;
      int n_param = m->num_q_heads;
      int c_param = num_tokens;
      int h_param = 1;
      int w_param = num_tokens;
      checkCUDNN(cudnnSetTensor4dDescriptor(m->qk_tensor,
                                            CUDNN_TENSOR_NCHW,
                                            cudnn_data_type,
                                            n_param,
                                            c_param,
                                            h_param,
                                            w_param));
      checkCUDNN(cudnnSoftmaxBackward(m->handle.dnn,
                                      CUDNN_SOFTMAX_ACCURATE,
                                      CUDNN_SOFTMAX_MODE_CHANNEL,
                                      &alpha,
                                      m->qk_tensor,
                                      m->softmax_activation_buffer,
                                      m->qk_tensor,
                                      m->qk_prods_softmax,
                                      &beta,
                                      m->qk_tensor,
                                      m->qk_prods));

      if (m->inference_debugging) {
        DT *C = static_cast<DT *>(m->qk_prods);
        std::string filename =
            get_peft_dbg_folder(m, shard_id) + ".qk_prods.softmax_grad_in";
        save_tensor(
            C, num_tokens * num_tokens * m->num_q_heads, filename.c_str());
      }

      //  TODO: fill all elements above diagonal to force causal attention
      size_t entries_above_diagonal = num_tokens * (num_tokens - 1) / 2;
      if (entries_above_diagonal > 0) {
        size_t parallelism = m->num_q_heads * entries_above_diagonal;
        fill_entries_above_diagonal<<<GET_BLOCKS(parallelism),
                                      min((size_t)CUDA_NUM_THREADS,
                                          parallelism),
                                      0,
                                      stream>>>(static_cast<DT *>(m->qk_prods),
                                                num_tokens,
                                                num_tokens,
                                                m->num_q_heads,
                                                entries_above_diagonal,
                                                DT(0.0f));
      }
      if (m->inference_debugging) {
        DT *C = static_cast<DT *>(m->qk_prods);
        std::string filename = get_peft_dbg_folder(m, shard_id) +
                               ".qk_prods.softmax_grad_in.masked";
        save_tensor(
            C, num_tokens * num_tokens * m->num_q_heads, filename.c_str());
      }
    }
    // Step 5: compute gradients w.r.t. key
    {
      float alpha = 1.0f, beta = 0.0f;
      if (*m->qk_prod_scaling) {
        alpha = 1.0f / sqrt(m->kProjSize);
      }
      // matrix A: gradients w.r.t. qk_prods
      // matrix A's layout: [num_new_tokens, num_tokens, num_heads]
      DT const *A = static_cast<DT *>(m->qk_prods);
      // matrix B: query activation (in query_activation_buffer)
      // matrix B's layout: [m->qProjSize * num_heads, num_new_tokens]
      DT const *B = static_cast<DT *>(m->query_activation_buffer);
      // matrix C: gradients for key (saved as part of m->devQKVProjArray)
      // matrix C's layout: [num_tokens, qProjsize * num_heads, 3]
      DT *C =
          static_cast<DT *>(m->devQKVProjArray) +
          num_tokens *
              (m->qProjSize *
               m->num_q_heads); // skip over regions reserved for Q gradients
      // after transposition & striding
      int m_ = num_tokens;
      int n_ = m->kProjSize;
      int k_ = num_tokens; // num_new_tokens
      // before transposition and striding
      int lda = num_tokens; // num_new_tokens
      int ldb = m->kProjSize * m->num_q_heads;
      int ldc = num_tokens;
      int strideA = num_tokens * num_tokens;
      int strideB = m->kProjSize;
      int strideC = num_tokens * m->kProjSize;
      checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_T,
                                           m_,
                                           n_,
                                           k_,
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
                                           m->num_q_heads,
                                           compute_type,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      if (m->inference_debugging) {
        std::string filename =
            get_peft_dbg_folder(m, shard_id) + ".query_activation";
        save_tensor(
            B, m->qProjSize * m->num_q_heads * num_tokens, filename.c_str());
        std::string filename2 =
            get_peft_dbg_folder(m, shard_id) + ".devkproj_pre";
        save_tensor(
            C, num_tokens * (m->qProjSize * m->num_q_heads), filename2.c_str());
      }
    }
    // Step 6: compute gradients w.r.t query
    {
      float alpha = 1.0f, beta = 0.0f;
      if (*m->qk_prod_scaling) {
        alpha = 1.0f / sqrt(m->kProjSize);
      }
      // matrix A: gradients w.r.t. qk_prods
      // matrix A's layout: [num_new_tokens, num_tokens, num_heads]
      DT const *A = static_cast<DT *>(m->qk_prods);
      // matrix B: key cache
      // matrix B's layout: [vProjSize * num_heads, max_num_tokens, num_req]
      DT const *B = static_cast<DT *>(m->keyCache) + i * kt_req_block_size;
      // matrix C: gradients for query (saved as part of m->devQKVProjArray)
      // matrix C's layout: [num_tokens, qProjsize * num_heads, 3]
      DT *C = static_cast<DT *>(m->devQKVProjArray);
      // after transposition & striding
      int m_ = num_tokens; // num_new_tokens
      int n_ = m->qProjSize;
      int k_ = num_tokens;
      // before transposition and striding
      int lda = num_tokens; // num_new_tokens
      int ldb = m->qProjSize * m->num_q_heads;
      int ldc = num_tokens;
      int strideA = num_tokens * num_tokens;
      int strideB = m->qProjSize;
      int strideC = num_tokens * m->qProjSize;
      checkCUDA(cublasGemmStridedBatchedEx(m->handle.blas,
                                           CUBLAS_OP_N,
                                           CUBLAS_OP_T,
                                           m_,
                                           n_,
                                           k_,
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
                                           m->num_q_heads,
                                           compute_type,
                                           CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      if (m->inference_debugging) {
        std::string filename =
            get_peft_dbg_folder(m, shard_id) + ".devQKVPRojArray_pre";
        save_tensor(C,
                    num_tokens * m->qProjSize * m->num_q_heads * 3,
                    filename.c_str());
      }
    }

    // Step 7: perform rotary position embeddings (RoPE) bwd
    {
      if (m->rotary_embedding_meta->apply_rotary_embedding) {
        assert(m->hidden_size == m->qProjSize * m->num_q_heads);
        assert(m->qProjSize == m->kProjSize);
        /*q&k*/
        int parallelism = num_tokens * m->hidden_size;
        DT *A = static_cast<DT *>(m->devQKVProjArray);
        apply_rotary_embedding_bwd<<<GET_BLOCKS(parallelism),
                                     min(CUDA_NUM_THREADS, parallelism),
                                     0,
                                     stream>>>(
            A,
            m->complex_input,
            m->token_infos,
            m->rotary_embedding_meta->rope_theta,
            (m->rotary_embedding_meta->rope_type == "llama3"),
            m->rotary_embedding_meta->factor,
            m->rotary_embedding_meta->low_freq_factor,
            m->rotary_embedding_meta->high_freq_factor,
            m->rotary_embedding_meta->original_max_position_embeddings,
            m->qProjSize,
            num_tokens,
            m->hidden_size);
        DT *C = static_cast<DT *>(m->devQKVProjArray);
        if (m->inference_debugging) {
          std::string filename =
              get_peft_dbg_folder(m, shard_id) + ".devQKVPRojArray";
          save_tensor(C,
                      num_tokens * m->qProjSize * m->num_q_heads * 3,
                      filename.c_str());
        }
      }

      // matrix C: gradients for key (saved as part of m->devQKVProjArray)
      // matrix C's layout: [num_tokens, qProjsize * num_heads, 3]
      DT *C =
          static_cast<DT *>(m->devQKVProjArray) +
          num_tokens *
              (m->qProjSize *
               m->num_q_heads); // skip over regions reserved for Q gradients
      if (m->inference_debugging) {
        std::string filename = get_peft_dbg_folder(m, shard_id) + ".devkproj";
        save_tensor(
            C, num_tokens * (m->qProjSize * m->num_q_heads), filename.c_str());
      }
    }

    // Step 8: compute gradients w.r.t. input
    {
      float alpha = 1.0f, beta = 0.0f;
      if (!m->reset_input_grads[0]) {
        beta = 1.0f;
      }
      // matrix B: gradients w.r.t. QKV (concatenated in devQKVArray)
      // matrix B's layout: [num_tokens, qProjsize * num_heads, 3]
      DT const *B = static_cast<DT *>(m->devQKVProjArray);
      // matrix C: gradients w.r.t. input
      // matrix C's layout: [m->qSize, num_tokens]
      DT *C = input_grad_ptr +
              bc->requestsInfo[i].first_token_offset_in_batch * m->qSize;
      // int m_ = m->qSize;
      int n_ = num_tokens;
      int k_ = m->num_q_heads * (m->qProjSize + m->kProjSize + m->vProjSize);

      // The original version uses existing result and attention's projection to
      // do further calculation in a way different than the usual dense layer,
      // they are off by a transpose. So an explicit transpose is needed here.
      // The add here is just for gradient accumulation.
      transposeAdd(C, B, n_, k_, alpha, beta, stream);

      if (m->inference_debugging) {
        std::string filename =
            get_peft_dbg_folder(m, shard_id) + ".self_attn.input_gradient_0";
        save_tensor(C, num_tokens * m->qSize, filename.c_str());
      }
    }
  }
}

} // namespace IncMultiHeadAttention
} // namespace Kernels

using namespace Kernels::IncMultiHeadAttention;

/*static*/
void IncMultiHeadSelfAttention::inference_kernel_wrapper(
    IncMultiHeadSelfAttentionMeta *m,
    BatchConfig const *bc,
    int shard_id,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  assert(input.data_type == output.data_type);

  if (input.data_type == DT_HALF) {
    Kernels::IncMultiHeadAttention::inference_kernel(
        m, bc, shard_id, input.get_half_ptr(), output.get_half_ptr(), stream);
  } else if (input.data_type == DT_FLOAT) {
    Kernels::IncMultiHeadAttention::inference_kernel(
        m, bc, shard_id, input.get_float_ptr(), output.get_float_ptr(), stream);
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
    printf("IncMultiHeadSelfAttention forward time = %.9fms\n", elapsed);
  }
}

/*static*/
void IncMultiHeadSelfAttention::peft_bwd_kernel_wrapper(
    IncMultiHeadSelfAttentionMeta *m,
    BatchConfig const *bc,
    int shard_id,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &output_grad) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }

  assert(input_grad.data_type == output_grad.data_type);

  if (input_grad.data_type == DT_HALF) {
    assert(!m->offload);
    Kernels::IncMultiHeadAttention::peft_bwd_kernel(m,
                                                    bc,
                                                    shard_id,
                                                    input_grad.get_half_ptr(),
                                                    output_grad.get_half_ptr(),
                                                    stream);
  } else if (input_grad.data_type == DT_FLOAT) {
    assert(!m->offload);
    Kernels::IncMultiHeadAttention::peft_bwd_kernel(m,
                                                    bc,
                                                    shard_id,
                                                    input_grad.get_float_ptr(),
                                                    output_grad.get_float_ptr(),
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
    printf("IncMultiHeadSelfAttention PEFT backward time = %.9fms\n", elapsed);
  }
}

IncMultiHeadSelfAttentionMeta::IncMultiHeadSelfAttentionMeta(
    FFHandler handler,
    IncMultiHeadSelfAttention const *attn,
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
                                    attn->rotary_embedding_meta,
                                    attn->scaling_query,
                                    attn->qk_prod_scaling,
                                    attn->position_bias,
                                    attn->scaling_factor,
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
    RotaryEmbeddingMeta _rotary_embedding_meta,
    bool _scaling_query,
    bool _qk_prod_scaling,
    bool _position_bias,
    float _scaling_factor,
    MemoryAllocator &gpu_mem_allocator,
    int num_samples,
    int _global_num_q_heads,
    int _global_num_kv_heads,
    int _num_q_heads,
    int _num_kv_heads,
    DataType _quantization_type,
    bool _offload)
    : OpMeta(handler, attn) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(cudnnSetStream(handler.dnn, stream));
  checkCUDNN(cudnnCreateTensorDescriptor(&qk_tensor));
  qSize = _qSize;
  kSize = _kSize;
  vSize = _vSize;
  // assume dimensions match for now
  assert(qSize == kSize);
  assert(kSize == vSize);
  qProjSize = _qProjSize;
  kProjSize = _kProjSize;
  assert(qProjSize == kProjSize); // required for attention QK.T matmul
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

  rotary_embedding_meta =
      (RotaryEmbeddingMeta *)calloc(1, sizeof(RotaryEmbeddingMeta));
  *rotary_embedding_meta = _rotary_embedding_meta;
  scaling_query = (bool *)calloc(1, sizeof(bool));
  *scaling_query = _scaling_query;
  scaling_factor = _scaling_factor;
  qk_prod_scaling = (bool *)calloc(1, sizeof(bool));
  *qk_prod_scaling = _qk_prod_scaling;
  position_bias = (bool *)calloc(1, sizeof(bool));
  *position_bias = _position_bias;

  // allocate memory for the seqArray and reserve space
  {
    int max_tokens_per_batch = infer_mode == TREE_VERIFY_MODE
                                   ? BatchConfig::max_verify_tokens_per_batch()
                                   : BatchConfig::max_tokens_per_batch();
    size_t qkv_max_proj_size = max_tokens_per_batch * (qProjSize * num_q_heads +
                                                       kProjSize * num_q_heads +
                                                       vProjSize * num_q_heads);
    size_t key_cache_size = 0, value_cache_size = 0;
    switch (infer_mode) {
      case INC_DECODING_MODE: {
        key_cache_size = num_q_heads * kProjSize *
                         BatchConfig::max_requests_per_batch() *
                         BatchConfig::max_sequence_length();
        value_cache_size = num_q_heads * vProjSize *
                           BatchConfig::max_requests_per_batch() *
                           BatchConfig::max_sequence_length();
        break;
      }
      case BEAM_SEARCH_MODE:
      case TREE_VERIFY_MODE: {
        // a K-ary tree max node is (k^n - 1) / 2
        key_cache_size = num_q_heads * kProjSize *
                         BeamSearchBatchConfig::max_requests_per_batch() *
                         (BatchConfig::max_sequence_length() +
                          BatchConfig::max_spec_tree_token_num());
        value_cache_size = num_q_heads * vProjSize *
                           BeamSearchBatchConfig::max_requests_per_batch() *
                           (BatchConfig::max_sequence_length() +
                            BatchConfig::max_spec_tree_token_num());
        break;
      }
      default:
        assert(false && "Unkown inference mode");
    }
    size_t requestinfo_size = BatchConfig::max_requests_per_batch();
    // size_t tokeninfo_size = max_tokens_per_batch;
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

    token_infos = static_cast<BatchConfig::PerTokenInfo *>(
        handler.batch_config_metadata->tokens_info);
    request_infos = static_cast<BatchConfig::PerRequestInfo *>(
        handler.batch_config_metadata->requestsInfo);

    if (offload) {
      qk_prods = gpu_mem_allocator.allocate_reserved_untyped(qk_prod_size *
                                                             size_of_dt);
      qk_prods_softmax = gpu_mem_allocator.allocate_reserved_untyped(
          qk_prod_size * size_of_dt);
      attn_heads = gpu_mem_allocator.allocate_reserved_untyped(attn_heads_size *
                                                               size_of_dt);
      complex_input =
          gpu_mem_allocator.allocate_reserved<cuFloatComplex>(complex_size);
    } else {
      qk_prods = gpu_mem_allocator.allocate_instance_untyped(qk_prod_size *
                                                             size_of_dt);
      qk_prods_softmax = gpu_mem_allocator.allocate_instance_untyped(
          qk_prod_size * size_of_dt);
      attn_heads = gpu_mem_allocator.allocate_instance_untyped(attn_heads_size *
                                                               size_of_dt);
      complex_input =
          gpu_mem_allocator.allocate_instance<cuFloatComplex>(complex_size);
    }

    // allocate more size for quantization data
    if (quantization_type != DT_NONE) {
      assert(offload);
    }
    if (!offload) {
      assert(gpu_mem_allocator.reserved_total_size ==
             gpu_mem_allocator.reserved_allocated_size);
    }
  }
  allocated_peft_buffer_size1 = 0;
  allocated_peft_buffer_size2 = 0;
  cudaStreamSynchronize(stream);
}

IncMultiHeadSelfAttentionMeta::~IncMultiHeadSelfAttentionMeta(void) {
  if (reserveInst != Realm::RegionInstance::NO_INST) {
    reserveInst.destroy();
  }
}

template void
    Kernels::IncMultiHeadAttention::compute_attention_kernel_generation<float>(
        IncMultiHeadSelfAttentionMeta const *m,
        BatchConfig const *bc,
        float *output_ptr,
        cudaStream_t stream);

template void
    Kernels::IncMultiHeadAttention::compute_attention_kernel_generation<half>(
        IncMultiHeadSelfAttentionMeta const *m,
        BatchConfig const *bc,
        half *output_ptr,
        cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::compute_qkv_kernel<float>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    int shard_id,
    float *output_ptr,
    cudaStream_t stream);

template void Kernels::IncMultiHeadAttention::compute_qkv_kernel<half>(
    IncMultiHeadSelfAttentionMeta const *m,
    BatchConfig const *bc,
    int shard_id,
    half *output_ptr,
    cudaStream_t stream);

}; // namespace FlexFlow

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
#include "flexflow/ops/kernels/inc_multihead_self_attention_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_utils.cuh"
#include "flexflow/ops/tree_inc_multihead_self_attention.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

#define WARP_SIZE 32

using namespace Kernels::IncMultiHeadAttention;

namespace Kernels {
namespace TreeIncMultiHeadAttention {

template <typename DT,
          int THREADS_PER_BLOCK,
          int Dh,
          int Dh_MAX,
          int THREADS_PER_KEY,
          int THREADS_PER_VALUE>
__global__ void compute_attention_kernel_fused_kernel(
    DT const *query,
    DT const *key_cache,
    DT const *value_cache,
    DT *output_ptr,
    float const scale,
    int const max_seq_length,
    int const max_token_per_batch,
    int per_head_size,
    int hidden_size,
    BatchConfig::PerRequestInfo *request_infos,
    int num_heads,
    int num_requests,
    BatchConfig::BitMask *causalMask,
    bool *request_completed,
    int qk_smem_sz) {

  // q, k
  using Q_vec = typename VEC_K<DT, THREADS_PER_KEY>::Type;
  using K_vec = typename VEC_K<DT, THREADS_PER_KEY>::Type;
  using V_vec = typename VEC_V<DT>::Type;
  using Out_sum = typename Vec_fp32_<V_vec>::Type;

  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

  constexpr int K_VEC_SIZE = sizeof(K_vec) / sizeof(DT);
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
  int const qlength =
      request_infos[batch_config_request_id].num_tokens_in_batch;

  BatchConfig::BitMask bitmask = causalMask[batch_config_request_id];

  int first_token_idx = 0;
  for (int r = 0; r < batch_config_request_id; r++) {
    first_token_idx +=
        request_completed[r] ? 0 : request_infos[r].num_tokens_in_batch;
  }

  bool prompt_phase = request_infos[batch_config_request_id].prompt_phase;
  int q_start =
      request_infos[batch_config_request_id].first_token_depth_in_request;

  // shared memory objects
  extern __shared__ char smem_[];

  float *qk_smem = reinterpret_cast<float *>(smem_);
  float *out_smem = reinterpret_cast<float *>(smem_ + qk_smem_sz);

  float qk_max = -FLT_MAX;

  // first WARPS_PER_BLOCK for store qk_max, second WARPS_PER_BLOCK for sum
  __shared__ float red_smem[WARPS_PER_BLOCK * 2];

  const DT *q_ptr = query + first_token_idx * hidden_size * QKV_WEIGHT_NUM +
                    head_idx * per_head_size;
  __shared__ Q_vec q_vecs[THREADS_PER_KEY][K_VECS_PER_THREAD];

  // the start offset of the element eg. (0, 1, 2, 3) * K_VEC_SIZE
  int ki = tidx % THREADS_PER_KEY * K_VEC_SIZE;
  int ki_o = tidx % THREADS_PER_KEY;
  // the first key's offset for this thread
  // ko = 0, 0, 0, 0, 1, 1, 1, 1, ....
  int ko = tidx / THREADS_PER_KEY;
  // load q tensor
  Q_vec q_vec[K_VECS_PER_THREAD];

  constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;
  // The number of keys per warp.
  constexpr int K_PER_WARP = WARP_SIZE / THREADS_PER_KEY;

  DT const *k_cache_batch =
      key_cache + batch_config_request_id * max_seq_length * hidden_size + ki;

  int ti_end =
      div_up(tlength - first_step, K_PER_WARP) * K_PER_WARP + first_step;

  for (int qi = 0; qi < qlength; qi += 1) {
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
      q_vecs[ki_o][ii] = *reinterpret_cast<Q_vec const *>(
          q_ptr + (hidden_size * QKV_WEIGHT_NUM * qi) + ki +
          ii * THREADS_PER_KEY * K_VEC_SIZE);

      // if (head_idx == 0 && request_idx == 1 && tidx == 0) {
      //     printf("laod q %d,  %d %.10f\n",
      //     request_idx,
      //            qi,q_vecs[ki_o][ii].x);
      //   }
    }

    __syncthreads();
    for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
      K_vec k[K_VECS_PER_THREAD];
      int const ti_circ = ti % max_seq_length;

      for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
        int jj = ii * THREADS_PER_KEY * K_VEC_SIZE;
        if (ti < tlength) {
          k[ii] = *reinterpret_cast<K_vec const *>(
              k_cache_batch + ti_circ * hidden_size + head_idx * per_head_size +
              jj);
        }
      }
      float qk = scale * Qk_dot<DT, THREADS_PER_KEY>::dot(q_vecs[ki_o], k);

      if (ti < tlength && tidx % THREADS_PER_KEY == 0) {
        bool const mask =
            prompt_phase ? (qi + q_start < ti)
                         : (ti >= bitmask.non_tree_cache_size &&
                            (!(bitmask.mask[ti - bitmask.non_tree_cache_size] &
                               (1 << qi))));

        qk_max = mask ? qk_max : fmaxf(qk_max, qk);

        // if (head_idx == 0 && !mask) {
        //   printf("tree attn qkqkqkqk request id %d qi%d, ti %d, %.10f, %.10f,
        //   %.10f, %d\n",
        //          request_idx,
        //          qi,
        //          ti,
        //          qk,
        //          q_vecs[ki_o][0].x,
        //          k[0].x,
        //          bitmask.non_tree_cache_size);
        // }
        qk_smem[ti - first_step] = mask ? 0.0f : qk;
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

    // if (head_idx == 0 && qi == 9 && tidx == 0) {
    //   printf("tree attn first token qk_max %f\n", qk_max);
    // }

    float exp_sum = 0.f;
    for (int ti = first_step + tidx; ti < tlength; ti += THREADS_PER_BLOCK) {
      bool const mask =
          prompt_phase ? (q_start + qi < ti)
                       : (ti >= bitmask.non_tree_cache_size &&
                          (!(bitmask.mask[ti - bitmask.non_tree_cache_size] &
                             (1 << qi))));
      float logit = mask ? 0.0f : __expf(qk_smem[ti - first_step] - qk_max);
      exp_sum += logit;
      qk_smem[ti - first_step] = mask ? 0.0f : logit;
    }

    // Compute the sum.
    exp_sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], exp_sum);

    // softmax
    float inv_sum = __fdividef(1.f, exp_sum + 1.e-6);
    for (int ti = first_step + tidx; ti < tlength; ti += THREADS_PER_BLOCK) {
      qk_smem[ti - first_step] *= inv_sum;
    }

    __syncthreads();

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
        value_cache + batch_config_request_id * max_seq_length * hidden_size +
        vi;

    if (Dh == Dh_MAX || vi < Dh) {
      for (int ti = first_step + vo; ti < tlength; ti += V_PER_ITER) {
        // Load the values from the cache.
        int const ti_circ = ti % max_seq_length;
        // int const real_cache_idx = topology.real_token_pos[sub_req_idx][ti];
        V_vec v = *reinterpret_cast<V_vec const *>(
            v_cache_batch + ti_circ * hidden_size + head_idx * per_head_size);

        if (ti < tlength) {
          bool const mask =
              prompt_phase
                  ? (q_start + qi < ti)
                  : (ti >= bitmask.non_tree_cache_size &&
                     (!(bitmask.mask[ti - bitmask.non_tree_cache_size] &
                        (1 << qi))));
          float logit = mask ? 0.0f : qk_smem[ti - first_step];
          out = FlexFlow::fma(logit, cast_to_float(v), out);
        }
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
      convert_from_float(*reinterpret_cast<V_vec *>(
                             output_ptr + (first_token_idx + qi) * hidden_size +
                             head_idx * per_head_size + vi),
                         out);
      // if (blockIdx.y == 0 && blockIdx.x == 0 && tidx == 0 && qi == 1) {
      //   printf("tree attn final value, %.9f, %.9f, %.9f, %.9f, %d, %d\n",
      //          out.x,
      //          out.y,
      //          out.z,
      //          out.w,
      //          vi,
      //          (first_token_idx + qi) * hidden_size + head_idx *
      //          per_head_size +
      //              vi);
      // }
    }
  }
}

template <typename DT>
__global__ void commit_tokens_kernel(
    DT const *devQKVProjArray,
    DT *kCache_ptr,
    DT *vCache_ptr,
    TreeVerifyBatchConfig::CommittedTokensInfo const *committedTokenInfos,
    int qProjSize,
    int kProjSize,
    int vProjSize,
    int num_tokens_to_commit,
    int num_active_tokens_in_last_batch,
    int max_seq_len,
    int hidden_size) {

  CUDA_KERNEL_LOOP(i, num_tokens_to_commit * hidden_size) {

    int token_pos = i / (hidden_size);
    int token_idx_in_last_batch = committedTokenInfos[token_pos].token_index;
    int offset = i % hidden_size;
    assert(token_idx_in_last_batch < num_active_tokens_in_last_batch);

    size_t val_idx = token_idx_in_last_batch * QKV_WEIGHT_NUM * hidden_size +
                     hidden_size + offset;

    DT kVal = devQKVProjArray[val_idx];
    DT vVal = devQKVProjArray[val_idx + hidden_size];

    int const req_id = committedTokenInfos[token_pos].request_index;
    int const tok_id = committedTokenInfos[token_pos].token_depth;

    kCache_ptr[req_id * (hidden_size * max_seq_len) + tok_id * hidden_size +
               offset] = kVal;
    vCache_ptr[req_id * (hidden_size * max_seq_len) + tok_id * hidden_size +
               offset] = vVal;
  }
}

template <typename DT>
void commit_tokens(TreeIncMultiHeadSelfAttentionMeta const *m,
                   TreeVerifyBatchConfig const *bc,
                   cudaStream_t stream) {
  int num_tokens_to_commit = bc->num_tokens_to_commit;
  if (num_tokens_to_commit > 0) {
    int parallelism = m->hidden_size * KV_WEIGHT_NUM * num_tokens_to_commit;
    commit_tokens_kernel<<<GET_BLOCKS(parallelism),
                           min(CUDA_NUM_THREADS, parallelism),
                           0,
                           stream>>>(
        static_cast<DT *>(m->devQKVProjArray),
        static_cast<DT *>(m->keyCache),
        static_cast<DT *>(m->valueCache),
        m->committed_token_infos,
        m->qProjSize,
        m->kProjSize,
        m->vProjSize,
        num_tokens_to_commit,
        m->num_active_tokens, // number of active tokens in previous batch
        BatchConfig::max_sequence_length() +
            BatchConfig::max_spec_tree_token_num(),
        m->hidden_size);
  }
}

template <typename DT>
__global__ void update_tree_branch_kv_cache(
    DT const *devQKVProjArray,
    DT *kCache_ptr,
    DT *vCache_ptr,
    TreeVerifyBatchConfig::PerTokenInfo const *tokenInfos,
    int qProjSize,
    int kProjSize,
    int vProjSize,
    int num_tokens_in_branch,
    int processed_tokens_in_batch,
    int total_tokens_in_batch,
    int max_seq_len,
    int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens_in_branch * hidden_size) {

    int token_idx = i / (hidden_size);
    int offset = i % hidden_size;

    token_idx += processed_tokens_in_batch; // get index in the whole batch
    size_t val_idx =
        token_idx * QKV_WEIGHT_NUM * hidden_size + hidden_size + offset;

    DT kVal = devQKVProjArray[val_idx];
    DT vVal = devQKVProjArray[val_idx + hidden_size];

    int const req_id = tokenInfos[token_idx].request_index;
    int const tok_id = tokenInfos[token_idx].abs_depth_in_request;
    kCache_ptr[req_id * (hidden_size * max_seq_len) + tok_id * hidden_size +
               offset] = kVal;
    vCache_ptr[req_id * (hidden_size * max_seq_len) + tok_id * hidden_size +
               offset] = vVal;
  }
}

template <typename DT>
__global__ void update_tree_branch_kv_cache_fused(
    DT const *devQKVProjArray,
    DT *kCache_ptr,
    DT *vCache_ptr,
    TreeVerifyBatchConfig::PerTokenInfo const *tokenInfos,
    BatchConfig::PerRequestInfo *request_infos,
    int qProjSize,
    int kProjSize,
    int vProjSize,
    int num_new_tokens,
    int max_seq_len,
    int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_new_tokens * hidden_size) {

    int token_idx = i / hidden_size;
    int offset = i % hidden_size;
    size_t val_idx =
        token_idx * QKV_WEIGHT_NUM * hidden_size + hidden_size + offset;

    DT kVal = devQKVProjArray[val_idx];
    DT vVal = devQKVProjArray[val_idx + hidden_size];

    int const req_id = tokenInfos[token_idx].request_index;
    // int const tok_id = tokenInfos[token_idx].abs_depth_in_request;

    int const request_token_offset =
        request_infos[req_id].first_token_offset_in_batch;
    int const first_token_depth =
        request_infos[req_id].first_token_depth_in_request;

    // if(i % hidden_size == 0){
    //   printf("update token request id: %d, %d, %d  real id %d, value%.10f\n",
    //   req_id, token_idx, request_token_offset,(token_idx + first_token_depth
    //   - request_token_offset), kVal);
    // }
    kCache_ptr[req_id * (hidden_size * max_seq_len) +
               (token_idx + first_token_depth - request_token_offset) *
                   hidden_size +
               offset] = kVal;
    vCache_ptr[req_id * (hidden_size * max_seq_len) +
               (token_idx + first_token_depth - request_token_offset) *
                   hidden_size +
               offset] = vVal;
  }
}

template <typename DT>
__global__ void tree_fill_entries_above_diagonal(DT *matrix,
                                                 size_t new_tokens,
                                                 size_t total_tokens_in_request,
                                                 size_t num_q_heads,
                                                 DT value) {
  CUDA_KERNEL_LOOP(i, new_tokens * total_tokens_in_request * num_q_heads) {
    // size_t head_idx = i / (new_tokens * total_tokens_in_request);
    size_t src_idx = (i / new_tokens) % total_tokens_in_request;
    size_t dst_idx = i % new_tokens + total_tokens_in_request - new_tokens;
    // Casual Mask
    if (src_idx > dst_idx) {
      matrix[i] = value;
    }
  }
}

template <typename DT>
void compute_attention_kernel(TreeIncMultiHeadSelfAttentionMeta const *m,
                              TreeVerifyBatchConfig const *bc,
                              int shard_id,
                              DT *output_ptr,
                              DT const *bias_ptr,
                              DT const *weight_ptr,
                              cudaStream_t stream) {
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  cudaDataType_t cublas_data_type = ff_to_cuda_datatype(m->output_type[0]);
  cudnnDataType_t cudnn_data_type = ff_to_cudnn_datatype(m->output_type[0]);
  assert(data_type_size(m->output_type[0]) == sizeof(DT));
#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
  cudaDataType_t compute_type = cublas_data_type;
#else
  // For best performance, set the default cublas compute type to
  // CUBLAS_COMPUTE_16F for half precision and to
  // CUBLAS_COMPUTE_32F_FAST_16F for full precision
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  if (m->output_type[0] == DT_FLOAT) {
    compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
  }
#endif
  // int num_requests = bc->num_active_requests();
  int processed_tokens_in_batch = 0;
  // int qkv_block_size =
  //     (m->qProjSize + m->kProjSize + m->vProjSize) * bc->num_active_tokens();
  int q_block_size = m->qProjSize;
  int kt_block_size = m->kProjSize;
  int kt_req_block_size =
      kt_block_size * m->num_q_heads * BatchConfig::max_sequence_length() +
      BatchConfig::max_spec_tree_token_num();
  int vt_block_size = m->vProjSize;
  int vt_req_block_size =
      vt_block_size * m->num_q_heads * BatchConfig::max_sequence_length() +
      BatchConfig::max_spec_tree_token_num();
  assert(m->qProjSize == m->kProjSize);

  for (int i = 0; i < bc->max_requests_per_batch(); i++) {
    if (bc->request_completed[i]) {
      continue;
    }
    assert(processed_tokens_in_batch ==
           bc->requestsInfo[i].first_token_offset_in_batch);
    int last_token_idx_of_the_request =
        processed_tokens_in_batch + bc->requestsInfo[i].num_tokens_in_batch - 1;
    while (processed_tokens_in_batch <= last_token_idx_of_the_request) {
      int num_new_tokens = 1;
      int j = processed_tokens_in_batch;
      while ((j + 1 <= last_token_idx_of_the_request) &&
             (bc->tokensInfo[j].abs_depth_in_request + 1 ==
              bc->tokensInfo[j + 1].abs_depth_in_request)) {
        j++;
        num_new_tokens++;
      }

      int total_tokens_in_request = bc->tokensInfo[j].abs_depth_in_request + 1;
      assert(num_new_tokens >= 1 && total_tokens_in_request >= num_new_tokens);
      {
        // update K-V cache
        int parallelism = m->hidden_size * KV_WEIGHT_NUM * num_new_tokens;
        update_tree_branch_kv_cache<<<GET_BLOCKS(parallelism),
                                      min(CUDA_NUM_THREADS, parallelism),
                                      0,
                                      stream>>>(
            static_cast<DT *>(m->devQKVProjArray),
            static_cast<DT *>(m->keyCache),
            static_cast<DT *>(m->valueCache),
            m->token_infos,
            m->qProjSize,
            m->kProjSize,
            m->vProjSize,
            num_new_tokens,            // num_tokens_in_branch
            processed_tokens_in_batch, // num_processed_tokens_in_batch
            m->num_active_tokens,      // total_tokens_in_batch
            BatchConfig::max_sequence_length(),
            m->hidden_size);
      }

      // bc->token_last_available_idx[i] + 1;
      // Compute (QK^T/sqrt(d_k))
      int m_ = num_new_tokens;
      int n = total_tokens_in_request;
      int k = m->qProjSize;
      int lda = k * m->num_q_heads * QKV_WEIGHT_NUM, ldb = k * m->num_q_heads,
          ldc = m_;
      int strideA = q_block_size;
      int strideB = kt_block_size;
      int strideC = num_new_tokens * total_tokens_in_request;

      // a flag of using this scaling alpha
      DT alpha = 1.0f, beta = 0.0f;
      if (*m->qk_prod_scaling) {
        alpha = static_cast<DT>(1.0f / sqrt(m->kProjSize));
      }
      // To get A, skip over Q entries from previous requests (same head)
      DT const *A = static_cast<DT *>(m->devQKVProjArray) +
                    processed_tokens_in_batch * m->qProjSize * m->num_q_heads *
                        QKV_WEIGHT_NUM;
      // To get B, skip over K entries from previous requests (all heads +
      // padding)
      DT const *B = static_cast<DT *>(m->keyCache) + i * kt_req_block_size;
      // To get C, skip over QK^T products from previous requests
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
      // add alibi position bias to qk production
      // add alibi position bias to qk production
      if (*m->position_bias) {
        size_t parallelism =
            m->num_q_heads * total_tokens_in_request * num_new_tokens;
        apply_position_bias_qkprd<<<GET_BLOCKS(parallelism),
                                    min((size_t)CUDA_NUM_THREADS, parallelism),
                                    0,
                                    stream>>>(C,
                                              num_new_tokens,
                                              total_tokens_in_request,
                                              m->num_q_heads,
                                              m->global_num_q_heads,
                                              shard_id);
      }

      // Fill all elements above diagonal in qk prods with -inf to force
      // causal attention.
      assert(num_new_tokens <= total_tokens_in_request);
      if (num_new_tokens > 1) {
        size_t parallelism =
            m->num_q_heads * num_new_tokens * total_tokens_in_request;
        tree_fill_entries_above_diagonal<<<GET_BLOCKS(parallelism),
                                           min((size_t)CUDA_NUM_THREADS,
                                               parallelism),
                                           0,
                                           stream>>>(
            C,
            num_new_tokens,
            total_tokens_in_request,
            m->num_q_heads,
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
      int c_param = total_tokens_in_request;
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
      // Matmul softmax(QK^T/sqrt(d_k)) by V
      alpha = 1.0f, beta = 0.0f;
      m_ = m->vProjSize;
      n = num_new_tokens;
      k = total_tokens_in_request;
      lda = m_ * m->num_q_heads, ldb = n, ldc = m_ * m->num_q_heads;
      strideA = vt_block_size;
      strideB = num_new_tokens * total_tokens_in_request;
      strideC = m->vProjSize;
      // To get A, skip over V^T entries from previous requests (all heads +
      // padding)
      A = static_cast<DT *>(m->valueCache) + i * vt_req_block_size;
      // To get B, skip over softmax(QK^T/sqrt(d_k)) entries from previous
      // requests (all heads)
      B = C_softmax;
      // To get C, skip over softmax(QK^T/sqrt(d_k))V products from previous
      // requests
      C = static_cast<DT *>(m->attn_heads) +
          processed_tokens_in_batch * m->num_q_heads * m->vProjSize;
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
      processed_tokens_in_batch += num_new_tokens;
    }
    // Before moving to the next request
    // check that we have finished all tokens of the request
    assert(last_token_idx_of_the_request + 1 == processed_tokens_in_batch);
  }
  // Project to output, save result directly on output tensor
  DT alpha = 1.0f, beta = 0.0f;
  int m_ = m->oProjSize;
  int k = m->vProjSize * m->num_q_heads;
  int n = processed_tokens_in_batch;
  int lda = k, ldb = k, ldc = m_;
  DT const *A = weight_ptr + m->qSize * (m->qProjSize * m->num_q_heads +
                                         m->kProjSize * m->num_q_heads +
                                         m->vProjSize * m->num_q_heads);
  DT const *B = static_cast<DT *>(m->attn_heads);
  DT *C = static_cast<DT *>(output_ptr);

  checkCUDA(cublasGemmEx(m->handle.blas,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
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
  if (*m->final_bias && shard_id == 0) {
    int parallelism = m->oProjSize * processed_tokens_in_batch;
    int qkv_weight_size = m->qProjSize * m->global_num_q_heads +
                          m->kProjSize * m->global_num_q_heads +
                          m->vProjSize * m->global_num_q_heads;
    apply_proj_bias_w<<<GET_BLOCKS(parallelism),
                        min(CUDA_NUM_THREADS, parallelism),
                        0,
                        stream>>>(output_ptr,
                                  bias_ptr,
                                  processed_tokens_in_batch,
                                  qkv_weight_size,
                                  m->oProjSize);
  }

  assert(processed_tokens_in_batch == bc->num_active_tokens());
}

#define LAUNCH_TREE_VERIFY_ATTENTION_SCORE_KERNEL(                             \
    DT, Dh, Dh_MAX, THDS_PER_KEY, THDS_PER_VALUE, THDS_PER_BLOCK, stream)      \
  smem_size_in_bytes_tree<DT>(m->qProjSize,                                    \
                              BatchConfig::max_sequence_length() +             \
                                  BatchConfig::max_spec_tree_token_num(),      \
                              THDS_PER_VALUE,                                  \
                              THDS_PER_BLOCK,                                  \
                              bc,                                              \
                              smem_sz);                                        \
  compute_attention_kernel_fused_kernel<DT,                                    \
                                        THDS_PER_BLOCK,                        \
                                        Dh,                                    \
                                        Dh_MAX,                                \
                                        THDS_PER_KEY,                          \
                                        THDS_PER_VALUE>                        \
      <<<grid, THDS_PER_BLOCK, smem_sz[1], stream>>>(                          \
          static_cast<DT *>(m->devQKVProjArray),                               \
          static_cast<DT *>(m->keyCache),                                      \
          static_cast<DT *>(m->valueCache),                                    \
          output_ptr,                                                          \
          scale,                                                               \
          BatchConfig::max_sequence_length() +                                 \
              BatchConfig::BatchConfig::max_spec_tree_token_num(),             \
          BatchConfig::max_tokens_per_batch(),                                 \
          m->qProjSize,                                                        \
          m->hidden_size,                                                      \
          m->request_infos,                                                    \
          m->num_q_heads,                                                      \
          bc->num_active_requests(),                                           \
          m->causalMask,                                                       \
          m->request_completed,                                                \
          smem_sz[0])

template <typename DT>
void compute_attention_kernel_fused(TreeIncMultiHeadSelfAttentionMeta const *m,
                                    TreeVerifyBatchConfig const *bc,
                                    DT *output_ptr,
                                    cudaStream_t stream) {

  // update the kv cache
  //  update K-V cache
  int num_new_tokens = bc->num_active_tokens();
  int parallelism = m->hidden_size * num_new_tokens;
  update_tree_branch_kv_cache_fused<<<GET_BLOCKS(parallelism),
                                      min(CUDA_NUM_THREADS, parallelism),
                                      0,
                                      stream>>>(
      static_cast<DT *>(m->devQKVProjArray),
      static_cast<DT *>(m->keyCache),
      static_cast<DT *>(m->valueCache),
      m->token_infos,
      m->request_infos,
      m->qProjSize,
      m->kProjSize,
      m->vProjSize,
      num_new_tokens,
      BatchConfig::max_sequence_length() +
          BatchConfig::max_spec_tree_token_num(),
      m->hidden_size);

  dim3 grid(m->num_q_heads, bc->num_active_requests());
  int const per_head_size = m->qProjSize;
  float scale = (*m->qk_prod_scaling) ? 1.0f / sqrt(m->kProjSize) : 1.0f;
  // 0->qk production size, 1->total shared size
  int smem_sz[2];
  if (per_head_size == 64) {
    constexpr int THREADS_PER_VALUE_64 = threads_per_value_t<DT, 64>::value;
    LAUNCH_TREE_VERIFY_ATTENTION_SCORE_KERNEL(
        DT, 64, 64, 4, THREADS_PER_VALUE_64, 128, stream);
  } else if (per_head_size == 128) {
    constexpr int THREADS_PER_VALUE_128 = threads_per_value_t<DT, 128>::value;
    LAUNCH_TREE_VERIFY_ATTENTION_SCORE_KERNEL(
        DT, 128, 128, 4, THREADS_PER_VALUE_128, 128, stream);
  } else {
    assert(false && "a unsupported head size");
  }
}

template <typename DT>
void inference_kernel(TreeIncMultiHeadSelfAttentionMeta *m,
                      TreeVerifyBatchConfig const *bc,
                      int shard_id,
                      DT const *input_ptr,
                      DT const *weight_ptr,
                      DT *output_ptr,
                      DT const *bias_ptr,
                      cudaStream_t stream) {
  // additional processing for weight uploading
  if (m->handle.offload_reserve_space != nullptr) {
    // Note that we update weight_ptr and bias_ptr when uploading weight and
    // bias
    cudaMemcpyAsync(m->weight_ptr,
                    weight_ptr,
                    m->weightSize,
                    cudaMemcpyHostToDevice,
                    stream);
    weight_ptr = static_cast<DT *>(m->weight_ptr);
    if (m->biasSize > 0) {
      cudaMemcpyAsync(
          m->bias_ptr, bias_ptr, m->biasSize, cudaMemcpyHostToDevice, stream);
      bias_ptr = static_cast<DT *>(m->bias_ptr);
    }
  }

  // copy committed tokens info to GPU for the commit_tokens kernel
  // Note that m->num_active_tokens stores the number of active
  // tokens in the previous batch, which is needed for committing
  // keys/values to the key-value cache
  // std::cout << "tokens to be committed: " << bc->num_tokens_to_commit <<
  // "\n";

  commit_tokens<DT>(m, bc, stream);

  // After commit we update m->num_active_tokens to be the number of active
  // tokens for the current batch
  m->num_active_tokens = bc->num_active_tokens();

  // here because we need postion info in infernece 1
  if (m->offload && m->biasSize > 0) {
    cudaMemcpyAsync(
        m->bias_ptr, bias_ptr, m->biasSize, cudaMemcpyHostToDevice, stream);
    bias_ptr = static_cast<DT *>(m->bias_ptr);
  }
  // phase 1: Implement kernel to compute KQV for input tokens
  compute_qkv_kernel(m,
                     bc,
                     shard_id,
                     input_ptr,
                     weight_ptr,
                     static_cast<DT *>(m->devQKVProjArray),
                     bias_ptr,
                     stream);

  // phase 2: No need to update key/val cache
  // IncMultiHeadSelfAttention::update_kv_cache_kernel(
  //    m, bc, stream);
  // use the new kernel
  compute_attention_kernel_fused<DT>(
      m, bc, static_cast<DT *>(m->attn_heads), stream);

  int processed_tokens_in_batch = bc->num_active_tokens();

  compute_o_prod_bias(m,
                      bc,
                      shard_id,
                      output_ptr,
                      weight_ptr,
                      bias_ptr,
                      processed_tokens_in_batch,
                      stream);
}

} // namespace TreeIncMultiHeadAttention
} // namespace Kernels

/*static*/
void TreeIncMultiHeadSelfAttention::inference_kernel_wrapper(
    TreeIncMultiHeadSelfAttentionMeta *m,
    TreeVerifyBatchConfig const *bc,
    int shard_id,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &weight,
    GenericTensorAccessorW const &output,
    GenericTensorAccessorR const &bias) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  bool use_bias = *m->qkv_bias || *m->final_bias;

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
    Kernels::TreeIncMultiHeadAttention::inference_kernel(
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
    Kernels::TreeIncMultiHeadAttention::inference_kernel(
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
  }
}

TreeIncMultiHeadSelfAttentionMeta::TreeIncMultiHeadSelfAttentionMeta(
    FFHandler handler,
    TreeIncMultiHeadSelfAttention const *attn,
    GenericTensorAccessorR const &weight,
    MemoryAllocator &gpu_mem_allocator,
    int num_samples,
    int _num_q_heads,
    int _num_kv_heads)
    : IncMultiHeadSelfAttentionMeta(handler,
                                    TREE_VERIFY_MODE,
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
                                    attn->offload),
      num_active_tokens(0) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(cudnnSetStream(handler.dnn, stream));

  // allocate memory for the seqArray and reserve space
  {

    causalMask = reinterpret_cast<BatchConfig::BitMask *>(
        reinterpret_cast<char *>(handler.batch_config_metadata) +
        sizeof(BatchConfig::tokensInfo) + sizeof(BatchConfig::requestsInfo));
    committed_token_infos =
        reinterpret_cast<TreeVerifyBatchConfig::CommittedTokensInfo *>(
            reinterpret_cast<char *>(handler.batch_config_metadata) +
            sizeof(BatchConfig::tokensInfo) +
            sizeof(BatchConfig::requestsInfo) +
            sizeof(BatchConfig::causalMask));

    request_completed = reinterpret_cast<bool *>(
        reinterpret_cast<char *>(handler.batch_config_metadata) +
        sizeof(BatchConfig::tokensInfo) + sizeof(BatchConfig::requestsInfo) +
        sizeof(BatchConfig::causalMask) +
        sizeof(TreeVerifyBatchConfig::committed_tokens));
  }

  cudaStreamSynchronize(stream);
}

TreeIncMultiHeadSelfAttentionMeta::~TreeIncMultiHeadSelfAttentionMeta(void) {
  if (committed_token_reserve_inst != Realm::RegionInstance::NO_INST) {
    committed_token_reserve_inst.destroy();
  }
}

}; // namespace FlexFlow

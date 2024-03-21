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
#include "flexflow/ops/spec_inc_multihead_self_attention.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

#define WARP_SIZE 32

// declare Legion names
using Legion::coord_t;
using Legion::Memory;
using namespace Kernels::IncMultiHeadAttention;

namespace Kernels {
namespace SpecIncMultiHeadSelfAttention {

template <typename DT,
          int THREADS_PER_BLOCK,
          int Dh,
          int Dh_MAX,
          int THREADS_PER_KEY,
          int THREADS_PER_VALUE>
__global__ void compute_spec_inc_attention_kernel_generation_kernel(
    DT const *query,
    DT const *key_cache,
    DT const *value_cache,
    DT *output_ptr,
    float const scale,
    int const max_seq_length,
    int per_head_size,
    int hidden_size,
    BatchConfig::PerRequestInfo *request_infos,
    BeamSearchBatchConfig::BeamSearchPerRequestInfo *beam_request_infos,
    BatchConfig::BitMask *causalMask,
    bool *request_completed) {

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
  // nth request idx
  int const request_idx = blockIdx.y;

  // request id in batch config
  int const batch_config_request_id =
      request_infos[request_idx].batch_config_request_id;

  // request_idx = re

  BatchConfig::BitMask bitmask = causalMask[batch_config_request_id];

  int const first_step = 0;

  // int const tlength =
  //     request_infos[batch_config_request_id].first_token_depth_in_request +
  //     request_infos[batch_config_request_id].num_tokens_in_batch;

  int const totalCacheSize =
      bitmask.non_tree_cache_size + bitmask.tree_size + bitmask.prompt_size - 1;

  int first_token_idx = 0;
  for (int r = 0; r < batch_config_request_id; r++) {
    first_token_idx += request_completed[r] ? 0 : causalMask[r].this_layer_size;
  }

  int const tree_branch_num =
      beam_request_infos[batch_config_request_id].sub_request_num;

  // shared memory objects
  extern __shared__ char smem_[];

  float *qk_smem = reinterpret_cast<float *>(smem_);
  float *out_smem = reinterpret_cast<float *>(smem_);

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
      div_up(totalCacheSize - first_step, K_PER_WARP) * K_PER_WARP + first_step;

  for (int qi = 0; qi < tree_branch_num; qi += 1) {
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
      q_vecs[ki_o][ii] = *reinterpret_cast<Q_vec const *>(
          q_ptr + (hidden_size * QKV_WEIGHT_NUM * qi) + ki +
          ii * THREADS_PER_KEY * K_VEC_SIZE);
    }

    int const query_token =
        bitmask.prompt_size + bitmask.tree_size - 1 - tree_branch_num + qi;

    __syncthreads();
    for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
      K_vec k[K_VECS_PER_THREAD];
      int const ti_circ = ti % max_seq_length;

      for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
        int jj = ii * THREADS_PER_KEY * K_VEC_SIZE;
        if (ti < totalCacheSize) {

          k[ii] = *reinterpret_cast<K_vec const *>(
              k_cache_batch + ti_circ * hidden_size + head_idx * per_head_size +
              jj);
        }
      }
      float qk = scale * Qk_dot<DT, THREADS_PER_KEY>::dot(q_vecs[ki_o], k);

      if (ti < totalCacheSize && tidx % THREADS_PER_KEY == 0) {
        // todo add alobi here
        // bool const mask = ti_circ >= totalCacheSize;
        bool const mask = (ti >= bitmask.non_tree_cache_size &&
                           (!(bitmask.mask[ti - bitmask.non_tree_cache_size] &
                              (1 << query_token))));

        // if (head_idx == 0 && ti == 0 && request_idx == 15 && !mask) {
        //   printf("spec inc attn qkqkqk  request id %d,  %.10f, %d\n",
        //          batch_config_request_id,
        //          ti,
        //          qk,
        //          qi);
        // }
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

    // if (blockIdx.y == 0 && blockIdx.x == 0 && tidx == 0) {
    //   printf("spec inc attn first token qk_max %.10f\n", qk_max);
    // }

    float exp_sum = 0.f;
    for (int ti = first_step + tidx; ti < totalCacheSize;
         ti += THREADS_PER_BLOCK) {
      bool const mask = (ti >= bitmask.non_tree_cache_size &&
                         (!(bitmask.mask[ti - bitmask.non_tree_cache_size] &
                            (1 << query_token))));
      float logit = mask ? 0.0f : __expf(qk_smem[ti - first_step] - qk_max);
      exp_sum += logit;
      qk_smem[ti - first_step] = mask ? 0.0f : logit;
    }

    // Compute the sum.
    exp_sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], exp_sum);

    // softmax
    float inv_sum = __fdividef(1.f, exp_sum + 1.e-6);
    for (int ti = first_step + tidx; ti < totalCacheSize;
         ti += THREADS_PER_BLOCK) {
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
      for (int ti = first_step + vo; ti < totalCacheSize; ti += V_PER_ITER) {
        // Load the values from the cache.
        int const ti_circ = ti % max_seq_length;
        V_vec v = *reinterpret_cast<V_vec const *>(
            v_cache_batch + ti_circ * hidden_size + head_idx * per_head_size);

        bool const mask = (ti >= bitmask.non_tree_cache_size &&
                           (!(bitmask.mask[ti - bitmask.non_tree_cache_size] &
                              (1 << query_token))));
        float logit = mask ? 0.0f : qk_smem[ti - first_step];
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
      convert_from_float(*reinterpret_cast<V_vec *>(
                             output_ptr + (first_token_idx + qi) * hidden_size +
                             head_idx * per_head_size + vi),
                         out);
    }
  }
}

template <typename DT>
__global__ void spec_inc_store_kv_cache(
    DT const *devQKVProjArray,
    DT *kCache_ptr,
    DT *vCache_ptr,
    BatchConfig::PerTokenInfo *tokenInfos,
    BatchConfig::PerRequestInfo *requestInfo,
    BeamSearchBatchConfig::BeamSearchPerTokenInfo *beamTokenInfos,
    BeamSearchBatchConfig::BeamSearchPerRequestInfo *beamRequestInfos,
    BatchConfig::BitMask *causalMask,
    int qProjSize,
    int kProjSize,
    int vProjSize,
    int num_tokens,
    int max_seq_len,
    bool is_root,
    int hidden_size) {
  CUDA_KERNEL_LOOP(i, num_tokens * hidden_size) {
    int token_idx = i / (hidden_size);
    int offset = i % hidden_size;

    size_t val_idx =
        token_idx * QKV_WEIGHT_NUM * hidden_size + hidden_size + offset;

    DT kVal = devQKVProjArray[val_idx];
    DT vVal = devQKVProjArray[val_idx + hidden_size];

    int const req_id = tokenInfos[token_idx].request_index;
    // int const tok_id = tokenInfos[token_idx].abs_depth_in_request;

    int const request_token_offset =
        requestInfo[req_id].first_token_offset_in_batch;

    BatchConfig::BitMask bitmask = causalMask[req_id];

    // if prompt token -> token id
    // if tree token:

    int const cache_idx = bitmask.prompt_size + bitmask.non_tree_cache_size +
                          bitmask.tree_size - 1 - bitmask.this_layer_size +
                          token_idx - request_token_offset;

    kCache_ptr[req_id * (hidden_size * max_seq_len) + (cache_idx)*hidden_size +
               offset] = kVal;
    vCache_ptr[req_id * (hidden_size * max_seq_len) + (cache_idx)*hidden_size +
               offset] = vVal;
  }
}

template <typename DT>
void update_kv_cache_kernel(SpecIncMultiHeadSelfAttentionMeta const *m,
                            BeamSearchBatchConfig const *bc,
                            cudaStream_t stream) {
  int num_tokens = bc->num_active_tokens();
  int curr_depth = bc->beamRequestsInfo[0].current_depth;
  if (num_tokens > 0) {
    int parallelism = m->hidden_size * KV_WEIGHT_NUM * num_tokens;
    spec_inc_store_kv_cache<<<GET_BLOCKS(parallelism),
                              min(CUDA_NUM_THREADS, parallelism),
                              0,
                              stream>>>(
        static_cast<DT *>(m->devQKVProjArray),
        static_cast<DT *>(m->keyCache),
        static_cast<DT *>(m->valueCache),
        m->token_infos,
        m->request_infos,
        m->beam_token_infos,
        m->beam_request_infos,
        m->causalMask,
        m->qProjSize,
        m->kProjSize,
        m->vProjSize,
        num_tokens,
        BatchConfig::max_sequence_length() +
            BatchConfig::max_spec_tree_token_num(),
        /*root*/ curr_depth == 0,
        m->hidden_size);
  }
}

#define LAUNCH_SPEC_INC_ATTENTION_SCORE_KERNEL(                                \
    DT, Dh, Dh_MAX, THDS_PER_KEY, THREADS_PER_VALUE, THDS_PER_BLOCK, stream)   \
  smem_sz = smem_size_in_bytes<DT>(m->qProjSize,                               \
                                   BatchConfig::max_sequence_length() +        \
                                       BatchConfig::max_spec_tree_token_num(), \
                                   THREADS_PER_VALUE,                          \
                                   THDS_PER_BLOCK);                            \
  compute_spec_inc_attention_kernel_generation_kernel<DT,                      \
                                                      THDS_PER_BLOCK,          \
                                                      Dh,                      \
                                                      Dh_MAX,                  \
                                                      THDS_PER_KEY,            \
                                                      THREADS_PER_VALUE>       \
      <<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(                             \
          static_cast<DT *>(m->devQKVProjArray),                               \
          static_cast<DT *>(m->keyCache),                                      \
          static_cast<DT *>(m->valueCache),                                    \
          output_ptr,                                                          \
          scale,                                                               \
          BatchConfig::max_sequence_length() +                                 \
              BatchConfig::max_spec_tree_token_num(),                          \
          m->qProjSize,                                                        \
          m->hidden_size,                                                      \
          m->request_infos,                                                    \
          m->beam_request_infos,                                               \
          m->causalMask,                                                       \
          m->request_completed)

template <typename DT>
void compute_spec_inc_attention_kernel_generation(
    SpecIncMultiHeadSelfAttentionMeta const *m,
    BeamSearchBatchConfig const *bc,
    DT *output_ptr,
    cudaStream_t stream) {
  // one block == one head per request
  // how many generation requests
  dim3 grid(m->num_q_heads, bc->get_speculative_request_num());
  int const per_head_size = m->qProjSize;
  float scale = (*m->qk_prod_scaling) ? 1.0f / sqrt(m->kProjSize) : 1.0f;
  size_t smem_sz;
  if (per_head_size == 64) {
    constexpr int THREADS_PER_VALUE_64 = threads_per_value_t<DT, 64>::value;
    LAUNCH_SPEC_INC_ATTENTION_SCORE_KERNEL(
        DT, 64, 64, 4, THREADS_PER_VALUE_64, 128, stream);
  } else if (per_head_size == 128) {
    constexpr int THREADS_PER_VALUE_128 = threads_per_value_t<DT, 128>::value;
    LAUNCH_SPEC_INC_ATTENTION_SCORE_KERNEL(
        DT, 128, 128, 4, THREADS_PER_VALUE_128, 128, stream);
  } else {
    assert(false && "a unsupported head size");
  }
}

template <typename DT>
__global__ void spec_fill_entries_above_diagonal(DT *matrix,
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
void compute_attention_kernel_prompt(SpecIncMultiHeadSelfAttentionMeta const *m,
                                     BeamSearchBatchConfig const *bc,
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
  int num_tokens = bc->num_active_tokens();
  int tokens_previous_requests = 0;
  int tokens_prev_requests_squares = 0;
  // int qkv_block_size =
  //     (m->qProjSize + m->kProjSize + m->vProjSize) * num_tokens;
  int q_block_size = m->qProjSize;

  int kt_block_size = m->kProjSize;
  int kt_req_block_size = kt_block_size * m->num_q_heads *
                          (BatchConfig::max_sequence_length() +
                           BatchConfig::max_spec_tree_token_num());
  int vt_block_size = m->vProjSize;
  int vt_req_block_size = vt_block_size * m->num_q_heads *
                          (BatchConfig::max_sequence_length() +
                           BatchConfig::max_spec_tree_token_num());
  assert(m->qProjSize == m->kProjSize);

  for (int i = 0; i < bc->max_requests_per_batch(); i++) {
    if (bc->request_completed[i] || (!bc->requestsInfo[i].prompt_phase) ||
        (bc->requestsInfo[i].num_tokens_in_batch == 0)) {
      continue;
    } else if (tokens_previous_requests < bc->num_generation_tokens) {
      tokens_previous_requests += bc->requestsInfo[i].num_tokens_in_batch;
      continue;
    }

    // all requests in prompt phase should only have one sub requests;
    assert(bc->sub_requests[i] == 1);
    // int num_new_tokens = bc->num_processing_tokens[i];
    // int total_tokens = bc->token_last_available_idx[i] + 1;

    int num_new_tokens = bc->requestsInfo[i].num_tokens_in_batch;
    int total_tokens = bc->requestsInfo[i].first_token_depth_in_request +
                       bc->requestsInfo[i].num_tokens_in_batch;

    if (num_new_tokens <= 0) {
      continue;
    }

    // Compute (QK^T/sqrt(d_k))
    int m_ = num_new_tokens;
    int n = total_tokens;
    int k = m->qProjSize;
    int lda = k * m->num_q_heads * QKV_WEIGHT_NUM, ldb = k * m->num_q_heads,
        ldc = m_;
    int strideA = q_block_size;
    int strideB = kt_block_size;
    int strideC = num_new_tokens * total_tokens;

    // a flag of using this scaling alpha
    DT alpha = 1.0f, beta = 0.0f;
    if (*m->qk_prod_scaling) {
      alpha = static_cast<DT>(1.0f / sqrt(m->kProjSize));
    }
    // To get A, skip over Q entries from previous requests (same head)
    DT const *A = static_cast<DT *>(m->devQKVProjArray) +
                  bc->requestsInfo[i].first_token_offset_in_batch *
                      m->qProjSize * m->num_q_heads * QKV_WEIGHT_NUM;
    // To get B, skip over K entries from previous requests (all heads +
    // padding)

    // print_tensor<float>((float*)A, 32, "A");
    DT const *B = static_cast<DT *>(m->keyCache) + i * kt_req_block_size;

    // if (i == 0 && sub_req_id == 0 &&
    //     bc->beam_slots.at(0).current_depth == 1) {
    //   int offset = (float *)B - m->keyCache;
    //   printf("key cache offset %d\n", kt_req_block_size);
    // }
    // To get C, skip over QK^T products from previous requests
    DT *C = static_cast<DT *>(m->qk_prods) +
            m->num_q_heads * tokens_prev_requests_squares;
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
    // print_tensor<float>((float*)C, 32, "C");
    // add alibi position bias to qk production
    // add alibi position bias to qk production
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
    // Fill all elements above diagonal in qk prods with -inf to force
    // causal attention.
    assert(num_new_tokens <= total_tokens);
    if (num_new_tokens > 1) {
      size_t parallelism = m->num_q_heads * num_new_tokens * total_tokens;
      spec_fill_entries_above_diagonal<<<GET_BLOCKS(parallelism),
                                         min((size_t)CUDA_NUM_THREADS,
                                             parallelism),
                                         0,
                                         stream>>>(C,
                                                   num_new_tokens,
                                                   total_tokens,
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
    DT *C_softmax = static_cast<DT *>(m->qk_prods_softmax) +
                    m->num_q_heads * tokens_prev_requests_squares;
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
    k = total_tokens;
    lda = m_ * m->num_q_heads, ldb = n, ldc = m_ * m->num_q_heads;
    strideA = vt_block_size;
    strideB = num_new_tokens * total_tokens;
    strideC = m->vProjSize;
    // To get A, skip over V^T entries from previous requests (all heads +
    // padding)
    A = static_cast<DT *>(m->valueCache) + i * vt_req_block_size;
    // To get B, skip over softmax(QK^T/sqrt(d_k)) entries from previous
    // requests (all heads)
    B = C_softmax;
    // To get C, skip over softmax(QK^T/sqrt(d_k))V products from previous
    // requests

    int token_offset = bc->requestsInfo[i].first_token_offset_in_batch;

    C = static_cast<DT *>(m->attn_heads) +
        (token_offset)*m->num_q_heads * m->vProjSize;
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

    tokens_previous_requests += num_new_tokens;
    tokens_prev_requests_squares += num_new_tokens * total_tokens;
  }

  if (tokens_previous_requests != (num_tokens - bc->num_generation_tokens)) {
    bc->print();
    printf("tokens_previous_requests: %i\n", tokens_previous_requests);
    printf("num_tokens: %i\n", num_tokens);
    printf("bc->num_generation_tokens: %i\n", bc->num_generation_tokens);
  }
  assert(tokens_previous_requests == (num_tokens - bc->num_generation_tokens));
}

template <typename DT>
void inference_kernel(SpecIncMultiHeadSelfAttentionMeta const *m,
                      BeamSearchBatchConfig const *bc,
                      int shard_id,
                      DT const *input_ptr,
                      DT const *weight_ptr,
                      DT *output_ptr,
                      DT const *bias_ptr,
                      cudaStream_t stream) {
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
  if (bc->num_generation_tokens > 0) {
    compute_spec_inc_attention_kernel_generation<DT>(
        m, bc, static_cast<DT *>(m->attn_heads), stream);
  }
  // phase 3: Compute attention score
  // 3 kernels for pahse 3: matmul1 - softmax - matmal2
  if (bc->num_tokens > bc->num_generation_tokens) {
    compute_attention_kernel_prompt(
        m, bc, shard_id, output_ptr, bias_ptr, weight_ptr, stream);
  }
  // compute output production and bias together for all tokens
  int num_tokens = bc->num_active_tokens();

  compute_o_prod_bias(
      m, bc, shard_id, output_ptr, weight_ptr, bias_ptr, num_tokens, stream);
}

} // namespace SpecIncMultiHeadSelfAttention
} // namespace Kernels

/*static*/
void SpecIncMultiHeadSelfAttention::inference_kernel_wrapper(
    SpecIncMultiHeadSelfAttentionMeta const *m,
    BeamSearchBatchConfig const *bc,
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

  assert(input.data_type == weight.data_type);
  assert(input.data_type == output.data_type);
  if (use_bias) {
    assert(input.data_type == bias.data_type);
  }

  if (input.data_type == DT_HALF) {
    half const *bias_ptr =
        use_bias ? bias.get_half_ptr() : static_cast<half const *>(nullptr);
    Kernels::SpecIncMultiHeadSelfAttention::inference_kernel(
        m,
        bc,
        shard_id,
        input.get_half_ptr(),
        weight.get_half_ptr(),
        output.get_half_ptr(),
        bias_ptr,
        stream);
  } else if (input.data_type == DT_FLOAT) {
    float const *bias_ptr =
        use_bias ? bias.get_float_ptr() : static_cast<float const *>(nullptr);
    Kernels::SpecIncMultiHeadSelfAttention::inference_kernel(
        m,
        bc,
        shard_id,
        input.get_float_ptr(),
        weight.get_float_ptr(),
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
    printf("SpecIncMultiHeadSelfAttention forward time = %.2fms\n", elapsed);
    // print_tensor<3, float>(acc_query.ptr, acc_query.rect,
    // "[Attention:forward:query]"); print_tensor<3, float>(acc_output.ptr,
    // acc_output.rect, "[Attention:forward:output]");
  }
}

SpecIncMultiHeadSelfAttentionMeta::SpecIncMultiHeadSelfAttentionMeta(
    FFHandler handler,
    SpecIncMultiHeadSelfAttention const *attn,
    GenericTensorAccessorR const &weight,
    MemoryAllocator &gpu_mem_allocator,
    int num_samples,
    int _num_q_heads,
    int _num_kv_heads)
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
                                    DT_NONE,
                                    false) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(cudnnSetStream(handler.dnn, stream));

  // allocate memory for the seqArray and reserve space
  {
    beam_token_infos =
        reinterpret_cast<BeamSearchBatchConfig::BeamSearchPerTokenInfo *>(
            reinterpret_cast<char *>(handler.batch_config_metadata) +
            sizeof(BatchConfig::tokensInfo) +
            sizeof(BatchConfig::requestsInfo));

    beam_request_infos =
        reinterpret_cast<BeamSearchBatchConfig::BeamSearchPerRequestInfo *>(
            reinterpret_cast<char *>(handler.batch_config_metadata) +
            sizeof(BatchConfig::tokensInfo) +
            sizeof(BatchConfig::requestsInfo) +
            sizeof(BeamSearchBatchConfig::beamTokenInfo));
    causalMask = reinterpret_cast<BatchConfig::BitMask *>(
        reinterpret_cast<char *>(handler.batch_config_metadata) +
        sizeof(BatchConfig::tokensInfo) + sizeof(BatchConfig::requestsInfo) +
        sizeof(BeamSearchBatchConfig::beamTokenInfo) +
        sizeof(BeamSearchBatchConfig::beamRequestsInfo));

    request_completed = reinterpret_cast<bool *>(
        reinterpret_cast<char *>(handler.batch_config_metadata) +
        sizeof(BatchConfig::tokensInfo) + sizeof(BatchConfig::requestsInfo) +
        sizeof(BeamSearchBatchConfig::beamTokenInfo) +
        sizeof(BeamSearchBatchConfig::beamRequestsInfo) +
        sizeof(BatchConfig::causalMask));
  }

  cudaStreamSynchronize(stream);
}

SpecIncMultiHeadSelfAttentionMeta::~SpecIncMultiHeadSelfAttentionMeta(void) {
  if (beam_search_reserve_inst != Realm::RegionInstance::NO_INST) {
    beam_search_reserve_inst.destroy();
  }
}

}; // namespace FlexFlow

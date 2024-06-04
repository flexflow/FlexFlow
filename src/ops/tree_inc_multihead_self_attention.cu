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
    /* Reserved: BatchConfig Updated */
    BatchConfig::PerRequestInfo *request_infos,
    int num_heads,
    int num_requests,
    BatchConfig::BitMask *causalMask,
    bool *request_available,
    int qk_smem_sz,
    bool prompt_phase) {

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

  // request id in batch config
  int requext_idx_in_batch = -1;
  int cnt_1 = 0;
  while (cnt_1 < request_idx + 1) {
    requext_idx_in_batch++;
    if (request_available[requext_idx_in_batch]) {
      cnt_1++;
    }
  }

  // threads converge
  //   __syncthreads();

  int const first_step = 0;

  int const tlength =
      request_infos[requext_idx_in_batch].first_token_index_in_request +
      request_infos[requext_idx_in_batch].num_tokens_in_batch;
  int const qlength = request_infos[requext_idx_in_batch].num_tokens_in_batch;

  __shared__ uint64_t bit_mask[BatchConfig::MAX_SPEC_TREE_TOKEN_NUM]
                              [BatchConfig::MAX_SPEC_TREE_TOKEN_NUM / 64];
  for (int i = tidx; i < qlength; i += THREADS_PER_BLOCK) {
    for (int j = 0; j < BatchConfig::MAX_SPEC_TREE_TOKEN_NUM / 64; j++) {
      bit_mask[i][j] = causalMask[requext_idx_in_batch].bit_mask[i].bits[j];
    }
  }

  int non_tree_cache_size =
      causalMask[requext_idx_in_batch].non_tree_cache_size;

  int const first_token_idx =
      request_infos[requext_idx_in_batch].first_token_offset_in_batch;

  int q_start =
      request_infos[requext_idx_in_batch].first_token_index_in_request;

  // shared memory objects
  extern __shared__ char smem_[];

  float *qk_smem = reinterpret_cast<float *>(smem_);
  float *out_smem = reinterpret_cast<float *>(smem_);

  float qk_max = -FLT_MAX;

  // first WARPS_PER_BLOCK for store qk_max, second WARPS_PER_BLOCK for sum
  __shared__ float red_smem[WARPS_PER_BLOCK * 2];

  const DT *q_ptr = query + first_token_idx * hidden_size +
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
      key_cache + requext_idx_in_batch * max_seq_length * hidden_size + ki;

  int ti_end =
      div_up(tlength - first_step, K_PER_WARP) * K_PER_WARP + first_step;

  for (int qi = 0; qi < qlength; qi += 1) {
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
      q_vecs[ki_o][ii] = *reinterpret_cast<Q_vec const *>(
          q_ptr + (hidden_size * qi) + ki +
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
#pragma unroll
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
            prompt_phase
                ? (qi + q_start < ti)
                : (ti >= non_tree_cache_size &&
                   (!test_bit(bit_mask, qi, ti - non_tree_cache_size)));

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
        //          bitmask->non_tree_cache_size);
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
                       : (ti >= non_tree_cache_size &&
                          (!test_bit(bit_mask, qi, ti - non_tree_cache_size)));
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
        value_cache + requext_idx_in_batch * max_seq_length * hidden_size + vi;

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
    DT *kCache_ptr,
    DT *vCache_ptr,
    BatchConfig::CommittedTokensInfo const *committedTokenInfos,
    int qProjSize,
    int kProjSize,
    int vProjSize,
    int token_pos,
    int num_active_tokens_in_last_batch,
    int max_seq_len,
    int hidden_size) {
  int const index_in_kv_cache = committedTokenInfos[token_pos].index_in_kv_cache;
  if (index_in_kv_cache == -1) {
    return;
  }

  int const req_id = committedTokenInfos[token_pos].request_index;
  int const tok_id = committedTokenInfos[token_pos].token_depth;

  size_t from_idx = req_id * (hidden_size * max_seq_len) +
                    index_in_kv_cache * hidden_size;
  size_t to_idx = req_id * (hidden_size * max_seq_len) +
                  tok_id * hidden_size;
  assert(to_idx <= from_idx);

  CUDA_KERNEL_LOOP(offset, hidden_size) {
    kCache_ptr[to_idx + offset] = kCache_ptr[from_idx + offset];
    vCache_ptr[to_idx + offset] = vCache_ptr[from_idx + offset];
  }
}

template <typename DT>
void commit_tokens(TreeIncMultiHeadSelfAttentionMeta const *m,
                   BatchConfig const *bc,
                   cudaStream_t stream) {
  int num_tokens_to_commit = bc->num_tokens_to_commit;
  // TODO: parallel across queries
  for (int i = 0; i < num_tokens_to_commit; i++) {
    int parallelism = m->hidden_size;
    commit_tokens_kernel<<<GET_BLOCKS(parallelism),
                           min(CUDA_NUM_THREADS, parallelism),
                           0,
                           stream>>>(
        static_cast<DT *>(m->keyCache),
        static_cast<DT *>(m->valueCache),
        m->committed_token_infos,
        m->qProjSize,
        m->kProjSize,
        m->vProjSize,
        i,
        m->num_active_tokens, // number of active tokens in previous batch
        BatchConfig::max_sequence_length() +
            BatchConfig::max_spec_tree_token_num(),
        m->hidden_size);
  }
}

template <typename DT>
__global__ void update_tree_branch_kv_cache_kernel(
    DT *devQKVProjArray,
    DT *kCache_ptr,
    DT *vCache_ptr,
    BatchConfig::PerTokenInfo const *tokenInfos,
    BatchConfig::PerRequestInfo *request_infos,
    int qProjSize,
    int kProjSize,
    int vProjSize,
    int token_idx,
    int max_seq_len,
    int hidden_size) {
  int const req_idx = tokenInfos[token_idx].request_index;
  int const token_abs_idx = tokenInfos[token_idx].abs_index_in_request;

  size_t from_idx =
        token_idx * QKV_WEIGHT_NUM * hidden_size;
  size_t to_idx = req_idx * (hidden_size * max_seq_len) +
                  token_abs_idx * hidden_size;

  CUDA_KERNEL_LOOP(offset, hidden_size) {
    kCache_ptr[to_idx + offset] = 
               devQKVProjArray[from_idx + hidden_size + offset];
    vCache_ptr[to_idx + offset] = 
               devQKVProjArray[from_idx + hidden_size * 2 + offset];
    devQKVProjArray[token_idx * hidden_size + offset] = 
               devQKVProjArray[from_idx + offset];
  }
}

#define LAUNCH_TREE_VERIFY_ATTENTION_SCORE_KERNEL(DT,                          \
                                                  Dh,                          \
                                                  Dh_MAX,                      \
                                                  THDS_PER_KEY,                \
                                                  THDS_PER_VALUE,              \
                                                  THDS_PER_BLOCK,              \
                                                  stream,                      \
                                                  prompt_phase)                \
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
      <<<grid,                                                                 \
         THDS_PER_BLOCK,                                                       \
         smem_sz[1],                                                           \
         stream>>>(static_cast<DT *>(m->devQKVProjArray),                      \
                   static_cast<DT *>(m->keyCache),                             \
                   static_cast<DT *>(m->valueCache),                           \
                   output_ptr,                                                 \
                   scale,                                                      \
                   BatchConfig::max_sequence_length() +                        \
                       BatchConfig::max_spec_tree_token_num(),                 \
                   BatchConfig::max_tokens_per_batch(),                        \
                   m->qProjSize,                                               \
                   m->hidden_size,                                             \
                   m->request_infos,                                           \
                   m->num_q_heads,                                             \
                   bc->num_active_requests(),                                  \
                   m->causalMask,                                              \
                   m->request_available,                                       \
                   smem_sz[0],                                                 \
                   prompt_phase)

template <typename DT>
void compute_attention_kernel_fused(TreeIncMultiHeadSelfAttentionMeta const *m,
                                    BatchConfig const *bc,
                                    DT *output_ptr,
                                    cudaStream_t stream) {

  // update the kv cache
  //  update K-V cache
  int num_new_tokens = bc->num_active_tokens();
  int parallelism = m->hidden_size;
  // TODO: parallel across queries
  for (int i = 0; i < num_new_tokens; i++) {
    update_tree_branch_kv_cache_kernel<<<GET_BLOCKS(parallelism),
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
        i,
        BatchConfig::max_sequence_length() +
            BatchConfig::max_spec_tree_token_num(),
        m->hidden_size);
  }

  // cudaEvent_t t_start, t_end;
  // cudaEventCreate(&t_start);
  // cudaEventCreate(&t_end);
  // cudaEventRecord(t_start, stream);

  dim3 grid(m->num_q_heads, bc->num_active_requests());
  int const per_head_size = m->qProjSize;
  float scale = (*m->qk_prod_scaling) ? 1.0f / sqrt(m->kProjSize) : 1.0f;
  // 0->qk production size, 1->total shared size
  // per_head_size: 128, thd_per_v:32, prompt_phase: 0
  int smem_sz[2];
  if (per_head_size == 64) {
    constexpr int THREADS_PER_VALUE_64 = threads_per_value_t<DT, 64>::value;
    LAUNCH_TREE_VERIFY_ATTENTION_SCORE_KERNEL(
        DT, 64, 64, 4, THREADS_PER_VALUE_64, 128, stream, bc->prompt_phase);
  } else if (per_head_size == 128) {
    constexpr int THREADS_PER_VALUE_128 = threads_per_value_t<DT, 128>::value;
    LAUNCH_TREE_VERIFY_ATTENTION_SCORE_KERNEL(
        DT, 128, 128, 4, THREADS_PER_VALUE_128, 128, stream, bc->prompt_phase);
  } else {
    assert(false && "a unsupported head size");
  }

  // cudaEventRecord(t_end, stream);
  // checkCUDA(cudaEventSynchronize(t_end));
  // float elapsed = 0;
  // checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  // printf("TreeIncMultiHeadSelfAttention part 2 time: %.2f ms\n", elapsed);
  // cudaEventDestroy(t_start);
  // cudaEventDestroy(t_end);

}

template <typename DT>
void inference_kernel(TreeIncMultiHeadSelfAttentionMeta *m,
                      BatchConfig const *bc,
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

  // Debug output:
  //   int size = m->hidden_size * BatchConfig::max_tokens_per_batch();
  //   float *temp_output = new float[size];
  //   cudaDeviceSynchronize();
  //   cudaMemcpy(
  //       temp_output, m->attn_heads, size * sizeof(float),
  //       cudaMemcpyDeviceToHost);
  //   printf("Output: ");
  //   for (int i = 0; i < 1; ++i) {
  //     float temp = 0;
  //     for (int j = 0; j < m->hidden_size; ++j) {
  //       temp += temp_output[i * m->hidden_size + j];
  //     }
  //     printf("%.6f ", temp);
  //   }
  //   printf("\n");

  //   delete[] temp_output;

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
    BatchConfig const *bc,
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
        sizeof(BatchConfig::tokensInfo) + sizeof(BatchConfig::requestsInfo) +
        sizeof(BatchConfig::request_available));
    committed_token_infos =
        reinterpret_cast<BatchConfig::CommittedTokensInfo *>(
            reinterpret_cast<char *>(handler.batch_config_metadata) +
            sizeof(BatchConfig::tokensInfo) +
            sizeof(BatchConfig::requestsInfo) +
            sizeof(BatchConfig::request_available) +
            sizeof(BatchConfig::causalMask));
  }

  cudaStreamSynchronize(stream);
}

TreeIncMultiHeadSelfAttentionMeta::~TreeIncMultiHeadSelfAttentionMeta(void) {
  if (committed_token_reserve_inst != Realm::RegionInstance::NO_INST) {
    committed_token_reserve_inst.destroy();
  }
}

}; // namespace FlexFlow

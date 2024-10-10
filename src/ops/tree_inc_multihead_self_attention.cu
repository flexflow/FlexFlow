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
#include "flashinfer/prefill_attention_decl.cuh"
#include "flexflow/ffconst_utils.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_kernels.h"
#include "flexflow/ops/kernels/inc_multihead_self_attention_utils.cuh"
#include "flexflow/ops/tree_inc_multihead_self_attention.h"
#include "flexflow/utils/cuda_helper.h"

#include <sstream>
#include <stdexcept>

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

#define WARP_SIZE 32

using namespace Kernels::IncMultiHeadAttention;

namespace Kernels {
namespace TreeIncMultiHeadAttention {

using flashinfer::BatchPrefillHandler;
using flashinfer::BatchPrefillWithPagedKVCacheWrapperDispatched;
using flashinfer::LogitsPostHook;
using flashinfer::MaskMode;
using flashinfer::paged_kv_t;
using flashinfer::PageStorage;
using flashinfer::PosEncodingMode;
using flashinfer::QKVLayout;

template <typename DT>
__global__ void
    update_qkv_in_batch_verify_kernel(DT *qkv_proj_array,
                               half *qTmp_ptr,
                               half *kvCache_ptr,
                               int32_t *kv_indptr,
                               int32_t *kv_page_indices,
                               BatchConfig::PerTokenInfo const *tokenInfos,
                               int const max_num_pages,
                               int num_q_heads,
                               int num_kv_heads,
                               int head_dim,
                               int num_new_tokens) {
  int const q_hidden_size = num_q_heads * head_dim;
  int const temp_kv_hidden_size = num_q_heads * head_dim; // temporary hard code
  int const kv_hidden_size = num_kv_heads * head_dim;
  int const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int const token_idx = thread_idx / q_hidden_size;
  int const offset = thread_idx % q_hidden_size;
  if (token_idx >= num_new_tokens) {
    return;
  }

  int const req_idx = tokenInfos[token_idx].request_index;
  int token_abs_idx = tokenInfos[token_idx].abs_index_in_request;

  size_t from_idx = token_idx * (q_hidden_size + temp_kv_hidden_size * 2);
  qTmp_ptr[token_idx * q_hidden_size + offset] =
      static_cast<half>(qkv_proj_array[from_idx + offset]);

  if (offset < kv_hidden_size) {
    int start = kv_indptr[req_idx];
    int page_idx = kv_page_indices[start + (token_abs_idx / kPagesize)];
    size_t to_k_idx = get_k_entry_offset_verify(
           token_abs_idx, page_idx, num_kv_heads, head_dim),
           to_v_idx = get_v_entry_offset_verify(
           token_abs_idx, page_idx, num_kv_heads, head_dim);
    // key and value cache should be stored interleaved
    int const stride = num_q_heads / num_kv_heads;
    int const kv_offset =
        offset / head_dim * stride * head_dim + offset % head_dim;
    kvCache_ptr[to_k_idx + offset] =
        static_cast<half>(qkv_proj_array[from_idx + q_hidden_size + kv_offset]);
    kvCache_ptr[to_v_idx + offset] =
        static_cast<half>(qkv_proj_array[from_idx + q_hidden_size +
                                         temp_kv_hidden_size + kv_offset]);
  }
}

template <typename DT>
void update_qkv_in_batch_verify(IncMultiHeadSelfAttentionMeta const *m,
                         BatchConfig const *bc,
                         cudaStream_t stream) {
  int num_new_tokens = bc->num_active_tokens();
  if (num_new_tokens == 0) {
    return;
  }
  int parallelism = m->local_hidden_size * num_new_tokens;
  int const max_num_pages =
      round_up_pages(BatchConfig::max_sequence_length() +
                     BatchConfig::max_spec_tree_token_num());
  update_qkv_in_batch_verify_kernel<<<GET_BLOCKS(parallelism),
                               min(CUDA_NUM_THREADS, parallelism),
                               0,
                               stream>>>(static_cast<DT *>(m->devQKVProjArray),
                                         static_cast<half *>(m->queryTmp),
                                         static_cast<half *>(m->kvCache),
                                         m->handle.tree_verify_attention_metadata->kv_indptr,
                                         m->handle.tree_verify_attention_metadata->kv_indices,
                                         m->token_infos,
                                         max_num_pages,
                                         m->num_q_heads,
                                         m->num_kv_heads,
                                         m->qk_dim,
                                         num_new_tokens);
}

__global__ void commit_tokens_kernel(
    half *kCache_ptr,
    int32_t *kv_indptr,
    int32_t *kv_page_indices,
    BatchConfig::CommittedTokensInfo const *committedTokenInfos,
    bool const *request_available,
    int num_requests,
    int num_kv_heads,
    int head_dim,
    int const *num_committed_tokens,
    int const max_num_pages) {
  int const kv_hidden_size = num_kv_heads * head_dim;
  int const idx = blockIdx.x * blockDim.x + threadIdx.x;
  int const request_compact_idx = idx / kv_hidden_size;
  int const offset = idx % kv_hidden_size;
  // request id in batch config
  int requext_idx_in_batch = -1;
  int cnt_1 = 0;
  while (cnt_1 < request_compact_idx + 1) {
    requext_idx_in_batch++;
    if (request_available[requext_idx_in_batch]) {
      cnt_1++;
    }
  }

  int start = kv_indptr[requext_idx_in_batch];
  int end = kv_indptr[requext_idx_in_batch + 1] - 1;

  for (int i = 0; i < *num_committed_tokens; i++) {
    if (committedTokenInfos[i].request_index == requext_idx_in_batch) {
      int const index_in_kv_cache = committedTokenInfos[i].index_in_kv_cache;
      if (index_in_kv_cache == -1) {
        continue;
      }

      // int const req_id = committedTokenInfos[i].request_index;
      int const tok_id = committedTokenInfos[i].token_depth;
      int const page_to_idx = committedTokenInfos[i].token_depth / kPagesize;
      int const page_from_idx = committedTokenInfos[i].index_in_kv_cache / kPagesize;

      size_t from_k_idx = get_k_entry_offset_verify(
                  committedTokenInfos[i].index_in_kv_cache, page_from_idx, num_kv_heads, head_dim),
             from_v_idx = get_v_entry_offset_verify(
                  committedTokenInfos[i].index_in_kv_cache, page_from_idx, num_kv_heads, head_dim);
      size_t to_k_idx = get_k_entry_offset_verify(
                 committedTokenInfos[i].token_depth, page_to_idx, num_kv_heads, head_dim),
             to_v_idx = get_v_entry_offset_verify(
                 committedTokenInfos[i].token_depth, page_to_idx, num_kv_heads, head_dim);
      assert(to_k_idx <= from_k_idx);

      kCache_ptr[to_k_idx + offset] = kCache_ptr[from_k_idx + offset];
      kCache_ptr[to_v_idx + offset] = kCache_ptr[from_v_idx + offset];
    }
  }
}

void commit_tokens(TreeIncMultiHeadSelfAttentionMeta const *m,
                   BatchConfig const *bc,
                   cudaStream_t stream) {
  //   cudaEvent_t t_start, t_end;
  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  int const max_num_pages =
      round_up_pages(BatchConfig::max_sequence_length() +
                     BatchConfig::max_spec_tree_token_num());
  int const num_requests = bc->num_active_requests();
  int parallelism = m->num_kv_heads * m->qk_dim * num_requests;
  commit_tokens_kernel<<<GET_BLOCKS(parallelism),
                         min(CUDA_NUM_THREADS, parallelism),
                         0,
                         stream>>>(static_cast<half *>(m->kvCache),
                                   m->handle.tree_verify_attention_metadata->kv_indptr,
                                   m->handle.tree_verify_attention_metadata->kv_indices,
                                   m->committed_token_infos,
                                   m->request_available,
                                   num_requests,
                                   m->num_kv_heads,
                                   m->qk_dim,
                                   m->num_tokens_to_commit,
                                   max_num_pages);
  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   float elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   printf("Commit token time: %.2f ms\n", elapsed);
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
}

template <typename DT>
void tree_verify_attention(TreeIncMultiHeadSelfAttentionMeta *m,
                           BatchConfig const *bc,
                           DT *output_ptr,
                           cudaStream_t stream) {
  //   int device;
  //   checkCUDA(cudaGetDevice(&device));
  //   cudaEvent_t t_start, t_end;
  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  // global constant parameters
  uint32_t const num_q_heads = m->num_q_heads;
  uint32_t const num_kv_heads = m->num_kv_heads;
  uint32_t const head_dim = m->qk_dim;
  uint32_t const batch_size = bc->num_active_requests();
  float const sm_scale = (*m->qk_prod_scaling) ? 1.0f / sqrt(m->qk_dim) : 1.0f;

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   if (device == 0) {
  //     std::cout << "Update custom mask time: " << elapsed << " ms\n";
  //   }

  half *q = static_cast<half *>(m->queryTmp),
       *kv = static_cast<half *>(m->kvCache),
       *o = static_cast<half *>(m->outputTmp);
  paged_kv_t<PageStorage::kIndices, half, int32_t> paged_kv(
      num_kv_heads,
      kPagesize,
      head_dim,
      batch_size,
      QKVLayout::kNHD,
      kv,
      m->handle.tree_verify_attention_metadata->kv_indices,
      m->handle.tree_verify_attention_metadata->kv_indptr,
      m->handle.tree_verify_attention_metadata->kv_last_page_len);

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   float elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   if (device == 0) {
  //     printf("    attn prep time: %.4f ms\n", elapsed);
  //   }
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  BatchPrefillHandler *handler = nullptr;

  if (!bc->prompt_phase) {
    assert(m->handle.tree_verify_attention_metadata->decode_handler_collections
                   .count(batch_size) != 0 &&
           "Handler is not initialized");
    handler = static_cast<BatchPrefillHandler *>(
        m->handle.tree_verify_attention_metadata
            ->decode_handler_collections[batch_size]);
  } else {
    assert(m->handle.tree_verify_attention_metadata->prompt_handler_collections
                   .count(batch_size) != 0 &&
           "Handler is not initialized");
    handler = static_cast<BatchPrefillHandler *>(
        m->handle.tree_verify_attention_metadata
            ->prompt_handler_collections[batch_size]);
  }

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   if (device == 0) {
  //     printf("    BeginForward time: %.4f ms\n", elapsed);
  //   }
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  DISPATCH_HEADDIM(head_dim, HEAD_DIM, {
    cudaError_t result;
    if (bc->prompt_phase) {
      result =
          BatchPrefillWithPagedKVCacheWrapperDispatched<PageStorage::kIndices,
                                                        HEAD_DIM,
                                                        LogitsPostHook::kNone,
                                                        PosEncodingMode::kNone,
                                                        false,
                                                        MaskMode::kCausal,
                                                        half,
                                                        half,
                                                        half,
                                                        int32_t>(
              handler,
              q,
              m->handle.tree_verify_attention_metadata->q_indptr,
              /*q_offset=*/nullptr,
              paged_kv,
              /*custom_mask=*/nullptr,
              /*qk_indptr=*/nullptr,
              o,
              /*lse=*/nullptr,
              num_q_heads,
              /*window_left=*/-1,
              /*logits_soft_cap=*/0.f,
              sm_scale,
              /*rope_scale=*/1.f,
              /*rope_theta=*/static_cast<float>(1e4),
              stream);
    } else {
      result =
          BatchPrefillWithPagedKVCacheWrapperDispatched<PageStorage::kIndices,
                                                        HEAD_DIM,
                                                        LogitsPostHook::kNone,
                                                        PosEncodingMode::kNone,
                                                        false,
                                                        MaskMode::kCustom,
                                                        half,
                                                        half,
                                                        half,
                                                        int32_t>(
              handler,
              q,
              m->handle.tree_verify_attention_metadata->q_indptr,
              /*q_offset=*/nullptr,
              paged_kv,
              m->handle.tree_verify_attention_metadata->custom_mask,
              m->handle.tree_verify_attention_metadata->qk_indptr,
              o,
              /*lse=*/nullptr,
              num_q_heads,
              /*window_left=*/-1,
              /*logits_soft_cap=*/0.f,
              sm_scale,
              /*rope_scale=*/1.f,
              /*rope_theta=*/static_cast<float>(1e4),
              stream);
    }
    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to run "
                               "TreeVerifyAttentionKernel: " +
                               std::string(cudaGetErrorString(result)));
    }
  });

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   if (device == 0) {
  //     printf("    actual attn time: %.4f ms\n", elapsed);
  //   }
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  produce_output(m, bc, output_ptr, stream);

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   if (device == 0) {
  //     printf("    produce_output_kernel time: %.4f ms\n", elapsed);
  //   }
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
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

  //   int device;
  //   checkCUDA(cudaGetDevice(&device));
  //   cudaEvent_t t_start, t_end;
  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

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

  if (!bc->prompt_phase) {
    commit_tokens(m, bc, stream);
  }

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   float elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   if (device == 0) {
  //     std::cout << "Commit tokens time: " << elapsed << " ms\n";
  //   }

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  // After commit we update m->num_active_tokens to be the number of active
  // tokens for the current batch
  m->num_active_tokens = bc->num_active_tokens();

  // here because we need postion info in infernece 1
  if (m->offload && m->biasSize > 0) {
    cudaMemcpyAsync(
        m->bias_ptr, bias_ptr, m->biasSize, cudaMemcpyHostToDevice, stream);
    bias_ptr = static_cast<DT *>(m->bias_ptr);
  }
  // Implement kernel to compute KQV for input tokens
  compute_qkv(m,
              bc,
              shard_id,
              input_ptr,
              weight_ptr,
              static_cast<DT *>(m->devQKVProjArray),
              bias_ptr,
              stream);

  apply_pos_encoding_to_tokens_in_batch(
      m, bc, static_cast<DT *>(m->devQKVProjArray), stream);

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   if (device == 0) {
  //     std::cout << "Compute qkv time: " << elapsed << " ms\n";
  //   }

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  // Update key-val cache, compact q array
  update_qkv_in_batch_verify<DT>(m, bc, stream);

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   if (device == 0) {
  //     std::cout << "Update qkv time: " << elapsed << " ms\n";
  //   }

  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  // Compute attention
  tree_verify_attention<DT>(m, bc, static_cast<DT *>(m->attn_heads), stream);

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   if (device == 0) {
  //     std::cout << "Attn time: " << elapsed << " ms\n";
  //   }

  // Debug output:
  // {
  //   int size = m->local_hidden_size * bc->num_active_tokens();
  //   float *temp_output = new float[size];
  //   cudaDeviceSynchronize();
  //   cudaMemcpy(
  //       temp_output, m->attn_heads, size * sizeof(float),
  //       cudaMemcpyDeviceToHost);
  //   printf("Output (flashinfer attention) :");
  //   for (int i = 0; i < 1; ++i) {
  //     float temp = 0;
  //     for (int j = 0; j < m->local_hidden_size; ++j) {
  //       temp += temp_output[i * m->local_hidden_size + j];
  //     }
  //     printf("%.6f ", temp);
  //   }
  //   printf("\n");

  //   delete[] temp_output;
  // }
  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  int processed_tokens_in_batch = bc->num_active_tokens();

  compute_o_prod_bias(m,
                      bc,
                      shard_id,
                      output_ptr,
                      weight_ptr,
                      bias_ptr,
                      processed_tokens_in_batch,
                      stream);

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   if (device == 0) {
  //     std::cout << "Compute output proj time: " << elapsed << " ms\n";
  //   }
  // {
  //   int size = m->o_dim;
  //   DT *temp_output = new DT[size];
  //   cudaDeviceSynchronize();
  //   cudaMemcpy(
  //       temp_output, output_ptr + m->o_dim * (bc->num_active_tokens() -
  //       1), size * sizeof(DT), cudaMemcpyDeviceToHost);
  //   printf("Output :");
  //   for (int i = 0; i < size; ++i) {
  //     printf("%.6f ", static_cast<float>(temp_output[i]));
  //   }
  //   printf("\n");

  //   delete[] temp_output;
  // }
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

  //   int device;
  //   checkCUDA(cudaGetDevice(&device));
  //   cudaEvent_t t_start, t_end;
  //   cudaEventCreate(&t_start);
  //   cudaEventCreate(&t_end);
  //   cudaEventRecord(t_start, stream);

  // assert(input.data_type == weight.data_type);
  assert(input.data_type == output.data_type);
  if (use_bias) {
    assert(input.data_type == bias.data_type);
  }

  if (input.data_type == DT_HALF) {
    if (m->offload) {
      pre_build_weight<half>(m, weight, input.data_type, stream);
    }

    half const *bias_ptr =
        use_bias ? bias.get_half_ptr() : static_cast<half const *>(nullptr);
    Kernels::TreeIncMultiHeadAttention::inference_kernel<half>(
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
      pre_build_weight<float>(m, weight, input.data_type, stream);
    }
    float const *bias_ptr =
        use_bias ? bias.get_float_ptr() : static_cast<float const *>(nullptr);
    Kernels::TreeIncMultiHeadAttention::inference_kernel<float>(
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

  //   cudaEventRecord(t_end, stream);
  //   checkCUDA(cudaEventSynchronize(t_end));
  //   float elapsed = 0;
  //   checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
  //   cudaEventDestroy(t_start);
  //   cudaEventDestroy(t_end);
  //   if (device == 0) {
  //     std::cout << "TreeIncMultiHeadSelfAttention time: " << elapsed << "
  //     ms\n";
  //   }
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
                                    attn->hidden_size,
                                    attn->qk_dim,
                                    attn->v_dim,
                                    attn->o_dim,
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
                                    attn->offload,
                                    false),
      num_active_tokens(0) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(cudnnSetStream(handler.dnn, stream));

  // set attention constants
  handler.tree_verify_attention_metadata->set_enabled(true);
  handler.tree_verify_attention_metadata->set_num_q_heads(num_q_heads);
  handler.tree_verify_attention_metadata->set_num_kv_heads(num_kv_heads);
  handler.tree_verify_attention_metadata->set_head_dim(qk_dim);

  // allocate memory for the seqArray and reserve space
  {
    committed_token_infos =
        reinterpret_cast<BatchConfig::CommittedTokensInfo *>(
            reinterpret_cast<char *>(handler.batch_config_metadata) +
            sizeof(BatchConfig::tokensInfo) +
            sizeof(BatchConfig::requestsInfo) +
            sizeof(BatchConfig::request_available) +
            sizeof(BatchConfig::causalMask) +
            sizeof(BatchConfig::streamingCacheInfo));
    num_tokens_to_commit = reinterpret_cast<int *>(
        reinterpret_cast<char *>(committed_token_infos) +
        sizeof(BatchConfig::committed_tokens));
  }

  cudaStreamSynchronize(stream);
}

TreeIncMultiHeadSelfAttentionMeta::~TreeIncMultiHeadSelfAttentionMeta(void) {
  // delete static_cast<flashinfer::BatchPrefillHandler
  // *>(batch_prefill_handler);
}

}; // namespace FlexFlow

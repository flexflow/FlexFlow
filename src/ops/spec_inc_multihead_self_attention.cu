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
#include "flexflow/ops/spec_inc_multihead_self_attention.h"
#include "flexflow/utils/cuda_helper.h"

#include <sstream>
#include <stdexcept>

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;
using namespace Kernels::IncMultiHeadAttention;

namespace Kernels {
namespace SpecIncMultiHeadSelfAttention {

using flashinfer::BatchPrefillHandler;
using flashinfer::BatchPrefillWithPagedKVCacheWrapperDispatched;
using flashinfer::LogitsPostHook;
using flashinfer::MaskMode;
using flashinfer::paged_kv_t;
using flashinfer::PageStorage;
using flashinfer::PosEncodingMode;
using flashinfer::QKVLayout;

template <typename DT>
void tree_search_attention(SpecIncMultiHeadSelfAttentionMeta *m,
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
      m->handle.tree_search_attention_metadata->kv_indices,
      m->handle.tree_search_attention_metadata->kv_indptr,
      m->handle.tree_search_attention_metadata->kv_last_page_len);

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
    assert(m->handle.tree_search_attention_metadata->decode_handler_collections
                   .count(batch_size) != 0 &&
           "Handler is not initialized");
    handler = static_cast<BatchPrefillHandler *>(
        m->handle.tree_search_attention_metadata
            ->decode_handler_collections[batch_size]);
  } else {
    assert(m->handle.tree_search_attention_metadata->prompt_handler_collections
                   .count(batch_size) != 0 &&
           "Handler is not initialized");
    handler = static_cast<BatchPrefillHandler *>(
        m->handle.tree_search_attention_metadata
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
              m->handle.tree_search_attention_metadata->q_indptr,
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
              m->handle.tree_search_attention_metadata->q_indptr,
              /*q_offset=*/nullptr,
              paged_kv,
              m->handle.tree_search_attention_metadata->custom_mask,
              m->handle.tree_search_attention_metadata->qk_indptr,
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
                               "TreeSearchAttentionForwardKernel: " +
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
void inference_kernel(SpecIncMultiHeadSelfAttentionMeta *m,
                      BatchConfig const *bc,
                      int shard_id,
                      DT const *input_ptr,
                      DT const *weight_ptr,
                      DT *output_ptr,
                      DT const *bias_ptr,
                      cudaStream_t stream) {
  // phase 1: Compute QKV Projections of the batch
  compute_qkv(m,
              bc,
              shard_id,
              input_ptr,
              weight_ptr,
              static_cast<DT *>(m->devQKVProjArray),
              bias_ptr,
              stream);

  // phase 2: First maintain the streaming cache, because it need
  // pre-pos-encoding values
  if (m->streaming_cache) {
    // Move pre-pos-encoding cache to where took by attention
    update_kv_in_streaming_cache<DT>(m, bc, stream);
    // Apply pos-encoding to those k values
    apply_pos_encoding_to_streaming_proj<DT>(m, bc, stream);
    // Commit to the streaming cache
    if (bc->prompt_phase) {
      commit_kv<DT>(m, bc, stream);
    }
  }

  // phase 3: Take care of the batch
  {
    // Apply pos-encoding to the batch
    apply_pos_encoding_to_tokens_in_batch(
        m, bc, static_cast<DT *>(m->devQKVProjArray), stream);
    // Move the batch qkv values to where took by attention
    update_qkv_in_batch<DT>(m, bc, stream);
  }

  // phase 4: Attention computation
  tree_search_attention<DT>(m, bc, static_cast<DT *>(m->attn_heads), stream);

  // Debug output:
  //   int size = m->local_hidden_size * BatchConfig::max_tokens_per_batch();
  //   float *temp_output = new float[size];
  //   cudaDeviceSynchronize();
  //   cudaMemcpy(
  //       temp_output, m->attn_heads, size * sizeof(float),
  //       cudaMemcpyDeviceToHost);

  //   printf("Output: ");
  //   for (int i = 0; i < bc->num_tokens; ++i) {
  //     float temp = 0;
  //     for (int j = 0; j < m->local_hidden_size; ++j) {
  //       temp += temp_output[i * m->local_hidden_size + j];
  //     }
  //     printf("%.6f ", temp);
  //   }
  //   printf("\n");

  //   delete[] temp_output;

  // phase 5: Compute output production and bias together for all tokens
  int num_tokens = bc->num_active_tokens();
  compute_o_prod_bias(
      m, bc, shard_id, output_ptr, weight_ptr, bias_ptr, num_tokens, stream);
}

} // namespace SpecIncMultiHeadSelfAttention
} // namespace Kernels

/*static*/
void SpecIncMultiHeadSelfAttention::inference_kernel_wrapper(
    SpecIncMultiHeadSelfAttentionMeta *m,
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

  assert(input.data_type == weight.data_type);
  assert(input.data_type == output.data_type);
  if (use_bias) {
    assert(input.data_type == bias.data_type);
  }

  if (input.data_type == DT_HALF) {
    half const *bias_ptr =
        use_bias ? bias.get_half_ptr() : static_cast<half const *>(nullptr);
    Kernels::SpecIncMultiHeadSelfAttention::inference_kernel<half>(
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
    Kernels::SpecIncMultiHeadSelfAttention::inference_kernel<float>(
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
                                    TREE_SEARCH_MODE,
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
                                    DT_NONE,
                                    false,
                                    attn->streaming_cache) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  checkCUDNN(cudnnSetStream(handler.dnn, stream));

  // set attention constants
  handler.tree_search_attention_metadata->set_enabled(true);
  handler.tree_search_attention_metadata->set_num_q_heads(num_q_heads);
  handler.tree_search_attention_metadata->set_num_kv_heads(num_kv_heads);
  handler.tree_search_attention_metadata->set_head_dim(qk_dim);

  cudaStreamSynchronize(stream);
}

SpecIncMultiHeadSelfAttentionMeta::~SpecIncMultiHeadSelfAttentionMeta(void) {
  // for (auto &decode_handler: decode_handler_collections) {
  //   delete static_cast<flashinfer::BatchPrefillHandler
  //   *>(decode_handler.second);
  // }
  // for (auto &prompt_handler: prompt_handler_collections) {
  //   delete static_cast<flashinfer::BatchPrefillHandler
  //   *>(prompt_handler.second);
  // }
}

}; // namespace FlexFlow

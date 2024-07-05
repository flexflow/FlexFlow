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

#define DISPATCH_HEADDIM(head_dim, HEAD_DIM, ...)                              \
  switch (head_dim) {                                                          \
    case 64: {                                                                 \
      constexpr size_t HEAD_DIM = 64;                                          \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 128: {                                                                \
      constexpr size_t HEAD_DIM = 128;                                         \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 256: {                                                                \
      constexpr size_t HEAD_DIM = 256;                                         \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    default: {                                                                 \
      std::ostringstream err_msg;                                              \
      err_msg << "Unsupported head_dim: " << head_dim;                         \
      throw std::invalid_argument(err_msg.str());                              \
    }                                                                          \
  }

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

__device__ __forceinline__ size_t get_k_entry_offset(int const req_idx,
                                                     int const token_idx,
                                                     int const max_num_pages,
                                                     int const hidden_size) {
  return ((req_idx * max_num_pages + token_idx / kPagesize) * kPagesize * 2 +
          token_idx % kPagesize) *
         hidden_size;
}

__device__ __forceinline__ size_t get_v_entry_offset(int const req_idx,
                                                     int const token_idx,
                                                     int const max_num_pages,
                                                     int const hidden_size) {
  return ((req_idx * max_num_pages + token_idx / kPagesize) * kPagesize * 2 +
          kPagesize + token_idx % kPagesize) *
         hidden_size;
}

template <typename DT>
__global__ void
    update_qkv_cache_kernel(DT *devQKVProjArray,
                            half *qTmp_ptr,
                            half *kCache_ptr,
                            BatchConfig::PerTokenInfo const *tokenInfos,
                            BatchConfig::PerRequestInfo *request_infos,
                            int const max_num_pages,
                            int hidden_size,
                            int num_new_tokens) {
  int const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int const token_idx = thread_idx / hidden_size;
  int const offset = thread_idx % hidden_size;
  if (token_idx >= num_new_tokens) {
    return;
  }

  int const req_idx = tokenInfos[token_idx].request_index;
  int const token_abs_idx = tokenInfos[token_idx].abs_index_in_request;

  size_t from_idx = token_idx * QKV_WEIGHT_NUM * hidden_size;
  size_t to_k_idx = get_k_entry_offset(
             req_idx, token_abs_idx, max_num_pages, hidden_size),
         to_v_idx = get_v_entry_offset(
             req_idx, token_abs_idx, max_num_pages, hidden_size);

  // key and value cache should be stored interleaved
  kCache_ptr[to_k_idx + offset] =
      static_cast<half>(devQKVProjArray[from_idx + hidden_size + offset]);
  kCache_ptr[to_v_idx + offset] =
      static_cast<half>(devQKVProjArray[from_idx + hidden_size * 2 + offset]);
  qTmp_ptr[token_idx * hidden_size + offset] =
      static_cast<half>(devQKVProjArray[from_idx + offset]);
}

template <typename DT>
void update_qkv_cache(SpecIncMultiHeadSelfAttentionMeta const *m,
                      BatchConfig const *bc,
                      cudaStream_t stream) {
  // update the kv cache, compact the q array
  int num_new_tokens = bc->num_active_tokens();
  int parallelism = m->hidden_size * num_new_tokens;
  int const max_num_pages =
      (BatchConfig::max_sequence_length() +
       BatchConfig::max_spec_tree_token_num() + kPagesize - 1) /
      kPagesize;
  update_qkv_cache_kernel<<<GET_BLOCKS(parallelism),
                            min(CUDA_NUM_THREADS, parallelism),
                            0,
                            stream>>>(static_cast<DT *>(m->devQKVProjArray),
                                      static_cast<half *>(m->queryTmp),
                                      static_cast<half *>(m->keyCache),
                                      m->token_infos,
                                      m->request_infos,
                                      max_num_pages,
                                      m->hidden_size,
                                      num_new_tokens);
}

template <typename DT>
__global__ void produce_output_kernel(half const *input_ptr,
                                      DT *output_ptr,
                                      int parallelism) {
  CUDA_KERNEL_LOOP(idx, parallelism) {
    output_ptr[idx] = static_cast<DT>(input_ptr[idx]);
  }
}

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
  uint32_t const head_dim = m->qProjSize;
  uint32_t const batch_size = bc->num_active_requests();
  float const sm_scale =
      (*m->qk_prod_scaling) ? 1.0f / sqrt(m->kProjSize) : 1.0f;
  int32_t q_indptr_h[bc->max_requests_per_batch() + 1], kv_indptr_h[bc->max_requests_per_batch() + 1];
  q_indptr_h[0] = 0;
  kv_indptr_h[0] = 0;
  for (int req_idx = 0, indptr_idx = 0; req_idx < bc->max_requests_per_batch(); req_idx++) {
    if (bc->request_available[req_idx]) {
      int q_len = bc->requestsInfo[req_idx].num_tokens_in_batch;
      int kv_len = bc->requestsInfo[req_idx].num_tokens_in_batch +
                  bc->requestsInfo[req_idx].first_token_index_in_request;
      q_indptr_h[indptr_idx + 1] = q_indptr_h[indptr_idx] + q_len;
      kv_indptr_h[indptr_idx + 1] = kv_indptr_h[indptr_idx] + (kv_len + kPagesize - 1) / kPagesize;
      indptr_idx++;
    }
  }

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
       *kv = static_cast<half *>(m->keyCache),
       *o = static_cast<half *>(m->outputTmp);
  paged_kv_t<PageStorage::kIndices, QKVLayout::kNHD, half, int32_t> paged_kv(
      num_kv_heads,
      kPagesize,
      head_dim,
      batch_size,
      kv,
      m->handle.attention_metadata.kv_indices,
      m->handle.attention_metadata.kv_indptr,
      m->handle.attention_metadata.kv_last_page_len);

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

  BatchPrefillHandler *handler =
      static_cast<BatchPrefillHandler *>(m->batch_prefill_handler);
  handler->SetCUDAStream(stream);
  handler->BeginForward<half, int32_t>(m->workspace,
                                       m->workspace_size,
                                       static_cast<int32_t *>(q_indptr_h),
                                       static_cast<int32_t *>(kv_indptr_h),
                                       batch_size,
                                       num_q_heads,
                                       num_kv_heads,
                                       head_dim,
                                       kPagesize);

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

  DISPATCH_HEADDIM(
    head_dim, HEAD_DIM, {
      cudaError_t result;
      if (bc->prompt_phase) {
        result = BatchPrefillWithPagedKVCacheWrapperDispatched<
            PageStorage::kIndices,
            HEAD_DIM,
            LogitsPostHook::kNone,
            QKVLayout::kNHD,
            PosEncodingMode::kNone,
            false,
            MaskMode::kCausal,
            half,
            half,
            int32_t>(handler,
                      q,
                      m->handle.attention_metadata.q_indptr,
                      /*q_offset=*/nullptr,
                      paged_kv,
                      /*custom_mask=*/nullptr,
                      /*qk_indptr=*/nullptr,
                      o,
                      /*lse=*/nullptr,
                      num_q_heads,
                      /*logits_soft_cap=*/0.f,
                      sm_scale,
                      /*rope_scale=*/1.f,
                      /*rope_theta=*/static_cast<float>(1e4),
                      stream);
      } else {
        result = BatchPrefillWithPagedKVCacheWrapperDispatched<
            PageStorage::kIndices,
            HEAD_DIM,
            LogitsPostHook::kNone,
            QKVLayout::kNHD,
            PosEncodingMode::kNone,
            false,
            MaskMode::kCustom,
            half,
            half,
            int32_t>(handler,
                      q,
                      m->handle.attention_metadata.q_indptr,
                      /*q_offset=*/nullptr,
                      paged_kv,
                      m->handle.attention_metadata.custom_mask,
                      m->handle.attention_metadata.qk_indptr,
                      o,
                      /*lse=*/nullptr,
                      num_q_heads,
                      /*logits_soft_cap=*/0.f,
                      sm_scale,
                      /*rope_scale=*/1.f,
                      /*rope_theta=*/static_cast<float>(1e4),
                      stream);
      }
    if (result != cudaSuccess) {
      throw std::runtime_error(
          "Failed to run "
          "BatchPrefillWithPagedKVCacheWrapperDispatched" +
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

  {
    int parallelism = m->vProjSize * m->num_q_heads * bc->num_active_tokens();
    produce_output_kernel<<<GET_BLOCKS(parallelism),
                            min(CUDA_NUM_THREADS, parallelism),
                            0,
                            stream>>>(m->outputTmp, output_ptr, parallelism);
  }

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
  // phase 1: Implement kernel to compute KQV for input tokens

  // long long time_1 = Realm::Clock::current_time_in_microseconds(), time_2;
  compute_qkv_kernel(m,
                     bc,
                     shard_id,
                     input_ptr,
                     weight_ptr,
                     static_cast<DT *>(m->devQKVProjArray),
                     bias_ptr,
                     stream);
  // phase 2: Update key/val cache
  update_qkv_cache<DT>(m, bc, stream);

  // phase 3: Compute attention score
  // 3 kernels for pahse 3: matmul1 - softmax - matmal2
  tree_search_attention<DT>(m, bc, static_cast<DT *>(m->attn_heads), stream);

  // Debug output:
  //   int size = m->hidden_size * BatchConfig::max_tokens_per_batch();
  //   float *temp_output = new float[size];
  //   cudaDeviceSynchronize();
  //   cudaMemcpy(
  //       temp_output, m->attn_heads, size * sizeof(float),
  //       cudaMemcpyDeviceToHost);

  //   printf("Output: ");
  //   for (int i = 0; i < bc->num_tokens; ++i) {
  //     float temp = 0;
  //     for (int j = 0; j < m->hidden_size; ++j) {
  //       temp += temp_output[i * m->hidden_size + j];
  //     }
  //     printf("%.6f ", temp);
  //   }
  //   printf("\n");

  //   delete[] temp_output;

  // compute output production and bias together for all tokens
  int num_tokens = bc->num_active_tokens();

  compute_o_prod_bias(
      m, bc, shard_id, output_ptr, weight_ptr, bias_ptr, num_tokens, stream);
  // time_2 = Realm::Clock::current_time_in_microseconds();
  // std::cout << "SpecIncMultiHeadSelfAttention kernel time: "
  //           << (time_2 - time_1) << "us" << std::endl;
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

  {
    workspace_size = 32 * 1024 * 1024; // 32MB
    gpu_mem_allocator.create_legion_instance(
        flashinfer_reserve_inst, workspace_size);
    workspace = static_cast<void *>(
        gpu_mem_allocator.allocate_instance<char>(workspace_size));
    batch_prefill_handler =
        static_cast<void *>(new flashinfer::BatchPrefillHandler);
  }

  cudaStreamSynchronize(stream);
}

SpecIncMultiHeadSelfAttentionMeta::~SpecIncMultiHeadSelfAttentionMeta(void) {
  if (flashinfer_reserve_inst != Realm::RegionInstance::NO_INST) {
    flashinfer_reserve_inst.destroy();
  }
  delete static_cast<flashinfer::BatchPrefillHandler *>(batch_prefill_handler);
}

}; // namespace FlexFlow

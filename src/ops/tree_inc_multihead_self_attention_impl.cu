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
#include "flashinfer/attention_impl.cuh"

// This is for instantiating the template attention kernels
namespace flashinfer {

// warp_layout_literal[] = {
//   "WarpLayout::k4x1x2",
//   "WarpLayout::k4x1x1"
// }
// head_dim[] = {64, 128, 256};


/********** batch append instantiations for half precision **********/

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<PageStorage::kIndices, WarpLayout::k4x1x2, 64,
          LogitsPostHook::kNone, QKVLayout::kNHD, PosEncodingMode::kNone,
          false, MaskMode::kCustom, half, half, int32_t>(
  half* q, int32_t* request_indices, int32_t* q_tile_indices, int32_t* kv_tile_indices,
  int32_t* q_indptr, int32_t* q_offset,
  paged_kv_t<PageStorage::kIndices, QKVLayout::kNHD, half, int32_t> paged_kv, uint8_t* custom_mask,
  int32_t* qk_indptr, int32_t* o_indptr, half* o, half* tmp_v, float* tmp_s, float* lse,
  int32_t* merge_indptr, bool* block_valid_mask, int32_t* kv_chunk_size_ptr,
  uint32_t total_num_rows, uint32_t num_qo_heads, uint32_t padded_batch_size,
  float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<PageStorage::kIndices, WarpLayout::k4x1x2, 128,
          LogitsPostHook::kNone, QKVLayout::kNHD, PosEncodingMode::kNone,
          false, MaskMode::kCustom, half, half, int32_t>(
  half* q, int32_t* request_indices, int32_t* q_tile_indices, int32_t* kv_tile_indices,
  int32_t* q_indptr, int32_t* q_offset,
  paged_kv_t<PageStorage::kIndices, QKVLayout::kNHD, half, int32_t> paged_kv, uint8_t* custom_mask,
  int32_t* qk_indptr, int32_t* o_indptr, half* o, half* tmp_v, float* tmp_s, float* lse,
  int32_t* merge_indptr, bool* block_valid_mask, int32_t* kv_chunk_size_ptr,
  uint32_t total_num_rows, uint32_t num_qo_heads, uint32_t padded_batch_size,
  float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<PageStorage::kIndices, WarpLayout::k4x1x2, 256,
          LogitsPostHook::kNone, QKVLayout::kNHD, PosEncodingMode::kNone,
          false, MaskMode::kCustom, half, half, int32_t>(
  half* q, int32_t* request_indices, int32_t* q_tile_indices, int32_t* kv_tile_indices,
  int32_t* q_indptr, int32_t* q_offset,
  paged_kv_t<PageStorage::kIndices, QKVLayout::kNHD, half, int32_t> paged_kv, uint8_t* custom_mask,
  int32_t* qk_indptr, int32_t* o_indptr, half* o, half* tmp_v, float* tmp_s, float* lse,
  int32_t* merge_indptr, bool* block_valid_mask, int32_t* kv_chunk_size_ptr,
  uint32_t total_num_rows, uint32_t num_qo_heads, uint32_t padded_batch_size,
  float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<PageStorage::kIndices, WarpLayout::k4x1x1, 64,
          LogitsPostHook::kNone, QKVLayout::kNHD, PosEncodingMode::kNone,
          false, MaskMode::kCustom, half, half, int32_t>(
  half* q, int32_t* request_indices, int32_t* q_tile_indices, int32_t* kv_tile_indices,
  int32_t* q_indptr, int32_t* q_offset,
  paged_kv_t<PageStorage::kIndices, QKVLayout::kNHD, half, int32_t> paged_kv, uint8_t* custom_mask,
  int32_t* qk_indptr, int32_t* o_indptr, half* o, half* tmp_v, float* tmp_s, float* lse,
  int32_t* merge_indptr, bool* block_valid_mask, int32_t* kv_chunk_size_ptr,
  uint32_t total_num_rows, uint32_t num_qo_heads, uint32_t padded_batch_size,
  float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<PageStorage::kIndices, WarpLayout::k4x1x1, 128,
          LogitsPostHook::kNone, QKVLayout::kNHD, PosEncodingMode::kNone,
          false, MaskMode::kCustom, half, half, int32_t>(
  half* q, int32_t* request_indices, int32_t* q_tile_indices, int32_t* kv_tile_indices,
  int32_t* q_indptr, int32_t* q_offset,
  paged_kv_t<PageStorage::kIndices, QKVLayout::kNHD, half, int32_t> paged_kv, uint8_t* custom_mask,
  int32_t* qk_indptr, int32_t* o_indptr, half* o, half* tmp_v, float* tmp_s, float* lse,
  int32_t* merge_indptr, bool* block_valid_mask, int32_t* kv_chunk_size_ptr,
  uint32_t total_num_rows, uint32_t num_qo_heads, uint32_t padded_batch_size,
  float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<PageStorage::kIndices, WarpLayout::k4x1x1, 256,
          LogitsPostHook::kNone, QKVLayout::kNHD, PosEncodingMode::kNone,
          false, MaskMode::kCustom, half, half, int32_t>(
  half* q, int32_t* request_indices, int32_t* q_tile_indices, int32_t* kv_tile_indices,
  int32_t* q_indptr, int32_t* q_offset,
  paged_kv_t<PageStorage::kIndices, QKVLayout::kNHD, half, int32_t> paged_kv, uint8_t* custom_mask,
  int32_t* qk_indptr, int32_t* o_indptr, half* o, half* tmp_v, float* tmp_s, float* lse,
  int32_t* merge_indptr, bool* block_valid_mask, int32_t* kv_chunk_size_ptr,
  uint32_t total_num_rows, uint32_t num_qo_heads, uint32_t padded_batch_size,
  float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream);


/********** batch prefill instantiations for half precision **********/

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<PageStorage::kIndices, WarpLayout::k4x1x2, 64,
          LogitsPostHook::kNone, QKVLayout::kNHD, PosEncodingMode::kNone,
          false, MaskMode::kCausal, half, half, int32_t>(
  half* q, int32_t* request_indices, int32_t* q_tile_indices, int32_t* kv_tile_indices,
  int32_t* q_indptr, int32_t* q_offset,
  paged_kv_t<PageStorage::kIndices, QKVLayout::kNHD, half, int32_t> paged_kv, uint8_t* custom_mask,
  int32_t* qk_indptr, int32_t* o_indptr, half* o, half* tmp_v, float* tmp_s, float* lse,
  int32_t* merge_indptr, bool* block_valid_mask, int32_t* kv_chunk_size_ptr,
  uint32_t total_num_rows, uint32_t num_qo_heads, uint32_t padded_batch_size,
  float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<PageStorage::kIndices, WarpLayout::k4x1x2, 128,
          LogitsPostHook::kNone, QKVLayout::kNHD, PosEncodingMode::kNone,
          false, MaskMode::kCausal, half, half, int32_t>(
  half* q, int32_t* request_indices, int32_t* q_tile_indices, int32_t* kv_tile_indices,
  int32_t* q_indptr, int32_t* q_offset,
  paged_kv_t<PageStorage::kIndices, QKVLayout::kNHD, half, int32_t> paged_kv, uint8_t* custom_mask,
  int32_t* qk_indptr, int32_t* o_indptr, half* o, half* tmp_v, float* tmp_s, float* lse,
  int32_t* merge_indptr, bool* block_valid_mask, int32_t* kv_chunk_size_ptr,
  uint32_t total_num_rows, uint32_t num_qo_heads, uint32_t padded_batch_size,
  float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<PageStorage::kIndices, WarpLayout::k4x1x2, 256,
          LogitsPostHook::kNone, QKVLayout::kNHD, PosEncodingMode::kNone,
          false, MaskMode::kCausal, half, half, int32_t>(
  half* q, int32_t* request_indices, int32_t* q_tile_indices, int32_t* kv_tile_indices,
  int32_t* q_indptr, int32_t* q_offset,
  paged_kv_t<PageStorage::kIndices, QKVLayout::kNHD, half, int32_t> paged_kv, uint8_t* custom_mask,
  int32_t* qk_indptr, int32_t* o_indptr, half* o, half* tmp_v, float* tmp_s, float* lse,
  int32_t* merge_indptr, bool* block_valid_mask, int32_t* kv_chunk_size_ptr,
  uint32_t total_num_rows, uint32_t num_qo_heads, uint32_t padded_batch_size,
  float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<PageStorage::kIndices, WarpLayout::k4x1x1, 64,
          LogitsPostHook::kNone, QKVLayout::kNHD, PosEncodingMode::kNone,
          false, MaskMode::kCausal, half, half, int32_t>(
  half* q, int32_t* request_indices, int32_t* q_tile_indices, int32_t* kv_tile_indices,
  int32_t* q_indptr, int32_t* q_offset,
  paged_kv_t<PageStorage::kIndices, QKVLayout::kNHD, half, int32_t> paged_kv, uint8_t* custom_mask,
  int32_t* qk_indptr, int32_t* o_indptr, half* o, half* tmp_v, float* tmp_s, float* lse,
  int32_t* merge_indptr, bool* block_valid_mask, int32_t* kv_chunk_size_ptr,
  uint32_t total_num_rows, uint32_t num_qo_heads, uint32_t padded_batch_size,
  float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<PageStorage::kIndices, WarpLayout::k4x1x1, 128,
          LogitsPostHook::kNone, QKVLayout::kNHD, PosEncodingMode::kNone,
          false, MaskMode::kCausal, half, half, int32_t>(
  half* q, int32_t* request_indices, int32_t* q_tile_indices, int32_t* kv_tile_indices,
  int32_t* q_indptr, int32_t* q_offset,
  paged_kv_t<PageStorage::kIndices, QKVLayout::kNHD, half, int32_t> paged_kv, uint8_t* custom_mask,
  int32_t* qk_indptr, int32_t* o_indptr, half* o, half* tmp_v, float* tmp_s, float* lse,
  int32_t* merge_indptr, bool* block_valid_mask, int32_t* kv_chunk_size_ptr,
  uint32_t total_num_rows, uint32_t num_qo_heads, uint32_t padded_batch_size,
  float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<PageStorage::kIndices, WarpLayout::k4x1x1, 256,
          LogitsPostHook::kNone, QKVLayout::kNHD, PosEncodingMode::kNone,
          false, MaskMode::kCausal, half, half, int32_t>(
  half* q, int32_t* request_indices, int32_t* q_tile_indices, int32_t* kv_tile_indices,
  int32_t* q_indptr, int32_t* q_offset,
  paged_kv_t<PageStorage::kIndices, QKVLayout::kNHD, half, int32_t> paged_kv, uint8_t* custom_mask,
  int32_t* qk_indptr, int32_t* o_indptr, half* o, half* tmp_v, float* tmp_s, float* lse,
  int32_t* merge_indptr, bool* block_valid_mask, int32_t* kv_chunk_size_ptr,
  uint32_t total_num_rows, uint32_t num_qo_heads, uint32_t padded_batch_size,
  float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream);
} // namespace flashinfer

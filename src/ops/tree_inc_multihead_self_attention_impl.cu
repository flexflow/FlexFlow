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

// group_size[] = {1, 4, 8};
// head_dim[] = {64, 128, 256};

/********** prefill instantiations for half precision **********/

template cudaError_t SinglePrefillWithKVCacheDispatched<
  1, 64, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCausal, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched<
  1, 128, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCausal, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched<
  1, 256, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCausal, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched<
  4, 64, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCausal, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched<
  4, 128, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCausal, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched<
  4, 256, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCausal, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched<
  8, 64, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCausal, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched<
  8, 128, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCausal, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched<
  8, 256, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCausal, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);


/********** append instantiations for half precision **********/

template cudaError_t SinglePrefillWithKVCacheDispatched<
  1, 64, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCustom, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched<
  1, 128, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCustom, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched<
  1, 256, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCustom, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched<
  4, 64, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCustom, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched<
  4, 128, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCustom, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched<
  4, 256, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCustom, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched<
  8, 64, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCustom, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched<
  8, 128, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCustom, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched<
  8, 256, QKVLayout::kNHD, PosEncodingMode::kNone,
  false, MaskMode::kCustom, half, half>(
    half* q, half* k, half* v, float* custom_mask, half* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

  // template cudaError_t SinglePrefillWithKVCacheDispatched<
  //     1, 256, QKVLayout::kNHD, PosEncodingMode::kNone,
  //     false, MaskMode::kCustom, float, float>(
  //       float* q, float* k, float* v, float* custom_mask, float* o,
  //       float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
  //       float sm_scale, float rope_scale,
  //       float rope_theta, cudaStream_t stream);
} // namespace flashinfer

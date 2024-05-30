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
#include "flexflow/ops/kernels/decompress_kernels.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/utils/hip_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

namespace Kernels {

template <typename DT>
__global__ void decompress_int4_general_weights(char const *input_weight_ptr,
                                                DT *weight_ptr,
                                                int in_dim,
                                                int valueSize) {}

template <typename DT>
__global__ void decompress_int8_general_weights(char const *input_weight_ptr,
                                                DT *weight_ptr,
                                                int in_dim,
                                                int valueSize) {}

template <typename DT>
__global__ void decompress_int4_attention_weights(char *input_weight_ptr,
                                                  DT *weight_ptr,
                                                  int qProjSize,
                                                  int qSize,
                                                  int num_heads) {}

template <typename DT>
__global__ void decompress_int8_attention_weights(char *input_weight_ptr,
                                                  DT *weight_ptr,
                                                  int qProjSize,
                                                  int qSize,
                                                  int num_heads) {}

template __global__ void decompress_int4_general_weights<float>(
    char const *input_weight_ptr, float *weight_ptr, int in_dim, int valueSize);
template __global__ void decompress_int4_general_weights<half>(
    char const *input_weight_ptr, half *weight_ptr, int in_dim, int valueSize);
template __global__ void decompress_int8_general_weights<float>(
    char const *input_weight_ptr, float *weight_ptr, int in_dim, int valueSize);
template __global__ void decompress_int8_general_weights<half>(
    char const *input_weight_ptr, half *weight_ptr, int in_dim, int valueSize);
template __global__ void
    decompress_int4_attention_weights<float>(char *input_weight_ptr,
                                             float *weight_ptr,
                                             int qProjSize,
                                             int qSize,
                                             int num_heads);

template __global__ void
    decompress_int4_attention_weights<half>(char *input_weight_ptr,
                                            half *weight_ptr,
                                            int qProjSize,
                                            int qSize,
                                            int num_heads);

template __global__ void
    decompress_int8_attention_weights<float>(char *input_weight_ptr,
                                             float *weight_ptr,
                                             int qProjSize,
                                             int qSize,
                                             int num_heads);

template __global__ void
    decompress_int8_attention_weights<half>(char *input_weight_ptr,
                                            half *weight_ptr,
                                            int qProjSize,
                                            int qSize,
                                            int num_heads);

} // namespace Kernels
}; // namespace FlexFlow
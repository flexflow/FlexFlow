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
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

// declare Legion names
using Legion::coord_t;
using Legion::Memory;

namespace Kernels {

namespace Decommpress {

template <typename T1, typename T2>
__global__ void decompress_kernel(T1 *input_weight_ptr,
                                  T2 *weight_ptr,
                                  T2 *params,
                                  int group_size,
                                  int tensor_size) {
  CUDA_KERNEL_LOOP(i, tensor_size) {
    weight_ptr[i] = static_cast<T2>(input_weight_ptr);
    weight_ptr[i] *= scaling_ptr[(i % group_size) * 2];
    weight_ptr[i] += offet_ptr[(i % group_size) * 2 + 1];
  }
}

template <typename T1, typename T2>
void decompress_weight_bias(T1 *input_weight_ptr,
                            T2 *weight_ptr,
                            T2 *params,
                            int group_size,
                            int tensor_size) {

  // convert to DT, scaling, add offset;
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  int parallelism = tensor_size;
  decompress_kernel<<<GET_BLOCKS(parallelism),
                      min(CUDA_NUM_THREADS, parallelism),
                      0,
                      stream>>>(
      input_weight_ptr, weight_ptr, params, group_size);
}
} // namespace Decommpress
} // namespace Kernels
}; // namespace FlexFlow
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

#include "flexflow/ops/kernels/rms_norm_kernels.h"
#include "flexflow/ops/rms_norm.h"
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {
// declare Legion names
using Legion::coord_t;
using Legion::Domain;

RMSNormMeta::RMSNormMeta(FFHandler handler, RMSNorm const *rms)
    : OpMeta(handler, rms) {
  eps = rms->eps;
  // fixme
  checkCUDA(cudaMalloc(&mean_ptr, sizeof(float) * 1000));
}

namespace Kernels {
namespace RMSNorm {
template <typename T>
void forward_kernel_wrapper(RMSNormMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output,
                            GenericTensorAccessorR const &weight) {
}

namespace Internal {
/*static*/
template <typename T>
void forward_kernel(float const *input_ptr,
                             float const *weight_ptr,
                             float *output_ptr,
                             cudaStream_t stream) {

}
} // namespace Internal
} // namespace RMSNorm
} // namespace Kernels
} // namespace FlexFlow
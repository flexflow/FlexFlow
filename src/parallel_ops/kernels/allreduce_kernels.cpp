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

#include "flexflow/parallel_ops/kernels/allreduce_kernels.h"
#include "flexflow/utils/hip_helper.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

AllReduceMeta::AllReduceMeta(FFHandler handle, AllReduce const *reduct)
    : OpMeta(handle) {}

namespace Kernels {
namespace AllReduce {

void inference_kernel_wrapper(AllReduceMeta const *m,
                              BatchConfig const *bc,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorW const &output) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  assert(input.data_type == output.data_type);
  assert(input.domain == output.domain);
  assert(false && "To be implemented");
}

void forward_kernel_wrapper(AllReduceMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output) {
  hipStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  assert(input.data_type == output.data_type);
  assert(input.domain == output.domain);
  assert(false && "To be implemented");
}

void backward_kernel_wrapper(AllReduceMeta const *m,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad) {
  assert(false && "To be implemented");
}

} // namespace AllReduce
} // namespace Kernels
} // namespace FlexFlow

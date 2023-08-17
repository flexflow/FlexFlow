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
#include "flexflow/utils/cuda_helper.h"

namespace FlexFlow {

AllReduceMeta::AllReduceMeta(FFHandler handle, AllReduce const *reduct)
    : OpMeta(handle) {}

namespace Kernels {
namespace AllReduce {

void inference_kernel_wrapper(AllReduceMeta const *m,
                              BatchConfig const *bc,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  assert(input.data_type == output.data_type);
  assert(input.domain == output.domain);
  size_t hidden_dim_size = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  size_t num_elements = bc->num_tokens * hidden_dim_size;
#ifdef FF_USE_NCCL
  ncclDataType_t nccl_data_type = ff_to_nccl_datatype(input.data_type);
  checkNCCL(ncclAllReduce(input.ptr,
                          output.ptr,
                          num_elements,
                          nccl_data_type,
                          ncclSum,
                          m->handle.ncclComm,
                          stream));
#else
  assert(false && "Must enable FF_USE_NCCL to use AllReduce operators");
#endif
}

void forward_kernel_wrapper(AllReduceMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  assert(input.data_type == output.data_type);
  assert(input.domain == output.domain);
#ifdef FF_USE_NCCL
  ncclDataType_t nccl_data_type = ff_to_nccl_datatype(input.data_type);
  checkNCCL(ncclAllReduce(input.ptr,
                          output.ptr,
                          input.domain.get_volume(),
                          nccl_data_type,
                          ncclSum,
                          m->handle.ncclComm,
                          stream));
#else
  assert(false && "Must enable FF_USE_NCCL to use AllReduce operators");
#endif
}

void backward_kernel_wrapper(AllReduceMeta const *m,
                             GenericTensorAccessorW const &input_grad,
                             GenericTensorAccessorR const &output_grad) {
  assert(false && "To be implemented");
}

} // namespace AllReduce
} // namespace Kernels
} // namespace FlexFlow
